"""Generate referee games, raw outcomes and compact position datasets.

    python -m GameGeneration.generate --config GameGeneration/config.toml \
        --output GameGeneration/runs/pilot-001
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from training.DataExtraction.extract_game import extract_game, write_game
from training.DataExtraction.extract_features import extract_features, write_features
from .config import GameSpec, GenerationConfig, load_config, plan_games


PART_B_ROOT = Path(__file__).resolve().parents[1]


def save_json(path: Path, value) -> None:
    """Persist run metadata or progress as a complete JSON document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_match(process: subprocess.Popen) -> None:
    """Stop a stalled or interrupted match, including both agent processes.

    The process must be a referee started in its own POSIX process session.
    Returns after its referee process has exited.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def read_outcome(log_path: Path, return_code: int, game_id: str) -> tuple[dict, dict | None]:
    """Return the raw match outcome and, for a valid normal game, its dataset.

    Outcome fields include status, winner, ending_reason, accepted turns and
    any error. Status is completed, forfeit, referee_error or invalid_log.
    winner is RED, BLUE, DRAW or null. reported_winner retains a logged result
    even when validation fails. Only completed games return a training dataset;
    forfeits remain recorded results but are excluded from training positions.
    """
    outcome = {"status": "referee_error", "winner": None,
               "reported_winner": None, "ending_reason": None,
               "turns": None, "error": None}
    if not log_path.is_file():
        outcome["error"] = "The referee did not produce a game log"
        return outcome, None
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        outcome.update(status="invalid_log", error=f"Cannot read game log: {error}")
        return outcome, None
    outcome["turns"] = 0
    endings, errors = [], []
    fatal = False
    for line in lines:
        fields = line.split("\t", 3)
        # Error descriptions can contain continuation lines. Raw logs and
        # console output retain those details; only event lines count here.
        if len(fields) < 3 or not fields[0].startswith("T") or fields[1] != "referee":
            continue
        event = fields[2]
        if event == "board_update":
            outcome["turns"] += 1
        elif event == "game_end" and len(fields) == 4:
            endings.append(fields[3])
        elif event == "player_error" and len(fields) == 4:
            errors.append(fields[3])
        elif event == "unhandled_error":
            fatal = True
    winners = {"winner:RED": "RED", "winner:BLUE": "BLUE", "winner:None": "DRAW"}
    if len(endings) == 1:
        outcome["reported_winner"] = winners.get(endings[0])
    if return_code != 0 or fatal:
        outcome["error"] = f"Referee failed with exit code {return_code}; see console output"
        return outcome, None
    if len(endings) != 1 or outcome["reported_winner"] is None:
        outcome.update(status="invalid_log", error="Missing or invalid final game result")
        return outcome, None
    if errors:
        if outcome["reported_winner"] == "DRAW":
            outcome.update(status="invalid_log", error="A player-error forfeit cannot be a draw")
            return outcome, None
        message = "\n".join(errors)
        if "ILLEGAL ACTION:" in message:
            reason = "illegal_action"
        elif "exceeded available time" in message:
            reason = "cpu_time_limit"
        elif "exceeded space limit" in message:
            reason = "memory_limit"
        elif "recv time limit" in message:
            reason = "agent_response_timeout"
        else:
            reason = "agent_error"
        outcome.update(status="forfeit", winner=outcome["reported_winner"],
                       ending_reason=reason, error=message)
        return outcome, None
    try:
        game = extract_game(log_path, game_id)
    except (ValueError, OSError) as error:
        outcome.update(status="invalid_log", error=str(error))
        return outcome, None
    outcome.update(status="completed", winner=game["winner"],
                   ending_reason=game["ending_reason"], turns=len(game["states"]) - 1)
    return outcome, game


def run_match(spec: GameSpec, config: GenerationConfig, output_dir: Path) -> dict:
    """Generate one scheduled match and return its durable result record.

    Saves its referee log and console output. A valid naturally finished game
    also receives a compact .npz dataset. The record identifies agents, colours,
    seeds, raw outcome, wall duration, return code and relative artifact paths.
    No outcome scores or training rewards are calculated. A wall timeout or
    interruption stops the entire match process group and preserves diagnostics.
    """
    stem = f"game-{spec.number:06d}"
    log_path = output_dir / "games" / f"{stem}.log"
    console_path = output_dir / "games" / f"{stem}.console.txt"
    dataset_path = output_dir / "games" / f"{stem}.npz"
    features_path = output_dir / "features" / f"{stem}.npz"
    environment = dict(os.environ)
    environment.update({
        "CASCADE_RED_AGENT": spec.red_agent,
        "CASCADE_BLUE_AGENT": spec.blue_agent,
        "CASCADE_RED_SEED": str(spec.red_seed),
        "CASCADE_BLUE_SEED": str(spec.blue_seed),
    })
    command = [sys.executable, "-m", "referee",
               "GameGeneration.seeded_agent", "GameGeneration.seeded_agent",
               "--time", str(config.cpu_seconds), "--space", str(config.memory_mb),
               "--verbosity", "1", "--colourless", "--logfile", str(log_path)]
    result = {**asdict(spec), "status": "launch_error", "winner": None,
              "reported_winner": None, "ending_reason": None, "turns": None,
              "return_code": None, "error": None,
              "log": str(log_path.relative_to(output_dir)),
              "console": str(console_path.relative_to(output_dir)),
              "positions": None, "features": None}
    start = time.monotonic()
    try:
        with console_path.open("x", encoding="utf-8") as console:
            process = subprocess.Popen(command, cwd=PART_B_ROOT, env=environment,
                                       stdout=console, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=config.wall_seconds)
            except subprocess.TimeoutExpired:
                stop_match(process)
                result.update(status="wall_timeout", error="Match exceeded wall_seconds")
            except KeyboardInterrupt:
                stop_match(process)
                result.update(status="interrupted", error="Generation interrupted by the user")
            else:
                outcome, game = read_outcome(log_path, process.returncode, spec.game_id)
                result.update(outcome)
                if game is not None:
                    try:
                        write_game(game, dataset_path)
                        result["positions"] = str(dataset_path.relative_to(output_dir))
                        write_features(extract_features(game), features_path)
                        result["features"] = str(features_path.relative_to(output_dir))
                    except (ValueError, OSError) as error:
                        result.update(status="dataset_error", error=str(error))
            result["return_code"] = process.returncode
    except OSError as error:
        result["error"] = str(error)
    result["wall_seconds"] = round(time.monotonic() - start, 6)
    return result


def summarize(results: list[dict], scheduled: int, interrupted: bool) -> dict:
    """Return raw progress and outcome counts, without assigning scores.

    Natural-game outcomes and forfeits are counted separately. Runner failures
    and interruptions are not converted into wins, losses or draws.
    """
    statuses = Counter(row["status"] for row in results)
    natural = Counter(row["winner"] for row in results if row["status"] == "completed")
    return {
        "scheduled_games": scheduled,
        "attempted_games": len(results),
        "status_counts": dict(statuses),
        "completed_outcomes": {key: natural[key] for key in ["RED", "BLUE", "DRAW"]},
        "interrupted": interrupted,
    }


def generate(config_path: Path, output_dir: Path) -> dict:
    """Generate a configured batch in a new directory and return its summary.

    Saves config.toml, manifest.json, schedule.json, results.jsonl, summary.json
    and per-game artifacts. Results are flushed after each game and progress is
    saved between games. Existing output directories are never reused. The
    first version runs sequentially on macOS/Linux to avoid CPU contention.
    """
    if os.name != "posix":
        raise ValueError("This runner currently supports macOS and Linux")
    config = load_config(config_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "games").mkdir()
    run_id = f"{output_dir.name}-{uuid.uuid4().hex[:8]}"
    games = plan_games(config, run_id)
    shutil.copyfile(config_path, output_dir / "config.toml")
    save_json(output_dir / "manifest.json", {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": "config.toml",
    })
    save_json(output_dir / "schedule.json", [asdict(game) for game in games])
    results = []
    interrupted = False
    save_json(output_dir / "summary.json", summarize(results, len(games), interrupted))
    with (output_dir / "results.jsonl").open("x", encoding="utf-8") as output:
        try:
            for game in games:
                print(f"[{game.number}/{len(games)}] Red={game.red_agent}; Blue={game.blue_agent}", flush=True)
                result = run_match(game, config, output_dir)
                output.write(json.dumps(result, allow_nan=False) + "\n")
                output.flush()
                results.append(result)
                interrupted = result["status"] == "interrupted"
                save_json(output_dir / "summary.json", summarize(results, len(games), interrupted))
                print(f"  {result['status']}: {result['winner']}; {result['wall_seconds']:.2f}s", flush=True)
                if interrupted:
                    break
        except KeyboardInterrupt:
            interrupted = True
    summary = summarize(results, len(games), interrupted)
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Generate a batch or preview its assignments with --dry-run.

    --config selects the experiment file. --output selects a new run directory;
    without it, a dated directory under GameGeneration/runs is used. A normal
    exit means all scheduled games completed naturally. Exit 1 indicates match
    failures, exit 2 invalid configuration/output, and exit 130 interruption.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--output", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--dry-run", action="store_true", help="Show assignments and seeds without running games")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.dry_run:
            print(json.dumps([asdict(game) for game in plan_games(config, "preview")], indent=2))
            return
        output = args.output or Path(__file__).with_name("runs") / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        summary = generate(args.config, output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Generation failed: {error}\n")
    print(f"Run saved to {output.resolve()}")
    print(json.dumps(summary, indent=2))
    if summary["interrupted"]:
        parser.exit(130)
    if summary["attempted_games"] != summary["status_counts"].get("completed", 0):
        parser.exit(1)


if __name__ == "__main__":
    main()
