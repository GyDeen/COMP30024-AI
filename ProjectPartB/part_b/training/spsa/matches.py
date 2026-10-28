"""Evaluate named weights through the existing referee and data extractors."""

import json
import random
import uuid
from pathlib import Path
from typing import Any

from Agents.tune_agent.spsa.program import INITIAL_WEIGHTS
from GameGeneration.config import GameSpec, GenerationConfig
from GameGeneration.generate import run_match, save_json

from .config import CANDIDATE, SPSAConfig


def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score only naturally completed games, from the candidate's perspective.

    Any failed game or forfeit aborts the batch instead of silently biasing the
    objective. The raw failed result is retained and can be retried on resume.
    """
    if not results:
        raise ValueError("Cannot score an empty batch")
    counts = {"wins": 0, "draws": 0, "losses": 0}
    for row in results:
        if row["status"] != "completed":
            raise ValueError(f"Unscorable game: {row['status']}")
        winner, colour = row["winner"], row["candidate_colour"]
        if winner not in ("RED", "BLUE", "DRAW") or colour not in ("RED", "BLUE"):
            raise ValueError("Invalid winner or candidate colour")
        key = "draws" if winner == "DRAW" else "wins" if winner == colour else "losses"
        counts[key] += 1
    cpu_values = [row.get("candidate_action_cpu_seconds") for row in results]
    measured_cpu = sum(cpu_values) if all(value is not None for value in cpu_values) else None
    return {**counts, "games": len(results),
            "score": (counts["wins"] + 0.5 * counts["draws"]) / len(results),
            "candidate_action_cpu_seconds": measured_cpu}


def settings(weights: dict[str, float], config: SPSAConfig) -> dict[str, Any]:
    """Settings consumed by the playing agent, with no optimiser state."""
    return {"weights": weights, "move_seconds": config.move_seconds,
            "remaining_divisor": config.remaining_divisor, "max_depth": config.max_depth}


def schedule(config: SPSAConfig, iteration: int, validation: bool = False) -> list[dict[str, Any]]:
    """Identical seeds/openings for both probes, with adjacent colour swaps."""
    pool = config.validation_openings if validation else config.training_openings
    rng = random.Random(f"{config.seed}:{iteration}:{int(validation)}")
    rows: list[dict[str, Any]] = []
    for opponent_index, opponent in enumerate(config.opponents):
        for pair in range(config.pairs_per_opponent):
            opening_index = (iteration * config.pairs_per_opponent + pair) % len(pool)
            candidate_seed, opponent_seed = rng.getrandbits(32), rng.getrandbits(32)
            for colour in ("RED", "BLUE"):
                rows.append({"number": len(rows) + 1, "opponent": opponent,
                             "opponent_index": opponent_index, "candidate_colour": colour,
                             "candidate_seed": candidate_seed, "opponent_seed": opponent_seed,
                             "opening_index": opening_index, "opening": pool[opening_index]})
    return rows


def candidate_metrics(console: Path, colour: str) -> dict[str, Any]:
    """Read instrumented action CPU usage and depth, excluding opponent data."""
    measurements = []
    for line in console.read_text(encoding="utf-8").splitlines():
        if "TUNING_METRIC " in line:
            row = json.loads(line.split("TUNING_METRIC ", 1)[1])
            if row["colour"] == colour and row["turn_count"] >= 8:
                measurements.append(row)
    return {"candidate_action_cpu_seconds": (sum(row["cpu_seconds"] for row in measurements)
                                              if measurements else None),
            "candidate_measured_actions": len(measurements),
            "candidate_completed_depths": [row["completed_depth"] for row in measurements]}


def play_batch(weights: dict[str, float], config: SPSAConfig, output: Path,
               iteration: int, validation: bool = False) -> dict[str, Any]:
    """Run/resume paired games, preserving every failed attempt and raw log."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = schedule(config, iteration, validation)
    plan_path = output / "plan.json"
    previous_plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    batch_id = previous_plan["batch_id"] if previous_plan else uuid.uuid4().hex
    plan = {"batch_id": batch_id, "settings": settings(weights, config), "games": rows}
    if previous_plan is not None and previous_plan != plan:
        raise ValueError(f"Saved match plan differs: {plan_path}")
    save_json(plan_path, plan)
    runner_config = GenerationConfig(2, True, config.seed, config.cpu_seconds,
                                     config.memory_mb, config.wall_seconds, ())
    results: list[dict[str, Any]] = []
    for row in rows:
        game_dir = output / f"game-{row['number']:04d}"
        result_path = game_dir / "result.json"
        previous = json.loads(result_path.read_text()) if result_path.exists() else None
        if previous and previous["status"] == "completed":
            results.append(previous)
            continue
        game_dir.mkdir(exist_ok=True)
        attempt = 1
        while (game_dir / f"attempt-{attempt:03d}").exists():
            attempt += 1
        attempt_dir = game_dir / f"attempt-{attempt:03d}"
        (attempt_dir / "games").mkdir(parents=True)
        colour = row["candidate_colour"]
        candidate_red = colour == "RED"
        spec = GameSpec(
            f"{batch_id}-game-{row['number']:04d}-attempt-{attempt}", row["number"],
            row["opponent_index"], CANDIDATE if candidate_red else row["opponent"],
            row["opponent"] if candidate_red else CANDIDATE,
            row["candidate_seed"] if candidate_red else row["opponent_seed"],
            row["opponent_seed"] if candidate_red else row["candidate_seed"],
        )
        candidate_settings = json.dumps(settings(weights, config), allow_nan=False)
        control_settings = json.dumps(settings(INITIAL_WEIGHTS, config), allow_nan=False)
        environment = {
            "CASCADE_TUNING_RED": candidate_settings if candidate_red else control_settings,
            "CASCADE_TUNING_BLUE": control_settings if candidate_red else candidate_settings,
            "CASCADE_OPENING_ACTIONS": json.dumps(row["opening"]),
        }
        print(f"{output.name}: game {row['number']}/{len(rows)}, candidate {colour}", flush=True)
        result = run_match(spec, runner_config, attempt_dir, extra_env=environment)
        result.update(candidate_colour=colour, opening_index=row["opening_index"],
                      attempt_directory=str(attempt_dir.relative_to(output)))
        if result["status"] == "completed":
            result.update(candidate_metrics(attempt_dir / result["console"], colour))
        save_json(attempt_dir / "result.json", result)
        save_json(result_path, result)
        if result["status"] == "interrupted":
            raise KeyboardInterrupt
        if result["status"] != "completed":
            raise RuntimeError(f"{result['status']}: {result['error']}. See {attempt_dir}; resume to retry.")
        results.append(result)
    summary = score_results(results)
    save_json(output / "summary.json", summary)
    return summary


