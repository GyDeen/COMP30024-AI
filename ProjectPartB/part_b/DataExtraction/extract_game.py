"""Replay one referee log into a compact NumPy archive for the whole game.

    python -m DataExtraction.extract_game logs/baseline-selfplay.log \
        --output DataExtraction/output/baseline-selfplay.npz

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

from referee.game import (
    Action, Board, CascadeAction, Coord, Direction, EatAction,
    IllegalActionException, MoveAction, PlaceAction, PlayerColor,
)
from Agents.common.board import BOARD_N, RED_FLAG, BLUE_FLAG, HEIGHT_MASK


@dataclass(frozen=True)
class LogEvent:
    """One parsed TSV line, retaining its line number for useful errors."""

    seconds: float
    actor: str
    name: str
    arguments: tuple[str, ...]
    line_number: int


def read_log(path: Path) -> list[LogEvent]:
    """Read the referee's --logfile output, rejecting malformed event lines."""
    argument_counts = {
        "game_begin": 0, "initialising": 0, "turn_begin": 1,
        "turn_end": 2, "board_update": 0, "game_end": 1,
        "player_error": 1, "unhandled_error": 1,
    }
    events = []
    previous_seconds = 0.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3 or not fields[0].startswith("T"):
            raise ValueError(f"Line {line_number}: expected a referee TSV event")
        timestamp, actor, name, *arguments = fields
        try:
            seconds = float(timestamp[1:])
        except ValueError as error:
            raise ValueError(f"Line {line_number}: invalid timestamp") from error
        if not math.isfinite(seconds) or seconds < previous_seconds:
            raise ValueError(f"Line {line_number}: invalid or decreasing timestamp")
        if actor not in {"referee", "RED", "BLUE"}:
            raise ValueError(f"Line {line_number}: unknown actor {actor!r}")
        if name not in argument_counts or len(arguments) != argument_counts[name]:
            raise ValueError(f"Line {line_number}: unknown event or wrong argument count")
        referee_event = name not in {"initialising", "turn_begin", "turn_end"}
        if referee_event != (actor == "referee"):
            raise ValueError(f"Line {line_number}: wrong actor for {name}")
        events.append(LogEvent(seconds, actor, name, tuple(arguments), line_number))
        previous_seconds = seconds
    if not events:
        raise ValueError("The log is empty")
    return events


def parse_action(text: str) -> Action:
    """Convert logged action text into a referee action without using eval()."""
    match = re.fullmatch(
        r"(PLACE|MOVE|EAT|CASCADE)\(([0-7])-([0-7])(?:, \[([↑↓←→])\])?\)",
        text,
    )
    if match is None:
        raise ValueError(f"Unsupported action text: {text!r}")
    name, row, column, arrow = match.groups()
    coord = Coord(int(row), int(column))
    if name == "PLACE":
        if arrow is not None:
            raise ValueError("PLACE must not have a direction")
        return PlaceAction(coord)
    if arrow is None:
        raise ValueError(f"{name} requires a direction")
    direction = {"↑": Direction.Up, "↓": Direction.Down,
                 "←": Direction.Left, "→": Direction.Right}[arrow]
    action_class = {"MOVE": MoveAction, "EAT": EatAction, "CASCADE": CascadeAction}[name]
    return action_class(coord, direction)


def snapshot_board(board: Board) -> np.ndarray:
    """Copy the board to 64 bytes in the baseline agent's cell encoding."""
    state = np.zeros(BOARD_N * BOARD_N, dtype=np.uint8)
    for row in range(BOARD_N):
        for column in range(BOARD_N):
            cell = board[Coord(row, column)]
            if cell.color is not None:
                if not 1 <= cell.height <= HEIGHT_MASK:
                    raise ValueError("Stack height does not fit the baseline encoding")
                flag = RED_FLAG if cell.color == PlayerColor.RED else BLUE_FLAG
                state[row * BOARD_N + column] = flag | cell.height
    return state


def ending_reason(board: Board) -> str:
    """Explain a natural ending using the referee's outcome precedence."""
    if board.red_tokens == 0 or board.blue_tokens == 0:
        return "elimination"
    if board._is_threefold_repetition():
        return "threefold_repetition"
    # The referee exposes only private helpers for repetition and stalemate.
    # Reuse them instead of duplicating its game rules or position history.
    if not board._has_legal_actions():
        return "stalemate"
    if board.turn_limit_reached:
        return "turn_limit"
    raise ValueError("The log ended before the board reached a game-ending condition")


def extract_game(log_path: Path, game_id: str | None = None) -> dict:
    """Validate and replay a complete, naturally finished game.

    states[0] is the empty board; states[t] follows t accepted actions.
    The last state is terminal. Aborted games and forfeits are rejected.
    """
    log_path = Path(log_path)
    game_id = log_path.stem if game_id is None else game_id
    if not game_id.strip():
        raise ValueError("game_id must not be empty")
    events = read_log(log_path)
    if events[0].name != "game_begin":
        raise ValueError("The first event must be game_begin")
    board = Board()
    states = [snapshot_board(board)]
    initialised = set()
    pending_turn = None
    awaiting_board_update = False
    finished = False

    for event in events[1:]:
        try:
            if finished:
                raise ValueError("Found an event after game_end")
            if event.name in {"player_error", "unhandled_error"}:
                raise ValueError(f"Game contains {event.name}: {event.arguments[0]}")
            if awaiting_board_update and event.name != "board_update":
                raise ValueError("An accepted action must be followed by board_update")

            if event.name == "initialising":
                if board.turn_count or pending_turn or event.actor in initialised:
                    raise ValueError("Unexpected or duplicate initialising event")
                initialised.add(event.actor)
            elif event.name == "turn_begin":
                if initialised != {"RED", "BLUE"} or pending_turn or board.game_over:
                    raise ValueError("Unexpected turn_begin event")
                if event.actor != board.turn_color.name:
                    raise ValueError("Player colour does not match the side to move")
                if int(event.arguments[0]) != board.turn_count + 1:
                    raise ValueError("Missing, duplicated, or out-of-order turn")
                pending_turn = event
            elif event.name == "turn_end":
                if pending_turn is None or event.actor != pending_turn.actor:
                    raise ValueError("turn_end has no matching turn_begin")
                if event.arguments[0] != pending_turn.arguments[0]:
                    raise ValueError("turn_end has a different turn number")
                action = parse_action(event.arguments[1])
                board.apply_action(action)
                awaiting_board_update = True
            elif event.name == "board_update":
                if not awaiting_board_update:
                    raise ValueError("board_update has no preceding action")
                states.append(snapshot_board(board))
                pending_turn = None
                awaiting_board_update = False
            elif event.name == "game_end":
                if pending_turn or not board.game_over:
                    raise ValueError("game_end does not match a completed game")
                winner = board.winner_color
                expected = f"winner:{winner}"
                if event.arguments[0] != expected:
                    raise ValueError(f"Logged result disagrees with replay; expected {expected}")
                reason = ending_reason(board)
                finished = True
            else:
                raise ValueError(f"Unexpected event: {event.name}")
        except (ValueError, IllegalActionException) as error:
            raise ValueError(f"Line {event.line_number}: {error}") from error

    if not finished:
        raise ValueError("Incomplete log: missing game_end")
    return {
        "game_id": game_id,
        "winner": winner.name if winner is not None else "DRAW",
        "ending_reason": reason,
        "states": np.stack(states),
    }


def write_game(game: dict, output_path: Path) -> None:
    """Save arrays and three scalar strings once, without pickled objects."""
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("Output must use the .npz extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        np.savez_compressed(output, **game)


def load_game(path: Path) -> dict:
    """Load the compact game; return metadata as ordinary Python strings."""
    with np.load(path, allow_pickle=False) as saved:
        return {
            "game_id": saved["game_id"].item(),
            "winner": saved["winner"].item(),
            "ending_reason": saved["ending_reason"].item(),
            "states": saved["states"],
        }


def main() -> None:
    """Provide a small command-line entry point for one game at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Complete referee --logfile output")
    parser.add_argument("--output", type=Path, required=True, help="New .npz file")
    parser.add_argument("--game-id", help="Unique game ID; defaults to the log filename stem")
    args = parser.parse_args()
    try:
        game = extract_game(args.log, args.game_id)
        write_game(game, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Extraction failed: {error}\n")
    count = len(game["states"])
    print(f"Saved {count} positions from {count - 1} actions to {args.output}")
    print(f"Result: {game['winner']}; ending: {game['ending_reason']}")


if __name__ == "__main__":
    main()
