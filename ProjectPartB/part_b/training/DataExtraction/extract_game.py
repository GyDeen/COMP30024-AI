"""Create a dataset of one Cascade game's board history and final result.

    python -m training.DataExtraction.extract_game logs/baseline-selfplay.log \
        --output training/DataExtraction/output/baseline-selfplay.npz

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
    """A recorded game event with its time, actor, payload and source line."""

    seconds: float
    actor: str
    name: str
    arguments: tuple[str, ...]
    line_number: int


def read_log(path: Path) -> list[LogEvent]:
    """Provide the recorded events needed to reconstruct a Cascade game.

    Args:
        path: Referee log file produced with the --logfile option.

    Returns:
        Events in file order, including their timestamps, actors, event names,
        arguments and original line numbers.

    Raises:
        ValueError: The log is empty or contains malformed event records.
    """
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
    """Identify the game action described by a logged action string.

    Args:
        text: A PLACE, MOVE, EAT or CASCADE string, such as 'MOVE(2-3, [→])'.

    Returns:
        The corresponding referee action, including its coordinate and any
        direction. Whether it is legal depends on the current board.

    Raises:
        ValueError: The action format, coordinate or direction is unsupported.
    """
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
    """Represent one board position in the dataset's agent-compatible format.

    Args:
        board: Referee board whose current position should be recorded.

    Returns:
        An independent uint8 array of 64 cells in row-major order. Empty cells
        are 0; Red stacks are 16 | height; Blue stacks are 32 | height.

    Raises:
        ValueError: A stack height cannot be represented by the cell encoding.
    """
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
    """Identify the game-ending condition responsible for the final result.

    Args:
        board: A terminal referee board, including its game history.

    Returns:
        'elimination', 'threefold_repetition', 'stalemate' or 'turn_limit'.
        If conditions overlap, the referee's outcome precedence applies.

    Raises:
        ValueError: No game-ending condition is present.
    """
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
    """Produce the board-history dataset and outcome for one completed game.

    Args:
        log_path: Complete referee log for a naturally finished game starting
            from the empty board. Games ending through agent errors are excluded.
        game_id: Identifier for this game; defaults to the log filename stem.

    Returns:
        A dictionary containing:
            game_id: Game identifier, as a string.
            winner: 'RED', 'BLUE' or 'DRAW'.
            ending_reason: The game-ending condition; see ending_reason().
            states: uint8 array of shape (N, 64), using snapshot_board()'s
                encoding. Row t is the board after t accepted actions. Row 0
                is empty and row N - 1 is terminal.

    Raises:
        ValueError: The identifier is empty, the log is incomplete or invalid,
            or the recorded result disagrees with the reconstructed game.
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
    """Save one extracted game's board history and outcome for later analysis.

    Args:
        game: Dictionary returned by extract_game(). Its entries are saved
            under the same names in the archive:
                game_id: Identifier linking these positions to their game.
                winner: Final outcome: 'RED', 'BLUE' or 'DRAW'.
                ending_reason: 'elimination', 'threefold_repetition',
                    'stalemate' or 'turn_limit'.
                states: uint8 array of shape (N, 64). Each row is a board in
                    agent encoding: 0 for empty, 16 | height for Red, and
                    32 | height for Blue. Rows run from the initial empty
                    board through every accepted action to the final board.
        output_path: Destination for a new .npz game archive. Game metadata is
            saved once for the whole sequence. Actions, timestamps, token
            totals and search-evaluation labels are not part of this dataset.

    Returns:
        None. The saved game is available through load_game(output_path).

    Raises:
        ValueError: The destination does not have a .npz extension.
        FileExistsError: The destination file already exists.
        OSError: The archive cannot be written.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("Output must use the .npz extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        np.savez_compressed(output, **game)


def load_game(path: Path) -> dict:
    """Retrieve a saved game's positions and outcome for feature extraction.

    Args:
        path: Game archive previously saved by write_game().

    Returns:
        The same four-field structure as extract_game(): game_id, winner and
        ending_reason as Python strings, and states as a uint8 array of shape
        (N, 64). Position order and cell encoding are preserved.
    """
    with np.load(path, allow_pickle=False) as saved:
        return {
            "game_id": saved["game_id"].item(),
            "winner": saved["winner"].item(),
            "ending_reason": saved["ending_reason"].item(),
            "states": saved["states"],
        }


def main() -> None:
    """Create one game archive from command-line input and report its outcome.

    Accepts a referee log path, a required --output .npz path and an optional
    --game-id. Reports the saved position count, winner and ending reason.
    Invalid input or an output failure produces exit status 2.
    """
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
