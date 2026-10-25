"""Add the baseline's unweighted feature vectors to a saved Cascade game.

    python -m training.DataExtraction.extract_features \
        training/DataExtraction/output/baseline-selfplay.npz \
        --output training/DataExtraction/output/baseline-selfplay-features.npz

Each board keeps two feature vectors, ordered RED then BLUE. No evaluation
scores or numeric training targets are generated. The final game outcome is
retained so the trainer can choose its own target scale and player perspective.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

from Agents.common.board import BOARD_N, BLUE_FLAG, HEIGHT_MASK, RED_FLAG, BoardState
from referee.game.constants import MAX_TURNS, PLACEMENT_TURNS
from .extract_game import load_game, write_game


FEATURE_SOURCE = "Agents.agent_ab-q1.heuristic"
FEATURE_NAMES = (
    "material", "positioning", "efficiency", "alignment",
    "threaten_enemy", "safety", "hunting", "control",
)
PERSPECTIVES = ("RED", "BLUE")
SCHEMA_VERSION = 1
_heuristic = import_module(FEATURE_SOURCE)


def restore_board(state: np.ndarray, position_index: int) -> BoardState:
    """Restore a validated snapshot, including the agent's cached token totals."""
    board = BoardState()
    board.state = bytearray(state.tobytes())
    board.turn_count = position_index
    board.red_tokens = sum(board.get_height(i) for i in range(BOARD_N ** 2) if board.is_red(i))
    board.blue_tokens = sum(board.get_height(i) for i in range(BOARD_N ** 2) if board.is_blue(i))
    return board


def extract_board_features(
    board: BoardState, root_color: int, is_red_turn: bool,
) -> np.ndarray:
    """Return eight float32 features using the existing agent implementations.

    root_color uses the agent's convention: 1 for Red, 0 for Blue. is_red_turn
    describes the actual side to move, independently of the chosen perspective.
    All features are calculated, including ones with zero baseline weight.
    These are the movement evaluator's features, not the placement evaluator's.
    """
    if root_color not in (_heuristic.RED, _heuristic.BLUE):
        raise ValueError("root_color must be 1 (RED) or 0 (BLUE)")
    friendly, enemy = _heuristic.get_stack_maps(board, root_color)
    is_my_turn = (root_color == _heuristic.RED) == is_red_turn
    return np.asarray([
        _heuristic.calc_material(friendly, enemy),
        _heuristic.calc_positioning(friendly, enemy),
        _heuristic.calc_efficiency(friendly, enemy),
        _heuristic.calc_alignment(friendly),
        _heuristic.calc_threat(friendly, enemy, is_my_turn),
        _heuristic.calc_safety(friendly, enemy, is_my_turn),
        _heuristic.calc_hunting(friendly, enemy),
        _heuristic.calc_control(friendly, enemy),
    ], dtype=np.float32)


def extract_features(game: dict) -> dict:
    """Prepare one complete game's features, boards, context and raw outcome.

    Accepts load_game()'s dictionary. features[t, 0] describes Red's perspective
    on states[t]; features[t, 1] describes Blue's. The vectors have eight columns
    in feature_names order. Original snapshots and metadata are retained.

    Position context is kept separately from model inputs. Full repetition
    history remains available through states and position_index; repetition_state_count
    alone is insufficient to reconstruct the history of a search position.
    This validates the archive's structure; legal play and a correct terminal
    result are established by extract_game()'s referee replay.
    """
    for name in ("game_id", "winner", "ending_reason"):
        if not isinstance(game[name], str) or not game[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if game["winner"] not in (*PERSPECTIVES, "DRAW"):
        raise ValueError("winner must be RED, BLUE or DRAW")
    if game["ending_reason"] not in (
        "elimination", "threefold_repetition", "stalemate", "turn_limit",
    ):
        raise ValueError("Unsupported game-ending reason")
    states = np.asarray(game["states"])
    if states.dtype != np.uint8 or states.ndim != 2 or states.shape[1] != BOARD_N ** 2:
        raise ValueError("states must be a uint8 array with shape (N, 64)")
    if not PLACEMENT_TURNS < len(states) <= PLACEMENT_TURNS + MAX_TURNS + 1:
        raise ValueError("Expected a complete game including placement and terminal states")
    if np.any(states[0]):
        raise ValueError("The first snapshot must be the empty board")
    flags = states & 0xF0
    valid_cells = (states == 0) | (
        ((flags == RED_FLAG) | (flags == BLUE_FLAG)) & ((states & HEIGHT_MASK) > 0)
    )
    if not np.all(valid_cells):
        raise ValueError("Invalid cell encoding in states")

    state_count = len(states)
    position_index = np.arange(state_count, dtype=np.int32)
    is_red_turn = position_index % 2 == 0
    play_turns = np.maximum(position_index - PLACEMENT_TURNS, 0)
    features = np.empty((state_count, len(PERSPECTIVES), len(FEATURE_NAMES)), dtype=np.float32)
    repetition_state_count = np.zeros(state_count, dtype=np.int32)
    for index, state in enumerate(states):
        board = restore_board(state, index)
        for perspective, root_color in enumerate((_heuristic.RED, _heuristic.BLUE)):
            features[index, perspective] = extract_board_features(
                board, root_color, bool(is_red_turn[index]),
            )
        if index >= PLACEMENT_TURNS:
            # Compare exact boards with the same side to move, including the
            # current occurrence. The eighth placement starts play history.
            first = PLACEMENT_TURNS + (index - PLACEMENT_TURNS) % 2
            history = states[first:index + 1:2]
            repetition_state_count[index] = np.count_nonzero(np.all(history == state, axis=1))

    is_terminal = np.zeros(state_count, dtype=np.bool_)
    is_terminal[-1] = True
    profile_names = tuple(_heuristic.PROFILES)
    baseline_weights = np.asarray([
        [_heuristic.PROFILES[profile][name] for name in FEATURE_NAMES]
        for profile in profile_names
    ], dtype=np.float32)
    return {
        "game_id": game["game_id"],
        "winner": game["winner"],
        "ending_reason": game["ending_reason"],
        "states": states.copy(),
        "features": features,
        "feature_names": np.asarray(FEATURE_NAMES),
        "perspectives": np.asarray(PERSPECTIVES),
        "position_index": position_index,
        "side_to_move": np.where(is_red_turn, "RED", "BLUE"),
        "phase": np.where(position_index < PLACEMENT_TURNS, "PLACEMENT", "PLAY"),
        "play_turns": play_turns,
        "remaining_play_turns": MAX_TURNS - play_turns,
        "repetition_state_count": repetition_state_count,
        "is_terminal": is_terminal,
        "feature_source": FEATURE_SOURCE,
        "schema_version": SCHEMA_VERSION,
        "baseline_profile_names": np.asarray(profile_names),
        "baseline_weights": baseline_weights,
    }


def write_features(dataset: dict, output_path: Path) -> None:
    """Save all fields as plain arrays in a new .npz file; refuse overwrites."""
    write_game(dataset, output_path)


def load_features(path: Path) -> dict:
    """Load a feature archive without pickle, restoring scalar metadata types."""
    with np.load(path, allow_pickle=False) as saved:
        dataset = {name: saved[name] for name in saved.files}
    for name in ("game_id", "winner", "ending_reason", "feature_source", "schema_version"):
        dataset[name] = dataset[name].item()
    if dataset["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported feature archive schema version")
    return dataset


def main() -> None:
    """Enrich one extracted game with feature vectors and save a new archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", type=Path, help="Existing .npz game archive")
    parser.add_argument("--output", type=Path, required=True, help="New .npz feature archive")
    args = parser.parse_args()
    try:
        dataset = extract_features(load_game(args.game))
        write_features(dataset, args.output)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(2, f"Feature extraction failed: {error}\n")
    print(f"Saved features {dataset['features'].shape} and boards {dataset['states'].shape}")
    print(f"Perspectives: RED, BLUE; result: {dataset['winner']}; output: {args.output}")


if __name__ == "__main__":
    main()
