"""Configurable version of q1's movement evaluator, with zero-weight skipping."""

from collections.abc import Callable, Mapping
import math

from Agents.common.board import BoardState
from .movement import (
    FEATURE_NAMES, get_stack_maps, calc_material, calc_positioning,
    calc_efficiency, calc_alignment, calc_threat, calc_safety,
    calc_hunting, calc_control,
)

Evaluator = Callable[[BoardState, int, bool], float]


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return finite, nonnegative weights in the shared feature order."""
    if set(weights) != set(FEATURE_NAMES):
        raise ValueError(f"Weights must contain exactly {FEATURE_NAMES}")
    result: dict[str, float] = {}
    for name in FEATURE_NAMES:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} weight must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} weight must be finite and nonnegative")
        result[name] = float(value)
    return result


def make_evaluator(weights: Mapping[str, float]) -> Evaluator:
    """Capture an independent weight snapshot; preserve q1's summation order."""
    material, positioning, efficiency, alignment, threat, safety, hunting, control = (
        validate_weights(weights).values()
    )

    def evaluate(board: BoardState, root_color: int, is_red: bool) -> float:
        friendly, enemy = get_stack_maps(board, root_color)
        is_my_turn = (root_color == 1) == is_red
        score = 0.0
        if material > 0:
            score += calc_material(friendly, enemy) * material
        if positioning > 0:
            score += calc_positioning(friendly, enemy) * positioning
        if efficiency > 0:
            score += calc_efficiency(friendly, enemy) * efficiency
        if alignment > 0:
            score += calc_alignment(friendly) * alignment
        if threat > 0:
            score += calc_threat(friendly, enemy, is_my_turn) * threat
        if safety > 0:
            score += calc_safety(friendly, enemy, is_my_turn) * safety
        if hunting > 0:
            score += calc_hunting(friendly, enemy) * hunting
        if control > 0:
            score += calc_control(friendly, enemy) * control
        return score

    return evaluate
