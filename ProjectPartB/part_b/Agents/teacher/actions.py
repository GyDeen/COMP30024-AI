"""Translate the bundled teacher's 832 action indices to referee actions."""

from referee.game import (
    Action, CascadeAction, Coord, Direction, EatAction, MoveAction, PlaceAction,
)


DIRECTIONS = (Direction.Up, Direction.Down, Direction.Left, Direction.Right)
ACTION_TYPES = (MoveAction, EatAction, CascadeAction)


def action_from_index(index: int) -> Action:
    if type(index) is not int or not 0 <= index < 832:
        raise ValueError(f"Teacher action index out of range: {index!r}")
    if index < 64:
        return PlaceAction(Coord(index // 8, index % 8))
    kind, offset = divmod(index - 64, 256)
    square, direction = divmod(offset, 4)
    return ACTION_TYPES[kind](Coord(square // 8, square % 8), DIRECTIONS[direction])


def action_to_index(action: Action) -> int:
    square = action.coord.r * 8 + action.coord.c
    if isinstance(action, PlaceAction):
        return square
    for kind, action_type in enumerate(ACTION_TYPES):
        if isinstance(action, action_type):
            return 64 + kind * 256 + square * 4 + DIRECTIONS.index(action.direction)
    raise ValueError(f"Unknown action: {action!r}")
