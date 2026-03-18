from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Node:
    # Immutable key representation of the board (see encode_board in program.py)
    board_key: tuple[int, ...]
    # Mutable board dictionary used for expanding successors
    board: dict[Coord, CellState]
    path: list[Action]
    g_cost: int                    # Actual cost from start (number of actions)
    h_cost: int                    # Heuristic estimate to the goal

    @property
    # Total priority for A* (g + h)
    def f_cost(self) -> int:       
        return self.g_cost + self.h_cost
    
    
    def __hash__(self):
        return hash(self.board_key)
    
    
    # Equality is based on the board state, not the path or costs
    def __eq__(self, other):
        return self.board_key == other.board_key
