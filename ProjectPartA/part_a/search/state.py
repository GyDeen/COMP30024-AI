from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Node:
    # Translate the board state into hashable format
    board: tuple[tuple[Coord, CellState], ...]
    path: list[Action]             
    g_cost: int                    # Actual cost from start (number of actions)
    h_cost: int                    # Heuristic estimate to the goal

    @property
    # Total priority for A* (g + h)
    def f_cost(self) -> int:       
        return self.g_cost + self.h_cost
    
    
    def __hash__(self):
        return hash(self.board_state)
    
    
    # Equality is based on the board state, not the path or costs
    def __eq__(self, other):
        return self.board_state == other.board_state