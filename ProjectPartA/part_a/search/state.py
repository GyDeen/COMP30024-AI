from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Node:
    board: dict[Coord, CellState]  
    path: list[Action]             
    g_cost: int                    # Actual cost from start (number of actions)
    h_cost: int                    # Heuristic estimate to the goal

    @property
    def f_cost(self) -> int:       # Total priority for A* (g + h)
        return self.g_cost + self.h_cost