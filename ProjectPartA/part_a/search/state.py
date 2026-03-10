from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Node:
    board: dict[Coord, CellState]  # The current board configuration [cite: 444, 445]
    path: list[Action]             # The list of actions taken from the start [cite: 459]
    g_cost: int                    # Actual cost from start (number of actions) [cite: 419]
    h_cost: int                    # Heuristic estimate to the goal [cite: 591]

    @property
    def f_cost(self) -> int:       # Total priority for A* (g + h)
        return self.g_cost + self.h_cost