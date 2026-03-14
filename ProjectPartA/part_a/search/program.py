# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .utils import render_board
from .action import cascade
import heapq
from .state import Node


def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to `CellState` instances (each with a `.color` and
            `.height` attribute).

    Returns:
        A list of actions (MoveAction, EatAction, or CascadeAction), or `None`
        if no solution is possible.
    """
    
    start_state = tuple(sorted(board.items()))
    start_node = Node(board=start_state, path=[], g_cost=0, h_cost=0)
    
    # The pq for next potential states to explore, ordered by f_cost (g_cost + h_cost)
    priority_queue = []
    # To break ties in the priority queue
    entry_count = 0  
    actions = list[Action] = []
    # To track visited board states and avoid cycles
    visited = set()  
    
    heapq.heappush(priority_queue, (start_node.f_cost, entry_count, start_node))
    entry_count += 1
    print(render_board(board, ansi=True))
    

    
    return actions


def no_blue(board_state: tuple) -> bool:
    """
    Check if there are no blue cells left on the board.
    """
    for coord, cell_state in board_state:
        if cell_state.color == PlayerColor.BLUE:
            return False
    return True



