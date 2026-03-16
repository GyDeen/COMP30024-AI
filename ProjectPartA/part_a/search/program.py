# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor
from .utils import render_board
from .action import cascade, eat, move
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
    actions = []
    # To track visited board states and avoid cycles
    visited = set()  
    
    heapq.heappush(priority_queue, (start_node.f_cost, entry_count, start_node))
    current_node = start_node
    entry_count += 1
    print(render_board(board, ansi=True))
    
    while priority_queue:
        f_val, _, next_node = heapq.heappop(priority_queue)
        
        # If next board already visited, skip
        if next_node.board in visited:
            continue
        visited.add(next_node.board)
        
        # There is no blue cell, we find the solution
        if no_blue(next_node.board):
            return next_node.path
        
        # Convert the board back to dict format for easier manipulation
        board_dict = dict(next_node.board)
        # Generate all possible actions from the current board state
        for coord, cell_state in board_dict.items():
            if cell_state.color == PlayerColor.RED:
                for direction in Direction:
                    # Check if the move is within bounds before creating the action
                    if is_within_bounds(coord, direction):
                        target_coord = coord + direction
                        if board_dict.get(target_coord) is None or board_dict[target_coord].color == PlayerColor.RED:
                            move_action = MoveAction(coord, direction)
                            new_board = move(board_dict, move_action)
                            new_path = next_node.path + [move_action]
                            new_node = Node(board=tuple(sorted(new_board.items())), path=new_path, g_cost=next_node.g_cost + 1, h_cost=0)
                            heapq.heappush(priority_queue, (new_node.f_cost, entry_count, new_node))
                            entry_count += 1
                        elif board_dict.get(target_coord).color == PlayerColor.BLUE and cell_state.height > board_dict[target_coord].height:
                            eat_action = EatAction(coord, direction)
                            new_board = eat(board_dict, eat_action)
                            new_path = next_node.path + [eat_action]
                            new_node = Node(board=tuple(sorted(new_board.items())), path=new_path, g_cost=next_node.g_cost + 1, h_cost=0)
                            heapq.heappush(priority_queue, (new_node.f_cost, entry_count, new_node))
                            entry_count += 1
                        elif cell_state.height >= 2:
                            cascade_action = CascadeAction(coord, direction)
                            new_board = cascade(board_dict, cascade_action)
                            new_path = next_node.path + [cascade_action]
                            new_node = Node(board=tuple(sorted(new_board.items())), path=new_path, g_cost=next_node.g_cost + 1, h_cost=0)
                            heapq.heappush(priority_queue, (new_node.f_cost, entry_count, new_node))
                            entry_count += 1
    return actions


def no_blue(board_state: tuple) -> bool:
    """
    Check if there are no blue cells left on the board.
    """
    for coord, cell_state in board_state:
        if cell_state.color == PlayerColor.BLUE:
            return False
    return True

def is_within_bounds(coord: Coord, direction: Direction) -> bool:
    new_r = coord.r + direction.r
    new_c = coord.c + direction.c
    return 0 <= new_r < 8 and 0 <= new_c < 8



