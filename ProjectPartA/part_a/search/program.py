# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor, BOARD_N
from .utils import render_board
from .action import cascade, eat, move
import heapq
from .state import Node
#added
from .heuristics import heuristic
from math import ceil

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
    
    start_key = encode_board(board)
    start_node = Node(board_key=start_key, board=board, path=[], g_cost=0, h_cost=heuristic(board))

    # Priority queue for A* / UCS ordered by f_cost (g_cost + h_cost)
    priority_queue: list[tuple[int, int, Node]] = []
    entry_count = 0  # tie‑breaker for heap order stability
    best_cost: dict[tuple[int, ...], int] = {}
    
    heapq.heappush(priority_queue, (start_node.f_cost, entry_count, start_node))
    entry_count += 1
    
    while priority_queue:
        f_val, _, next_node = heapq.heappop(priority_queue)

        # Skip if we have already seen this board with an equal or better cost
        if best_cost.get(next_node.board_key, float('inf')) < next_node.g_cost:
            continue
        best_cost[next_node.board_key] = next_node.g_cost

        # Goal test
        if no_blue_key(next_node.board_key):
            return next_node.path

        board_dict = next_node.board

        for coord, cell_state in board_dict.items():
            if cell_state.color != PlayerColor.RED:
                continue

            # CASCADE: always legal when height >= 2 (may push off-board)
            if cell_state.height >= 2:
                for direction in Direction:
                    action = CascadeAction(coord, direction)
                    new_board = cascade(board_dict, action)
                    new_board_key = encode_board(new_board)
                    new_g = next_node.g_cost + 1
                    if best_cost.get(new_board_key, float('inf')) <= new_g:
                        continue
                    best_cost[new_board_key] = new_g
                    new_path = next_node.path + [action]
                    new_node = Node(
                        board_key=new_board_key,
                        board=new_board,
                        path=new_path,
                        g_cost=new_g,
                        h_cost=heuristic(new_board),
                    )
                    heapq.heappush(priority_queue, (new_node.f_cost, entry_count, new_node))
                    entry_count += 1

            # MOVE / EAT (must remain on board)
            for direction in Direction:
                if not is_within_bounds(coord, direction):
                    continue
                target_coord = coord + direction
                target_cell = board_dict.get(target_coord)

                # EAT
                if target_cell is not None and target_cell.color == PlayerColor.BLUE and cell_state.height >= target_cell.height:
                    action = EatAction(coord, direction)
                    new_board = eat(board_dict, action)

                # MOVE
                elif target_cell is None or target_cell.color == PlayerColor.RED:
                    action = MoveAction(coord, direction)
                    new_board = move(board_dict, action)

                else:
                    continue

                new_board_key = encode_board(new_board)
                new_g = next_node.g_cost + 1
                if best_cost.get(new_board_key, float('inf')) <= new_g:
                    continue
                best_cost[new_board_key] = new_g

                new_path = next_node.path + [action]
                new_node = Node(
                    board_key=new_board_key,
                    board=new_board,
                    path=new_path,
                    g_cost=new_g,
                    h_cost=heuristic(new_board),
                )
                heapq.heappush(priority_queue, (new_node.f_cost, entry_count, new_node))
                entry_count += 1

    return None


def no_blue_key(board_key: tuple[int, ...]) -> bool:
    """Fast goal check using integer board encoding (no negative = no blue)."""
    return all(v >= 0 for v in board_key)

def is_within_bounds(coord: Coord, direction: Direction) -> bool:
    new_r = coord.r + direction.r
    new_c = coord.c + direction.c
    return 0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N


def encode_board(board: dict[Coord, CellState]) -> tuple[int, ...]:
    """Encode board into a length-64 tuple of signed ints for fast hashing/comparison.

    Positive values = red stack height, negative values = blue stack height, 0 = empty.
    Index = r * BOARD_N + c.
    """
    arr = [0] * (BOARD_N * BOARD_N)
    for coord, cell in board.items():
        idx = coord.r * BOARD_N + coord.c
        val = cell.height if cell.color == PlayerColor.RED else -cell.height
        arr[idx] = val
    return tuple(arr)
