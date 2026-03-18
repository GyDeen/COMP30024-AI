# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor, BOARD_N
from .utils import render_board
from .action import cascade, eat, move, UP, DOWN, LEFT, RIGHT, EAT, MOVE, CASCADE
import heapq
from .heuristics import heuristic, heuristic_bit
OFFSETS = [UP, DOWN, LEFT, RIGHT]

# Final direction mappping for final output
DIR_MAP = {UP: Direction.Up, DOWN: Direction.Down, LEFT: Direction.Left, RIGHT: Direction.Right}

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
    start_h = heuristic_bit(start_key)
    pq = [(start_h, 0, start_key, ())]
    best_g = {start_key: 0}

    while pq:
        f, g, curr_key, path = heapq.heappop(pq)
        next_key = curr_key

        # Skip if we have already seen this board with an equal or better cost
        if g > best_g.get(next_key, float('inf')):
            continue

        if no_blue_key(next_key):
            return reconstruct_path(path)
        
        for i, val in enumerate(next_key):
            # Only red stacks can act
            if val <= 0: continue
            
            r, c = divmod(i, BOARD_N)
            for d_off in OFFSETS:
                # Boundary check
                if not is_valid_boundary(r, c, d_off): continue
                
                target_idx = i + d_off
                target_val = next_key[target_idx]

                # EAT: target is BLUE and attacker can eat entire stack
                if target_val < 0 and val >= abs(target_val):
                    new_state = eat(next_key, i, d_off)
                    push_state(new_state, g + 1, path + ((EAT, i, d_off),), pq, best_g)
                
                # MOVE: target is EMPTY or RED (merging)
                elif target_val >= 0:
                    new_state = move(curr_key, i, d_off)
                    push_state(new_state, g + 1, path + ((MOVE, i, d_off),), pq, best_g)
                
                # CASCADE: Requires height >= 2 [cite: 510]
                if val >= 2:
                    new_state = cascade(curr_key, i, d_off)
                    push_state(new_state, g + 1, path + ((CASCADE, i, d_off),), pq, best_g)

    return None


def no_blue_key(board_key: tuple[int, ...]) -> bool:
    """Fast goal check using integer board encoding (no negative = no blue)."""
    return all(v >= 0 for v in board_key)


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

def is_valid_boundary(r: int, c: int, offset: int) -> bool:
    """Checks if the move stays within the 8x8 grid."""
    if offset == UP:    return r > 0
    if offset == DOWN:  return r < 7
    if offset == LEFT:  return c > 0
    if offset == RIGHT: return c < 7
    return False

def reconstruct_path(int_path: tuple[tuple[int, int, int], ...]) -> list[Action]:
    """Converts internal (type, idx, off) tuples to required Action objects."""
    actions = []
    for act_type, idx, off in int_path:
        r, c = divmod(idx, BOARD_N)
        coord = Coord(r, c)
        direction = DIR_MAP[off]
        if act_type == MOVE:    actions.append(MoveAction(coord, direction))
        elif act_type == EAT:   actions.append(EatAction(coord, direction))
        else:                   actions.append(CascadeAction(coord, direction))
    return actions

def push_state(state: tuple[int, ...], g: int, path: tuple[tuple[int, int, int], ...], pq: list[tuple[int, int, tuple[int, ...], tuple[tuple[int, int, int], ...]]], best_g: dict[tuple[int, ...], int]):
    """Prunes and pushes new states into the Priority Queue."""
    if g < best_g.get(state, float('inf')):
        best_g[state] = g
        h = heuristic_bit(state)
        heapq.heappush(pq, (g + h, g, state, path))
