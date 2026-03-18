import sys
from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from .utils import render_board
from .program import LEFT, RIGHT, UP, DOWN, BOARD_N


def move(board_key: tuple[int, ...], start_idx: int, offset: int) -> tuple[int, ...]:
    board = list(board_key)
    target_idx = start_idx + offset
    val = board[start_idx]
    
    # Remove the stack on the moving cell
    board[start_idx] = 0
    # Add the stack to the target cell
    board[target_idx] += val

    return tuple(board)


def eat(board_key: tuple[int, ...], start_idx: int, offset: int) -> tuple[int, ...]:
    board = list(board_key)
    target_idx = start_idx + offset
    
    # Attacker replaces enemy [cite: 469]
    board[target_idx] = board[start_idx]
    board[start_idx] = 0
    
    return tuple(board)


def cascade(board_key: tuple[int, ...], start_idx: int, offset: int) -> tuple[int, ...]:
    """
    Given a board state and a move action. If the move action's coord do exist in board, and the move action's direction is valid,
    then add the action to the list and perform CASCAD action.
    
    CASCAD action will spread one of their stacks across multiple cells in a cardinal direction. Only stacks with height 2 
    or greater can cascade. 
    
    1. The original stack is removed from its cell.
    2. For each position from 1 to h cells away in the cascade direction, a single token of the cascading
       players colour is placed.
    3. If a cell along the path contains an existing stack (of either colour), that stack is pushed (not
       merged) one cell further in the cascade direction before the new token is placed. Pushed
       stacks can in turn push other stacks ahead of them.
    4. Any stack that gets pushed off the edge of the board is eliminated from the game.
    5. Any token from the cascading stack that would land off the edge of the board is also eliminated.
    """
    board = list(board_key)
    h = board[start_idx]
    board[start_idx] = 0 # Original stack is removed
    
    # Spread 1 to h tokens away
    for dist in range(1, h + 1):
        # Calculate current landing spot
        # We must check bounds for every step of the spread
        pos = start_idx
        in_bounds = True
        for _ in range(dist):
            if is_valid_coord(pos, offset):
                pos += offset
            else:
                in_bounds = False
                break
        
        if not in_bounds:
            # Token lost over the edge
            continue
            
        if board[pos] != 0:
            # Existing stack is pushed one cell further 
            push_recursive(board, pos, offset)
            
        # Place the new single token
        board[pos] = 1 
        
    return tuple(board)


def get_cascade_path(start: Coord, direction: Direction, height: int):
    """Yields coordinates affected by a cascade of a certain height."""
    current = start
    for _ in range(height):
        try:
            current = current + direction
            yield current
        # If we hit the bound of the board, stop yielding further coords
        except ValueError:
            return
        

def push_recursive(board: tuple[int, ...], pos: int, offset: int):
    """Recursive logic to push stacks off the board or into next cells[cite: 514, 515]."""
    if not is_valid_coord(pos, offset):
        # Pushed off the edge
        board[pos] = 0
        return

    target = pos + offset
    if board[target] != 0:
        # Pushed stacks can in turn push other stacks 
        push_recursive(board, target, offset)
    
    board[target] = board[pos]
    board[pos] = 0


def is_valid_coord(idx, offset):
    """Fast boundary check for an index move."""
    r, c = divmod(idx, BOARD_N)
    if offset == UP:    return r > 0
    if offset == DOWN:  return r < 7
    if offset == LEFT:  return c > 0
    if offset == RIGHT: return c < 7
    return False
    
    
    
    
    
    
    
    
