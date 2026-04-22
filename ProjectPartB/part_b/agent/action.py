from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK

BOARD_N = 8
UP, DOWN, LEFT, RIGHT = -8, 8, -1, 1
MOVE = 1
EAT = 2
CASCADE = 3


def place(board_key: BoardState, idx: int, is_red: bool) -> BoardState:
    board = BoardState(board_key)
    board.spawn_stack(idx, is_red)
    return board

def move(board_key: BoardState, start_idx: int, offset: int) -> BoardState:
    board = BoardState(board_key)
    target_idx = start_idx + offset
    
    moving_stack = board.lift_stack(start_idx)

    color = moving_stack & (RED_FLAG | BLUE_FLAG)
    new_height = (moving_stack & HEIGHT_MASK) + board.get_height(target_idx)
    board.place_stack(target_idx, color | new_height)
    
    return board


def eat(board_key: BoardState, start_idx: int, offset: int) -> BoardState:
    board = BoardState(board_key)
    target_idx = start_idx + offset
    
    # Attacker replaces enemy
    board.clear_cell(start_idx + offset)
    board.place_stack(target_idx, board.lift_stack(start_idx))
    
    return board


def cascade(board_key: BoardState, start_idx: int, offset: int) -> BoardState:
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
    board = BoardState(board_key)
    is_red = board.is_red(start_idx)
    h = board.lift_stack(start_idx) & HEIGHT_MASK

    pos = start_idx

    # Spread 1 to h tokens away
    for dist in range(1, h + 1):
        
        # Check if the NEXT step falls off the board
        if not is_valid_coord(pos, offset):
            # Get the token count that remain on the board
            tokens_falling_off = (h + 1) - dist
            if is_red:
                board.red_tokens -= tokens_falling_off
            else:
                board.blue_tokens -= tokens_falling_off
            break # Once we hit the edge, all further tokens fall off instantly
            
        # Step forward
        pos += offset
        
        # Push existing pieces if necessary
        if not board.is_empty(pos):
            push_recursive(board, pos, offset)
            
        # Place the single spreading token
        board.set_cell(pos, is_red, 1)
        
    return board
        

def push_recursive(board: BoardState, pos: int, offset: int):
    """Recursive logic to push stacks off the board or into next cells."""
    if not is_valid_coord(pos, offset):
        # Pushed off the edge
        board.clear_cell(pos)
        return

    target = pos + offset
    if not board.is_empty(target):
        # Pushed stacks can in turn push other stacks
        push_recursive(board, target, offset)
    
    board.place_stack(target, board.lift_stack(pos))


def is_valid_coord(idx, offset):
    """Fast boundary check for an index move."""
    r, c = divmod(idx, BOARD_N)
    if offset == UP:    return r > 0
    if offset == DOWN:  return r < 7
    if offset == LEFT:  return c > 0
    if offset == RIGHT: return c < 7
    return False
    
    
    
    
    
    
    
    
