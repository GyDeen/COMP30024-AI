from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from .utils import render_board

def action():
    pass

def move():
    pass


def eat():
    pass

def cascade(board: dict[Coord, CellState], move_action: MoveAction, actions: list[Action]):
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
    # Create a new board state to apply the cascade effects without modifying the original board until all changes are determined
    new_board = board.copy()

    # Check if the move action's coord exists in the board and has height >= 2
    if move_action.coord not in new_board or new_board[move_action.coord].height < 2:
        return new_board

    stack = new_board[action.coord]
    height = stack.height
    color = stack.color
    
    
    for target_coord in get_cascade_path(action.coord, action.direction, height):
        # If the target coordinate is in the board and has a stack, we need to push it before placing the new token
        if target_coord in new_board:
            new_board = push_stack_recursion(new_board, target_coord, action.direction)
            
        new_board[target_coord] = CellState(color, 1)

    return new_board


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
        

def push_stack_recursion(board: dict[Coord, CellState], start_coord: Coord, direction: Direction) -> dict[Coord, CellState]:
    """Recursively pushes stacks in the given direction until an empty cell is found or the edge of the board is reached."""
    # Reached the edge of the board, stop pushing
    if start_coord not in board:
        return board

    # Find an empty cell, stop pushing
    stack = board[start_coord]
    if not stack.is_stack:
        return board

    # Calculate the next coordinate in the cascade direction
    next_coord = start_coord + direction

    # Recursively push the next stack
    board = push_stack_recursion(board, next_coord, direction)

    # Move the current stack to the next coordinate
    board[next_coord] = stack
    del board[start_coord]  # Remove the stack from its current position

    return board
    
    
    
    
    
    
    
    
