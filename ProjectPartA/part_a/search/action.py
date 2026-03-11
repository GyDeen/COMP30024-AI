import sys
from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from .utils import render_board


def move(board: dict[Coord, CellState], move_action: MoveAction, actions: list[Action]):
    # Create a new board state to apply the move effect without modifying the original board until all changes are determined
    new_board = board.copy()
    if move_action.coord not in new_board:
        return new_board
    
    curr_stack = new_board[move_action.coord]

    #check if the move is valid
    try:
        target_coord = move_action.coord + move_action.direction
    except ValueError:
        sys.exit("Invalid move: Move goes out of board bounds.")
        
    if target_coord not in new_board:
        new_board[target_coord] = curr_stack
        del new_board[move_action.coord]
    else: 
        #a stack already exists
        target_stack = new_board[target_coord]
        
        #if target stack is friendly
        if target_stack.color == curr_stack.color:
            new_height = target_stack.height + curr_stack.height
            new_board[target_coord] = CellState(curr_stack.color, new_height)
            del new_board[move_action.coord]

    return new_board


def eat(board: dict[Coord, CellState], eat_action: EatAction, actions: list[Action]):
    # Create a new board state to apply the eat effect without modifying the original board until all changes are determined
    new_board = board.copy()

    attacker_coord = eat_action.coord

    #check if attacker position is valid
    if attacker_coord not in new_board:
        return new_board
    
    attacker = new_board[attacker_coord]

    try:
        target_coord = attacker_coord + eat_action.direction
    except ValueError:
        return sys.exit("Invalid eat: Eat action goes out of board bounds.")
    
    if target_coord not in new_board:
        return new_board
    else:
        target = new_board[target_coord]

    #check if target stack is enemy and attacker is taller than target stack
    if target.color == attacker.color or attacker.height < target.height:
        return new_board
    
    del new_board[target_coord]
    new_board[target_coord] = CellState(attacker.color, attacker.height)
    del new_board[attacker_coord]
    return new_board


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

    stack = new_board[move_action.coord]
    height = stack.height
    color = stack.color


    for target_coord in get_cascade_path(move_action.coord, move_action.direction, height):
        # If the target coordinate is in the board and has a stack, we need to push it before placing the new token
        if target_coord in new_board:
            new_board = push_stack_recursion(new_board, target_coord, move_action.direction)
        new_board[target_coord] = CellState(color, 1)

    # Remove the original stack after cascading
    new_board.pop(move_action.coord)  
    actions.append(CascadeAction(move_action.coord, move_action.direction))
    # print(render_board(new_board))
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
    try:
        next_coord = start_coord + direction
    # We reach the bound of the board, all the stacks still need to be pushed will be eliminated
    except ValueError:
        return board

    # Recursively push the next stack
    board = push_stack_recursion(board, next_coord, direction)

    # Move the current stack to the next coordinate
    board[next_coord] = stack
    # Remove the stack from its current position
    del board[start_coord]  
    
    return board
    
    
    
    
    
    
    
    
