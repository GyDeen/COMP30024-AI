from .board import BoardState
from .action import pack_move, translate_to_Action, execute_move
from .action import UP, DOWN, LEFT, RIGHT, PLACE, MOVE, EAT, CASCADE
from referee.game.actions import Action
import time

RED = 1
BLUE = 0


# Store's all idx's adjacent positions
ADJACENT = []
for i in range(64):
    neighbors = {}
    r, c = divmod(i, 8)
    if r > 0: neighbors[-8] = i - 8
    if r < 7: neighbors[8]  = i + 8  
    if c > 0: neighbors[-1] = i - 1  
    if c < 7: neighbors[1]  = i + 1  
    ADJACENT.append(neighbors)

def get_best_move(board: BoardState, root_color: int, is_red: bool, time_remaining: float, eval_function) -> Action:
    time_limit_for_turn = min(time_remaining / 40.0, 3.0)
    is_placing = board.turn_count < 8

    if time_remaining < 5.0:
        time_limit_for_turn = 0.2
        
    start_time = time.perf_counter()
    best_move = None
    
    if is_placing:
        # We can maximum have 8 turns in placing phase
        max_depth = 8 - board.turn_count  
    else:
        max_depth = 20
    
    # Iterative Deepening Search
    for depth in range(1, max_depth + 1):
        try:
            move, score = search_all_move(board, depth, float('-inf'), float('inf'), root_color, True, is_red, start_time, time_limit_for_turn, eval_function)

            best_move = move
        except TimeoutError:
            break
    
    # If we couldn't finished searching in even depth of 1, we randomly pick one from a possible move
    if best_move is None:
        fallback_moves = generate_possible_moves(board, is_red=is_red)
        best_move = fallback_moves[0]
        

    return translate_to_Action(best_move)


def search_all_move(board: BoardState, depth: int, alpha: float, beta: float, root_color: int,
                    is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function):
    # No more time to go
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()
    
    
    best_move = None
    best_score = float('-inf')
    moves = generate_possible_moves(board, is_red=is_red)

    for current_move in moves:
        new_board = execute_move(board, current_move, is_red)
        score = alphabeta(new_board, depth - 1, alpha, beta, root_color, not is_max, not is_red, start_time, time_limit, eval_function)

        # Update best move and score if we found a better one
        if score > best_score:
            best_score = score
            best_move = current_move
        
        # Update alpha for pruning
        alpha = max(alpha, best_score)
        if beta <= alpha:
            break
    return best_move, best_score


def alphabeta(board: BoardState, depth: int, alpha: float, beta: float, root_color: int,
              is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function) -> float:
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()

    if depth == 0:
        return eval_function(board, root_color)

    moves = generate_possible_moves(board, is_red=is_red)
    if is_max:
        max_eval = float('-inf')
        for move in moves:
            new_board = execute_move(board, move, is_red)
            # Current is MAX level, flip to MIN level for next
            eval = alphabeta(new_board, depth - 1, alpha, beta, root_color, False, not is_red, start_time, time_limit, eval_function)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = execute_move(board, move, is_red)
            # Current is MIN level, flip to MAX level for next
            eval = alphabeta(new_board, depth - 1, alpha, beta, root_color, True, not is_red, start_time, time_limit, eval_function)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def generate_possible_moves(board: BoardState, is_red: bool) -> list[int]:
    """Generates all possible moves for current player. If is_placing is True, only generates place moves.
    Otherwise, generates eat, cascade, and normal moves."""
    if board.turn_count < 8:
        return generate_possible_place(board, is_red)
        
    eat_moves = []
    cascade_moves = []
    normal_moves = []

    for start in range(64):
        # Check if the piece belongs to the current player
        if (is_red and board.is_red(start)) or (not is_red and board.is_blue(start)):
            h = board.get_height(start)
            
            for offset, target in ADJACENT[start].items():
                target_empty = board.is_empty(target)
                target_friendly = (is_red and board.is_red(target)) or (not is_red and board.is_blue(target))
                target_enemy = not target_empty and not target_friendly

                # EAT: Must strictly be an enemy, and we must be tall enough
                if target_enemy and h >= board.get_height(target):
                    eat_moves.append(pack_move(EAT, start, target))

                # CASCADE: Just need height >= 2
                if h >= 2:
                    cascade_moves.append(pack_move(CASCADE, start, target))
                
                # MOVE: Must be empty or friendly
                if target_empty or target_friendly:
                    normal_moves.append(pack_move(MOVE, start, target))
                
    # Eats first, then Cascades, then Normal Moves
    return eat_moves + cascade_moves + normal_moves


def generate_possible_place(board: BoardState, is_red: bool) -> list[int]:
    """Generates all possible place moves for current player. 
    A place move is only valid if the cell is empty and all adjacent cells are empty."""
    place_moves = []
    for idx in range(64):
        # There is an empty cell
        if board.is_empty(idx):
            can_place = True
            # Check whether all adjacent cells are empty, if so, we can place a token here
            for neighbor_idx in ADJACENT[idx].values():
                # One of the neighbour is not friendly, we cannot place a token here
                if not board.is_empty(neighbor_idx):
                    if (is_red and board.is_red(neighbor_idx)) or (not is_red and board.is_blue(neighbor_idx)):
                        continue
                    else:
                        can_place = False
            if can_place:
                place_moves.append(pack_move(PLACE, idx, idx))
    return place_moves