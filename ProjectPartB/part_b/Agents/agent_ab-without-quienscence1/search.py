from .board import BoardState
from .action import pack_move, translate_to_Action, execute_move
from .action import PLACE, MOVE, EAT, CASCADE
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

# ==============================================================================
# Tactical closeout / move ordering helpers
# ==============================================================================

def _token_counts(board: BoardState, is_red: bool) -> tuple[int, int]:
    my_tokens = board.red_tokens if is_red else board.blue_tokens
    opp_tokens = board.blue_tokens if is_red else board.red_tokens
    return my_tokens, opp_tokens


def _stack_count(board: BoardState, is_red: bool) -> int:
    n = 0
    for idx in range(64):
        if (is_red and board.is_red(idx)) or ((not is_red) and board.is_blue(idx)):
            n += 1
    return n


def tactical_closeout_move(board: BoardState, is_red: bool, moves: list[int]) -> int | None:
    """
    Before alpha-beta, take obvious finishing moves. This is deliberately small:
    it only overrides search when a move wins immediately, or when we are clearly
    ahead and a cascade/eat causes major material damage without throwing the lead.
    """
    my_before, opp_before = _token_counts(board, is_red)
    if opp_before <= 0:
        return None

    best_move = None
    best_score = 0.0
    opp_stack_n = _stack_count(board, not is_red)
    ahead = my_before > opp_before

    for move in moves:
        action_type = move >> 12
        nb = execute_move(board, move, is_red)
        my_after, opp_after = _token_counts(nb, is_red)

        # Immediate win: always take it.
        if opp_after == 0:
            return move

        opp_lost = opp_before - opp_after
        my_lost = my_before - my_after
        net = opp_lost - my_lost

        score = 0.0
        if action_type == EAT:
            score += opp_lost * 30.0
        elif action_type == CASCADE:
            score += net * 35.0 + opp_lost * 15.0
            # Overstacked pieces should spend height, not sit forever.
            start = (move >> 6) & 0b111111
            if not board.is_empty(start):
                h = board.get_height(start)
                if h > 5:
                    score += (h - 5) * 25.0

        # Only override for non-winning closeout if we stay ahead and enemy is nearly dead.
        if ahead and opp_stack_n <= 2 and net > 0 and my_after >= opp_after:
            score += 80.0

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_score >= 90.0 else None


def order_moves_for_closeout(board: BoardState, is_red: bool, moves: list[int]) -> list[int]:
    """Good ordering lets alpha-beta see kills/cascades earlier and avoid drawish shuffling."""
    my_before, opp_before = _token_counts(board, is_red)

    def key(move: int) -> float:
        action_type = move >> 12
        start = (move >> 6) & 0b111111
        nb = execute_move(board, move, is_red)
        my_after, opp_after = _token_counts(nb, is_red)
        opp_lost = opp_before - opp_after
        my_lost = my_before - my_after
        net = opp_lost - my_lost

        score = 0.0
        if opp_after == 0:
            score += 100000.0
        if action_type == EAT:
            score += 5000.0 + opp_lost * 200.0
        elif action_type == CASCADE:
            score += 1000.0 + net * 250.0 + opp_lost * 100.0
            if not board.is_empty(start):
                h = board.get_height(start)
                if h > 5:
                    score += (h - 5) * 120.0
        elif action_type == MOVE:
            score += 50.0

        # Avoid moves that give away a won material lead.
        if my_before > opp_before and my_after < opp_after:
            score -= 5000.0
        return score

    return sorted(moves, key=key, reverse=True)


def get_best_move(board: BoardState, root_color: int, is_red: bool, time_remaining: float, eval_function, tt: dict) -> Action:
    time_limit_for_turn = min(time_remaining / 40.0, 3.0)
    is_placing = board.turn_count < 8

    if time_remaining < 5.0:
        time_limit_for_turn = 0.2
        
    start_time = time.perf_counter()
    best_move = None

    # First, do not miss obvious kills/closeout cascades. This is what stops
    # many winning positions from becoming draws.
    if not is_placing:
        root_moves = generate_possible_moves(board, is_red=is_red)
        forced = tactical_closeout_move(board, is_red, root_moves)
        if forced is not None:
            return translate_to_Action(forced)
    
    if is_placing:
        # We can maximum have 8 turns in placing phase
        max_depth = 8 - board.turn_count  
    else:
        max_depth = 20
    
    # Iterative Deepening Search
    for depth in range(1, max_depth + 1):
        try:
            move, score = search_all_move(board, depth, float('-inf'), float('inf'), root_color, True, is_red, start_time, time_limit_for_turn, eval_function, tt)

            best_move = move
        except TimeoutError:
            print(f"Depth {depth} search timed out. Returning best move found at depth {depth - 1}.")
            break
    
    # If we couldn't finished searching in even depth of 1, we randomly pick one from a possible move
    if best_move is None:
        fallback_moves = generate_possible_moves(board, is_red=is_red)
        best_move = fallback_moves[0]
        

    return translate_to_Action(best_move)


def search_all_move(board: BoardState, depth: int, alpha: float, beta: float, root_color: int,
                    is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function, tt: dict):
    # No more time to go
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()
    
    
    best_move = None
    best_score = float('-inf')
    moves = generate_possible_moves(board, is_red=is_red)
    
    board_key = (board.get_hashable(), is_max)
    if board_key in tt:
        _, _, _, entry_move = tt[board_key]
        if entry_move is not None and entry_move in moves:
            moves.remove(entry_move)
            moves.insert(0, entry_move)

    for current_move in moves:
        new_board = execute_move(board, current_move, is_red)
        score = alphabeta(new_board, depth - 1, alpha, beta, root_color, not is_max, not is_red, start_time, time_limit, eval_function, tt)

        # Update best move and score if we found a better one
        if score > best_score:
            best_score = score
            best_move = current_move
        
        # Update alpha for pruning
        alpha = max(alpha, best_score)
        if beta <= alpha:
            break
        
    tt[board_key] = (best_score, depth, 0, best_move)
    return best_move, best_score


def alphabeta(board: BoardState, depth: int, alpha: float, beta: float, root_color: int,
              is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function, tt: dict) -> float:
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()
    
    if depth == 0:
        return eval_function(board, root_color, is_red)
    
    original_alpha = alpha
    original_beta = beta
    board_key = (board.get_hashable(), is_max)
    cached_best_move = None

    if board_key in tt:
        entry_score, entry_depth, entry_flag, entry_move = tt[board_key]
        cached_best_move = entry_move
        
        if entry_depth >= depth:
            # It is the same as previous search, we can directly return the score
            if entry_flag == 0: 
                return entry_score
            # The board is a gauranteed lower bound with possible higher value, we can update alpha
            elif entry_flag == 1:
                alpha = max(alpha, entry_score)
            # The board is a gauranteed upper bound with possible lower value, we can update beta
            elif entry_flag == 2: 
                beta = min(beta, entry_score)
            
            if alpha >= beta:
                return entry_score
    
    moves = generate_possible_moves(board, is_red=is_red)
    
    if not moves:
        return eval_function(board, root_color, is_red)
    
    if cached_best_move is not None and cached_best_move in moves:
        moves.remove(cached_best_move)
        moves.insert(0, cached_best_move)
    
    if is_max:
        best_score = float('-inf')
        for move in moves:
            new_board = execute_move(board, move, is_red)
            # Current is MAX level, flip to MIN level for next
            eval = alphabeta(new_board, depth - 1, alpha, beta, root_color, False, not is_red, start_time, time_limit, eval_function, tt)
            if eval > best_score:
                best_score = eval
                best_move = move
                
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
    else:
        best_score = float('inf')
        for move in moves:
            new_board = execute_move(board, move, is_red)
            eval_score = alphabeta(new_board, depth - 1, alpha, beta, root_color, True, not is_red, start_time, time_limit, eval_function, tt)
            
            if eval_score < best_score:
                best_score = eval_score
                best_move = move
                
            beta = min(beta, best_score)
            if beta <= alpha:
                break
    
    # UPPERBOUND (failed high, score is at most this)
    if best_score <= original_alpha:
        flag = 2 
    # LOWERBOUND (failed low, score is at least this)
    elif best_score >= original_beta:
        flag = 1 
    # EXACT (score fell cleanly between alpha and beta)
    else:
        flag = 0 

    # Store in TT 
    tt[board_key] = (best_score, depth, flag, best_move)
    return best_score



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
            
            for _, target in ADJACENT[start].items():
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
                
    # Eats first, then useful cascades, then normal moves. The extra ordering is
    # important for alpha-beta: it sees forced kills earlier and avoids drawish
    # shuffling when ahead.
    return order_moves_for_closeout(board, is_red, eat_moves + cascade_moves + normal_moves)


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