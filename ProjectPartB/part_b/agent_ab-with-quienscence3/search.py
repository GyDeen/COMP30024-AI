from .board import BoardState
from .action import pack_move, unpack_move, translate_to_Action, execute_move
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


CENTER_CELLS = (27, 28, 35, 36)
CORNERS = (0, 7, 56, 63)


def _manhattan(a: int, b: int) -> int:
    ar, ac = divmod(a, 8)
    br, bc = divmod(b, 8)
    return abs(ar - br) + abs(ac - bc)


def _edge_dist(idx: int) -> int:
    r, c = divmod(idx, 8)
    return min(r, 7 - r, c, 7 - c)


def _center_dist(idx: int) -> int:
    return min(_manhattan(idx, c) for c in CENTER_CELLS)


def _corner_dist(idx: int) -> int:
    return min(_manhattan(idx, c) for c in CORNERS)


def _stack_maps(board: BoardState, is_red: bool):
    mine, opp = {}, {}
    for idx in range(64):
        if board.is_empty(idx):
            continue
        h = board.get_height(idx)
        if (is_red and board.is_red(idx)) or ((not is_red) and board.is_blue(idx)):
            mine[idx] = h
        else:
            opp[idx] = h
    return mine, opp


def _is_attacked_by(mine: dict, cell: int, target_h: int) -> bool:
    for nb in ADJACENT[cell].values():
        if nb in mine and mine[nb] >= target_h:
            return True
    return False


def _two_stack_endgame_score(before_mine: dict, before_opp: dict,
                             after_mine: dict, after_opp: dict,
                             action_type: int, damage: int) -> float:
    """
    Closeout stability rule:
    while the opponent is still alive, do not collapse into a single chasing stack.
    Two stacks are much better because one stack can cut off centre while the
    other threatens/eats/cascades. Immediate wins are exempt.
    """
    if not after_opp:
        return 0.0

    before_count = len(before_mine)
    after_count = len(after_mine)
    enemy_count = len(after_opp)
    enemy_tokens = sum(after_opp.values())
    my_tokens = sum(after_mine.values())
    total = my_tokens + enemy_tokens
    advantage = (my_tokens - enemy_tokens) / total if total > 0 else 0.0

    # Only force this hard when we are actually in chase/closeout mode.
    chasing = advantage > 0.05 and (enemy_count <= 3 or enemy_tokens <= 7 or total <= 14)
    if not chasing:
        return 0.0

    score = 0.0

    if after_count <= 1:
        # This is the big one: avoid ending up with a lone R4/R5 chasing forever.
        score -= 900.0
        if before_count >= 2:
            score -= 500.0  # extra penalty for throwing away the second stack
        if action_type == EAT and damage <= 1:
            score -= 250.0  # eating a tiny stack but becoming single-stack is drawish
        if action_type == CASCADE and damage <= 0:
            score -= 300.0
    else:
        # Prefer exactly 2-3 useful stacks in closeout: one chaser + one blocker/cutter.
        score += 180.0
        if after_count == 2:
            score += 120.0
        elif after_count == 3:
            score += 60.0

        # Reward keeping at least one strong chaser and one support stack.
        heights = sorted(after_mine.values(), reverse=True)
        if heights[0] >= 3 and len(heights) >= 2 and heights[1] >= 1:
            score += 100.0
        if heights[0] >= 4 and len(heights) >= 2:
            score += 80.0

    # If we are already down to one stack, encourage a useful cascade split
    # rather than endless one-stack shuffling.
    if before_count <= 1 and after_count >= 2:
        score += 450.0
        if action_type == CASCADE:
            score += 180.0

    return score


def _safe_escape_count(enemy_idx: int, enemy_h: int, mine: dict, opp: dict) -> int:
    current_center = _center_dist(enemy_idx)
    current_edge = _edge_dist(enemy_idx)
    count = 0
    for nb in ADJACENT[enemy_idx].values():
        if nb in mine:
            continue
        if _is_attacked_by(mine, nb, enemy_h):
            continue
        next_center = _center_dist(nb)
        next_edge = _edge_dist(nb)
        if next_center < current_center or next_edge > current_edge:
            count += 2
        else:
            count += 1
    return count


def _is_closeout_mode(board: BoardState, is_red: bool) -> bool:
    mine, opp = _stack_maps(board, is_red)
    my_tokens = sum(mine.values())
    opp_tokens = sum(opp.values())
    total = my_tokens + opp_tokens
    if not mine or not opp or total <= 0:
        return False
    advantage = (my_tokens - opp_tokens) / total
    return advantage > 0.05 and (len(opp) <= 3 or opp_tokens <= 7 or total <= 14)


def _closeout_move_score(board: BoardState, move: int, is_red: bool, forbidden_hashes=None) -> float:
    """Root move ordering/override score: prefer moves that actually finish the game."""
    before_mine, before_opp = _stack_maps(board, is_red)
    before_opp_tokens = sum(before_opp.values())
    action_type, start, target = unpack_move(move)

    try:
        nb = execute_move(board, move, is_red)
    except Exception:
        return -10**9

    # Absolute root anti-repeat: if this move returns to a recent board,
    # make it almost impossible to choose while other legal moves exist.
    # This specifically prevents up/down or left/right chase loops that draw.
    if forbidden_hashes is not None and nb.get_hashable() in forbidden_hashes:
        return -10**7

    after_mine, after_opp = _stack_maps(nb, is_red)
    if not after_opp:
        return 10**8
    if not after_mine:
        return -10**8

    after_opp_tokens = sum(after_opp.values())
    damage = before_opp_tokens - after_opp_tokens
    score = damage * 200.0

    # User-requested closeout rule: keep at least two friendly stacks alive
    # while chasing unless the move immediately eliminates the opponent.
    score += _two_stack_endgame_score(before_mine, before_opp, after_mine, after_opp, action_type, damage)

    if action_type == EAT:
        score += 80.0 + damage * 50.0
    elif action_type == CASCADE:
        # Cascades are good only if they damage or tighten the trap.
        score += 25.0 if damage > 0 else -20.0
        if board.get_height(start) > 5:
            score += 45.0  # use overstacked height instead of sitting forever
    elif action_type == MOVE:
        score += 5.0

    # Reward reducing enemy safe escapes, especially at edge/corner.
    escape_pressure = 0.0
    for e_idx, e_h in after_opp.items():
        safe = _safe_escape_count(e_idx, e_h, after_mine, after_opp)
        escape_pressure += (5 - safe) * e_h * 12.0
        escape_pressure += max(0, 3 - _edge_dist(e_idx)) * e_h * 8.0
        escape_pressure += max(0, 4 - _corner_dist(e_idx)) * e_h * 10.0
        escape_pressure += _center_dist(e_idx) * e_h * 3.0
        if safe <= 1:
            escape_pressure += e_h * 60.0

    score += escape_pressure

    # Chasing progress: when we are the taller side, do not just oscillate.
    # Reward moves that reduce distance from our threatening stack to the
    # most important enemy target. This fixes R4 bouncing up/down instead of
    # walking left to trap B3.
    if before_mine and before_opp and after_mine and after_opp:
        before_best = 99
        after_best = 99
        for e_idx, e_h in before_opp.items():
            for m_idx, m_h in before_mine.items():
                if m_h >= e_h:
                    before_best = min(before_best, _manhattan(m_idx, e_idx))
        for e_idx, e_h in after_opp.items():
            for m_idx, m_h in after_mine.items():
                if m_h >= e_h:
                    after_best = min(after_best, _manhattan(m_idx, e_idx))
        if before_best < 99 and after_best < 99:
            score += (before_best - after_best) * 180.0
            if after_best > before_best:
                score -= 220.0

    # Avoid making silly h6+ stacks unless the move is damaging/finishing.
    for _, h in after_mine.items():
        if h > 5 and damage <= 0:
            score -= 90.0 + (h - 5) * 60.0

    return score


def tactical_closeout_move(board: BoardState, is_red: bool, forbidden_hashes=None):
    """
    Before normal alpha-beta, take an immediate win or a clearly decisive closeout move.
    This prevents strong positions from drifting into draws.
    """
    if board.turn_count < 8:
        return None

    moves = generate_possible_moves(board, is_red)
    if not moves:
        return None

    # Always take immediate elimination.
    best_move = None
    best_score = -10**18
    for move in moves:
        score = _closeout_move_score(board, move, is_red, forbidden_hashes)
        if score > best_score:
            best_score = score
            best_move = move
    if best_score >= 10**7:
        return best_move

    if not _is_closeout_mode(board, is_red):
        return None

    # In closeout mode, only override when the move is clearly forcing/damaging.
    action_type, _, _ = unpack_move(best_move)
    if best_score >= 170.0 and action_type in (EAT, CASCADE, MOVE):
        return best_move
    return None


def order_moves_for_closeout(board: BoardState, moves: list[int], is_red: bool, forbidden_hashes=None) -> list[int]:
    if board.turn_count < 8 or not moves:
        return moves
    return sorted(moves, key=lambda m: _closeout_move_score(board, m, is_red, forbidden_hashes), reverse=True)

def get_best_move(board: BoardState, root_color: int, is_red: bool, time_remaining: float, 
                  place_eval_func, move_eval_func, tt: dict, forbidden_hashes=None) -> Action:
    time_limit_for_turn = min(time_remaining / 40.0, 3.0)
    is_placing = board.turn_count < 8

    if time_remaining < 5.0:
        time_limit_for_turn = 0.2
        
    start_time = time.perf_counter()
    best_move = None

    # Tactical closeout override:
    # If there is a direct win or a clearly forcing kill/trap move, take it
    # before search has a chance to prefer a safe-looking drawing move.
    if not is_placing:
        tactical = tactical_closeout_move(board, is_red, forbidden_hashes)
        if tactical is not None:
            return translate_to_Action(tactical)
    
    max_depth = 20
    
    # Iterative Deepening Search
    for depth in range(1, max_depth + 1):
        try:
            if is_placing:
                eval_function = place_eval_func
            else:
                eval_function = move_eval_func
            move, score = search_all_move(board, depth, float('-inf'), float('inf'), root_color, True, is_red, start_time, time_limit_for_turn, eval_function, tt, forbidden_hashes)

            best_move = move
        except TimeoutError:
            print(f"Depth {depth} search timed out. Returning best move found at depth {depth - 1}.")
            break
    
    # If we couldn't finished searching in even depth of 1, we randomly pick one from a possible move
    if best_move is None:
        fallback_moves = generate_possible_moves(board, is_red=is_red)
        if forbidden_hashes:
            nonrepeat = []
            for m in fallback_moves:
                try:
                    nb = execute_move(board, m, is_red)
                    if nb.get_hashable() not in forbidden_hashes:
                        nonrepeat.append(m)
                except Exception:
                    pass
            if nonrepeat:
                fallback_moves = order_moves_for_closeout(board, nonrepeat, is_red, forbidden_hashes)
        best_move = fallback_moves[0]
        

    # Final root safeguard: if alpha-beta selected a move that repeats a recent
    # board, replace it with the best non-repeating closeout move.
    if forbidden_hashes and best_move is not None:
        try:
            nb = execute_move(board, best_move, is_red)
            if nb.get_hashable() in forbidden_hashes:
                candidates = []
                for m in generate_possible_moves(board, is_red=is_red):
                    try:
                        mb = execute_move(board, m, is_red)
                        if mb.get_hashable() not in forbidden_hashes:
                            candidates.append(m)
                    except Exception:
                        pass
                if candidates:
                    best_move = order_moves_for_closeout(board, candidates, is_red, forbidden_hashes)[0]
        except Exception:
            pass

    return translate_to_Action(best_move)


def search_all_move(board: BoardState, depth: int, alpha: float, beta: float, root_color: int,
                    is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function, tt: dict, forbidden_hashes=None):
    # No more time to go
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()
    
    
    best_move = None
    best_score = float('-inf')
    raw_moves = generate_possible_moves(board, is_red=is_red)
    moves = raw_moves
    if forbidden_hashes and board.turn_count >= 8:
        nonrepeat = []
        for m in raw_moves:
            try:
                nb = execute_move(board, m, is_red)
                if nb.get_hashable() not in forbidden_hashes:
                    nonrepeat.append(m)
            except Exception:
                pass
        if nonrepeat:
            moves = nonrepeat
    moves = order_moves_for_closeout(board, moves, is_red, forbidden_hashes)
    
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
        return quiescence(board, alpha, beta, root_color, is_max, is_red, start_time, time_limit, eval_function)
    
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
    
    moves = order_moves_for_closeout(board, generate_possible_moves(board, is_red=is_red), is_red)
    
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

def quiescence(board: BoardState, alpha: float, beta: float, root_color: int,
              is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function) -> float:
    
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()

    # 1. "Stand Pat" - Evaluate the current board as a baseline
    stand_pat = eval_function(board, root_color, is_red)

    # 2. Stand Pat Pruning: If the baseline is already causing a cutoff, return immediately!
    if is_max:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

    # 3. Only look at noisy moves
    noisy_moves = generate_noisy_moves(board, is_red=is_red)

    # 4. If there are no noisy moves, the board is stable! Return the score.
    if not noisy_moves:
        return stand_pat

    # 5. Otherwise, search the noisy moves to resolve the instability
    if is_max:
        best_score = stand_pat
        for move in noisy_moves:
            new_board = execute_move(board, move, is_red)
            score = quiescence(new_board, alpha, beta, root_color, False, not is_red, start_time, time_limit, eval_function)
            
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = stand_pat
        for move in noisy_moves:
            new_board = execute_move(board, move, is_red)
            score = quiescence(new_board, alpha, beta, root_color, True, not is_red, start_time, time_limit, eval_function)
            
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
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


def generate_noisy_moves(board: BoardState, is_red: bool) -> list[int]:
    """Only generates EAT and CASCADE moves to resolve unstable boards."""
    noisy_moves = []
    
    # During the placing phase, we dont have any noisy move, return directly
    if board.turn_count < 8:
        return noisy_moves 

    for start in range(64):
        if (is_red and board.is_red(start)) or (not is_red and board.is_blue(start)):
            h = board.get_height(start)
            
            for _, target in ADJACENT[start].items():
                target_empty = board.is_empty(target)
                target_friendly = (is_red and board.is_red(target)) or (not is_red and board.is_blue(target))
                
                # NOISY MOVE 1: EAT
                if not target_empty and not target_friendly and h >= board.get_height(target):
                    noisy_moves.append(pack_move(EAT, start, target))

                # NOISY MOVE 2: CASCADE
                if h >= 2:
                    noisy_moves.append(pack_move(CASCADE, start, target))
                    
    return noisy_moves