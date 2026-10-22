from .board import BoardState
from .action import pack_move, translate_to_Action, execute_move, unpack_move
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
 
 

def _rule_based_placement(board: BoardState, is_red: bool) -> int:
    """
    Rule-based placement strategy.
 
    Phase 1 (0 own pieces): place at the best available CENTER_4 cell.
    Phase 2 (1 own piece):  place adjacent to it, starting an L arm.
    Phase 3 (2 own pieces): complete the L (perpendicular arm from elbow).
    Phase 4 (3 own pieces): nearby in-box blocker.
 
    Returns a packed move integer.
    """
    INNER = frozenset(r * 8 + c for r in range(2, 6) for c in range(2, 6))
 
    # Collect our pieces and enemy pieces
    my_pieces = []
    enemy_pieces = []
    for idx in range(64):
        if board.is_empty(idx):
            continue
        if (is_red and board.is_red(idx)) or (not is_red and board.is_blue(idx)):
            my_pieces.append(idx)
        else:
            enemy_pieces.append(idx)
 
    # Build legal-cell set from generator
    legal_moves = generate_possible_place(board, is_red)
    legal_cells = set()
    cell_to_move = {}
    for mv in legal_moves:
        _, start, _ = unpack_move(mv)
        legal_cells.add(start)
        cell_to_move[start] = mv
 
    # Helper: pick best cell from a candidate list by a scoring function.
    def _pick(candidates, score_fn):
        best_cell = None
        best_sc = float('-inf')
        for cell in candidates:
            if cell not in legal_cells:
                continue
            sc = score_fn(cell)
            if sc > best_sc:
                best_sc = sc
                best_cell = cell
        return best_cell
 
    # ── Phase 1: first piece → center-4 ──────────────────────
    if len(my_pieces) == 0:
        def score_center(cell):
            sc = 0.0
            # Distance from enemies: farther = safer in early game
            for e in enemy_pieces:
                sc += _manhattan(cell, e) * 2.0
            # Prefer the absolute center weight
            sc += _center_dist(cell) * (-5.0)  # lower dist to center = better
            return sc
 
        pick = _pick(CENTER_CELLS, score_center)
        if pick is not None:
            return cell_to_move[pick]
 
        # Fallback: best inner-square cell closest to any center-4
        inner_legal = [c for c in legal_cells if c in INNER]
        if inner_legal:
            pick = _pick(inner_legal, lambda c: -_center_dist(c) * 10.0)
            if pick is not None:
                return cell_to_move[pick]
 
    # ── Phase 2: second piece → adjacent to first (L arm start) ──
    if len(my_pieces) == 1:
        first = my_pieces[0]
        adj_candidates = [nb for nb in ADJACENT[first].values()
                          if nb in legal_cells and nb in INNER]
 
        def score_arm(cell):
            sc = 0.0
            r, c = cell >> 3, cell & 7
            # Prefer cells toward the center of the board
            sc += (4.0 - abs(r - 3.5) - abs(c - 3.5)) * 5.0
            # Slight bonus for blocking: distance 2 from enemy is diagonal
            for e in enemy_pieces:
                d = _manhattan(cell, e)
                if d == 2:
                    sc += 3.0
                elif d == 3:
                    sc += 1.0
            return sc
 
        pick = _pick(adj_candidates, score_arm)
        if pick is not None:
            return cell_to_move[pick]
 
        # Fallback: nearest in-box cell to first piece
        inner_legal = [c for c in legal_cells if c in INNER]
        if inner_legal:
            pick = _pick(inner_legal, lambda c: -_manhattan(c, first) * 10.0)
            if pick is not None:
                return cell_to_move[pick]
 
    # ── Phase 3: third piece → complete the L (perpendicular arm) ──
    if len(my_pieces) == 2:
        p0, p1 = my_pieces[0], my_pieces[1]
 
        # Determine which is the elbow. Prefer a center-4 piece as elbow.
        if _manhattan(p0, p1) == 1:
            # They are adjacent — one is the elbow, other is arm tip.
            if p0 in CENTER_CELLS:
                elbow, arm = p0, p1
            elif p1 in CENTER_CELLS:
                elbow, arm = p1, p0
            else:
                elbow, arm = p0, p1
 
            arm_offset = arm - elbow
            # Perpendicular directions
            if arm_offset in (-1, 1):   # horizontal arm → want vertical L-arm
                perp = [elbow - 8, elbow + 8]
            elif arm_offset in (-8, 8): # vertical arm → want horizontal L-arm
                perp = [elbow - 1, elbow + 1]
            else:
                perp = []
 
            perp_legal = [c for c in perp
                          if c in legal_cells and c in INNER
                          and 0 <= (c >> 3) < 8 and 0 <= (c & 7) < 8]
 
            def score_l_complete(cell):
                sc = 10.0  # base bonus for completing the L
                r, c = cell >> 3, cell & 7
                sc += (4.0 - abs(r - 3.5) - abs(c - 3.5)) * 3.0
                for e in enemy_pieces:
                    if _manhattan(cell, e) == 2:
                        sc += 4.0
                return sc
 
            pick = _pick(perp_legal, score_l_complete)
            if pick is not None:
                return cell_to_move[pick]
 
            # If perpendicular cells not legal, try any adjacent to elbow
            adj_elbow = [nb for nb in ADJACENT[elbow].values()
                         if nb in legal_cells and nb in INNER and nb != arm]
            pick = _pick(adj_elbow, score_l_complete)
            if pick is not None:
                return cell_to_move[pick]
        else:
            # Not adjacent — pieces were separated. Place adjacent to whichever
            # is more central, to start building the L.
            anchor = p0 if _center_dist(p0) <= _center_dist(p1) else p1
            adj_anchor = [nb for nb in ADJACENT[anchor].values()
                          if nb in legal_cells and nb in INNER]
            pick = _pick(adj_anchor, lambda c: -_center_dist(c) * 5.0)
            if pick is not None:
                return cell_to_move[pick]
 
        # Fallback: nearest in-box cell to existing pieces
        inner_legal = [c for c in legal_cells if c in INNER]
        if inner_legal:
            pick = _pick(inner_legal,
                         lambda c: -min(_manhattan(c, p) for p in my_pieces) * 10.0)
            if pick is not None:
                return cell_to_move[pick]
 
    # ── Phase 4: fourth piece → nearby in-box blocker ─────────
    if len(my_pieces) == 3:
        inner_legal = [c for c in legal_cells if c in INNER]
 
        def score_blocker(cell):
            sc = 0.0
            # Prefer close to the L formation
            min_d = min(_manhattan(cell, p) for p in my_pieces)
            if   min_d == 1: sc += 18.0
            elif min_d == 2: sc += 10.0
            elif min_d == 3: sc += 2.0
            else:            sc -= 20.0
            # Prefer blocking enemy (dist 2 = diagonal block)
            for e in enemy_pieces:
                d = _manhattan(cell, e)
                if   d == 2: sc += 6.0
                elif d == 3: sc += 2.0
            # Centrality tiebreak
            r, c = cell >> 3, cell & 7
            sc += (4.0 - abs(r - 3.5) - abs(c - 3.5)) * 2.0
            return sc
 
        if inner_legal:
            pick = _pick(inner_legal, score_blocker)
            if pick is not None:
                return cell_to_move[pick]
 
    # ── Ultimate fallback ─────────────────────────────────────
    if legal_moves:
        return legal_moves[0]
    raise RuntimeError("No legal placement available")


def get_best_move(board: BoardState, root_color: int, is_red: bool, time_remaining: float, 
                  place_eval_func, move_eval_func, tt: dict) -> Action:
    time_limit_for_turn = min(time_remaining / 40.0, 3.0)
    is_placing = board.turn_count < 8

    if time_remaining < 5.0:
        time_limit_for_turn = 0.2
        
        # ──────────────────────────────────────────────────────────
    # Placement phase: rule-based strategy.
    #
    # The user's requirements are very specific structural constraints:
    #   1. First piece MUST be on a center-4 cell: (3,3),(3,4),(4,3),(4,4)
    #   2. Build a right-angled L-elbow with pieces 2 and 3
    #   3. Place a nearby blocker with piece 4
    #   4. Everything inside the inner 4x4 box (rows 2-5, cols 2-5)
    #
    # The eval-based approach (eval_L_place) was too indirect — tiny
    # scoring differences (e.g., denial dot product) kept pulling the
    # first piece off center-4. A rule-based function is deterministic
    # and satisfies every constraint directly.
    # ──────────────────────────────────────────────────────────
    if is_placing:
        mv = _rule_based_placement(board, is_red)
        return translate_to_Action(mv)
 
    start_time = time.perf_counter()
    best_move = None
 
    max_depth = 20
    
    # Iterative Deepening Search
    for depth in range(1, max_depth + 1):
        try:
            if is_placing:
                eval_function = place_eval_func
            else:
                eval_function = move_eval_func
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

def quiescence(board: BoardState, alpha: float, beta: float, root_color: int,
              is_max: bool, is_red: bool, start_time: float, time_limit: float, eval_function) -> float:
    
    if time.perf_counter() - start_time > time_limit:
        raise TimeoutError()

    base_line = eval_function(board, root_color, is_red)

    # If the baseline is already causing a cutoff, return immediately
    if is_max:
        if base_line >= beta:
            return beta
        alpha = max(alpha, base_line)
    else:
        if base_line <= alpha:
            return alpha
        beta = min(beta, base_line)

    # Look at noisy moves
    noisy_moves = generate_noisy_moves(board, is_red=is_red)

    # If there are no noisy moves, the board is stable
    if not noisy_moves:
        return base_line

    # Search the all the noisy moves, and keep doing quiescence search until we reach a stable board
    if is_max:
        best_score = base_line
        for move in noisy_moves:
            new_board = execute_move(board, move, is_red)
            score = quiescence(new_board, alpha, beta, root_color, False, not is_red, start_time, time_limit, eval_function)
            
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = base_line
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
    A place move is only valid if the cell is empty and all adjacent cells are empty.
 
    HARD CONSTRAINT: prefer placements within the inner 4x4 box (rows 2-5,
    cols 2-5). The user requires all stacks to be inside this box. Only
    fall back to outside-box placements if every in-box cell is illegal
    (e.g., enemy adjacency has blocked them all).
    """
    INNER = frozenset(r * 8 + c for r in range(2, 6) for c in range(2, 6))
    in_box = []
    out_box = []
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
                m = pack_move(PLACE, idx, idx)
                if idx in INNER:
                    in_box.append(m)
                else:
                    out_box.append(m)
    return in_box if in_box else out_box


def generate_noisy_moves(board: BoardState, is_red: bool) -> list[int]:
    """Only generates EAT moves to resolve unstable boards."""
    noisy_moves = []
    
    # During the placing phase, we dont have any noisy move, return directly
    if board.turn_count < 8:
        return noisy_moves 

    for position in range(64):
        if (is_red and board.is_red(position)) or (not is_red and board.is_blue(position)):
            h = board.get_height(position)
            
            for _, target in ADJACENT[position].items():
                target_empty = board.is_empty(target)
                target_friendly = (is_red and board.is_red(target)) or (not is_red and board.is_blue(target))
                
                # Noisy Move directly Eat
                if not target_empty and not target_friendly and h >= board.get_height(target):
                    noisy_moves.append(pack_move(EAT, position, target))

                # # NOISY MOVE 2: CASCADE
                # if h >= 2:
                #     noisy_moves.append(pack_move(CASCADE, start, target))
                    
    return noisy_moves