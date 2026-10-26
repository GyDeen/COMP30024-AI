import time

from referee.game.actions import Action

from Agents.common.action import (
    CASCADE,
    EAT,
    MOVE,
    PLACE,
    execute_move,
    pack_move,
    translate_to_Action,
    unpack_move,
)
from Agents.common.board import BoardState

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

# Center-4 cells: (3,3), (3,4), (4,3), (4,4)
CENTER_CELLS = (27, 28, 35, 36)

# Edge distance for each cell
EDGE_DIST = [0] * 64
for _r in range(8):
    for _c in range(8):
        EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)

def _manhattan(a: int, b: int) -> int:
    return abs((a >> 3) - (b >> 3)) + abs((a & 7) - (b & 7))

def _edge_dist(idx: int) -> int:
    r, c = idx >> 3, idx & 7
    return min(r, 7 - r, c, 7 - c)

def _center_dist(idx: int) -> int:
    return min(_manhattan(idx, c) for c in CENTER_CELLS)

def _rule_based_placement(board: BoardState, is_red: bool) -> int:
    """
    Rule-based placement strategy.

    Phase 1 (0 own pieces): place at the best available CENTER_4 cell.
    Phase 2 (1 own piece):  place adjacent to it, starting an L arm.
    Phase 3 (2 own pieces): complete the L (perpendicular arm from elbow).
    Phase 4 (3 own pieces): nearby in-box blocker.

    Fallback: if no inner-square cell is available, pick a non-edge
    non-corner cell close to existing friendly stacks.

    Returns a packed move integer.
    """
    INNER = frozenset(r * 8 + c for r in range(2, 6) for c in range(2, 6))
    # Inner-box edge cells: cells on the perimeter of the 4x4 inner box.
    # An L's elbow goes on one of these so the L's outside faces the board edge.
    INNER_EDGE = frozenset(
        r * 8 + c
        for r in range(2, 6) for c in range(2, 6)
        if r in (2, 5) or c in (2, 5)
    )

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

    # Graceful fallback: non-edge, non-corner, close to existing friendlies.
    def _safe_fallback():
        # Prefer cells with EDGE_DIST >= 1 (not on the board border)
        # and close to existing friendly stacks.
        safe_legal = [c for c in legal_cells if EDGE_DIST[c] >= 1]
        if not safe_legal:
            safe_legal = list(legal_cells)  # truly nothing safe, use anything
        if not safe_legal:
            return legal_moves[0] if legal_moves else None

        def score_safe(cell):
            sc = 0.0
            # Prefer close to own stacks
            if my_pieces:
                sc -= min(_manhattan(cell, p) for p in my_pieces) * 10.0
            # Prefer central
            sc -= _center_dist(cell) * 5.0
            # Prefer not right on the edge
            sc += EDGE_DIST[cell] * 3.0
            return sc

        pick = _pick(safe_legal, score_safe)
        return cell_to_move[pick] if pick is not None else legal_moves[0]

    # ── Phase 1: first piece → center-4 ──────────────────────
    if len(my_pieces) == 0:
        def score_center(cell):
            sc = 0.0
            for e in enemy_pieces:
                sc += _manhattan(cell, e) * 2.0
            sc += _center_dist(cell) * (-5.0)
            return sc

        pick = _pick(CENTER_CELLS, score_center)
        if pick is not None:
            return cell_to_move[pick]

        # Fallback: best inner-square cell closest to center
        inner_legal = [c for c in legal_cells if c in INNER]
        if inner_legal:
            pick = _pick(inner_legal, lambda c: -_center_dist(c) * 10.0)
            if pick is not None:
                return cell_to_move[pick]

        # Graceful fallback: non-edge, non-corner, near center
        return _safe_fallback()

    # ── Phase 2: second piece → elbow on inner-box edge ──────
    # We want a SPREAD-OUT L. The elbow sits on the perimeter of the
    # inner 4x4 box so the L's "outside" faces the board edge.
    # User's preferred order: top → left → bottom → right (matches the
    # routine (3,3)→(2,3)→(5,3)→(2,4) where the elbow goes UP first
    # and the 4th piece extends the secondary arm by one cell).
    if len(my_pieces) == 1:
        first = my_pieces[0]
        adj_edge = [nb for nb in ADJACENT[first].values()
                    if nb in legal_cells and nb in INNER_EDGE]

        def score_elbow(cell):
            sc = 0.0
            # Stay safe: prefer farther from enemies
            if enemy_pieces:
                sc += min(_manhattan(cell, e) for e in enemy_pieces) * 4.0
            # Direction preference (matches user's stated routine)
            r, c = cell >> 3, cell & 7
            if   r == 2: sc += 0.4   # top
            elif c == 2: sc += 0.3   # left
            elif r == 5: sc += 0.2   # bottom
            elif c == 5: sc += 0.1   # right
            return sc

        pick = _pick(adj_edge, score_elbow)
        if pick is not None:
            return cell_to_move[pick]

        # Fallback: any adjacent in-box cell, scored the same way
        adj_inner = [nb for nb in ADJACENT[first].values()
                     if nb in legal_cells and nb in INNER]
        pick = _pick(adj_inner, score_elbow)
        if pick is not None:
            return cell_to_move[pick]

        return _safe_fallback()

    # ── Phase 3: third piece → far end of primary arm ────────
    # Spread-out L: the third piece goes to the opposite edge of the
    # inner 4x4 box from the elbow, along the column (or row) that the
    # elbow's first arm runs.
    #   elbow on top edge (row=2)    → primary arm DOWN  → far end at (5, ec)
    #   elbow on bottom edge (row=5) → primary arm UP    → far end at (2, ec)
    #   elbow on left edge (col=2)   → primary arm RIGHT → far end at (er, 5)
    #   elbow on right edge (col=5)  → primary arm LEFT  → far end at (er, 2)
    if len(my_pieces) == 2:
        p0, p1 = my_pieces[0], my_pieces[1]

        # Identify the elbow: the piece on the inner-box edge.
        # If both are on edge or neither, fall back to the second-placed (p1).
        if p1 in INNER_EDGE and p0 not in INNER_EDGE:
            elbow = p1
        elif p0 in INNER_EDGE and p1 not in INNER_EDGE:
            elbow = p0
        else:
            elbow = p1  # default to second-placed piece

        er, ec = elbow >> 3, elbow & 7

        # Compute far end of primary arm (opposite edge of inner box)
        far_primary = -1
        if   er == 2: far_primary = 5 * 8 + ec  # top → bottom
        elif er == 5: far_primary = 2 * 8 + ec  # bottom → top
        elif ec == 2: far_primary = er * 8 + 5  # left → right
        elif ec == 5: far_primary = er * 8 + 2  # right → left

        if far_primary >= 0 and far_primary in legal_cells:
            return cell_to_move[far_primary]

        # Fallback: any cell along the primary arm direction.
        # User said "fine to not have the extensive arm" if blocked.
        if er == 2 or er == 5:
            arm_cells = [r * 8 + ec for r in range(2, 6) if r != er]
        elif ec == 2 or ec == 5:
            arm_cells = [er * 8 + c for c in range(2, 6) if c != ec]
        else:
            arm_cells = []
        legal_arm = [c for c in arm_cells if c in legal_cells and c not in my_pieces]
        if legal_arm:
            # Prefer farther from elbow (toward where the far end would be)
            pick = max(legal_arm, key=lambda c: _manhattan(c, elbow))
            return cell_to_move[pick]

        # Fallback: best inner-box cell maximizing distance from existing pieces
        inner_legal = [c for c in legal_cells if c in INNER and c not in my_pieces]
        if inner_legal:
            pick = max(inner_legal,
                       key=lambda c: min(_manhattan(c, p) for p in my_pieces))
            return cell_to_move[pick]

        return _safe_fallback()

    # ── Phase 4: fourth piece → extend the secondary arm by one ──
    # Extensive arm: place ADJACENT to the elbow along the elbow's
    # edge, choosing the side with more room inside the inner box
    # (i.e. toward the farther inner-box corner). For elbow at (2,3),
    # the secondary arm runs along row 2; cols 2..5 give 1 cell of
    # room to the left of the elbow and 2 to the right, so the piece
    # is placed at (2,4) — making a contiguous (2,3)–(2,4) arm rather
    # than leaving a gap to (2,5).
    if len(my_pieces) == 3:
        # The elbow is the inner-edge piece adjacent to a NON-edge piece
        # (the original Phase-1 center piece). The far-end piece (placed
        # in Phase 3) is also on inner-edge but is NOT adjacent to the
        # center. Distinguishing them is essential.
        non_edge = [p for p in my_pieces if p not in INNER_EDGE]
        edge_pieces = [p for p in my_pieces if p in INNER_EDGE]

        elbow = None
        if non_edge and edge_pieces:
            for ep in edge_pieces:
                for nep in non_edge:
                    if _manhattan(ep, nep) == 1:
                        elbow = ep
                        break
                if elbow is not None:
                    break
        if elbow is None:
            # Fallback: take the first edge piece, or just any
            if edge_pieces:
                elbow = edge_pieces[0]
            else:
                elbow = my_pieces[1]

        er, ec = elbow >> 3, elbow & 7

        # Secondary arm option: the cell adjacent to the elbow along
        # the elbow's edge, on the side with more room inside the
        # inner 4x4 box (cols/rows 2..5).
        secondary_opts = []
        if er == 2 or er == 5:
            # Horizontal edge → extend along the row
            if (5 - ec) >= (ec - 2):
                target_c = ec + 1   # extend right (more room on the right)
            else:
                target_c = ec - 1   # extend left
            if 2 <= target_c <= 5:
                secondary_opts = [er * 8 + target_c]
        elif ec == 2 or ec == 5:
            # Vertical edge → extend along the column
            if (5 - er) >= (er - 2):
                target_r = er + 1   # extend down (more room below)
            else:
                target_r = er - 1   # extend up
            if 2 <= target_r <= 5:
                secondary_opts = [target_r * 8 + ec]

        secondary_opts = [s for s in secondary_opts
                          if s != elbow and s not in my_pieces]
        legal_opts = [s for s in secondary_opts if s in legal_cells]

        if legal_opts:
            # Pick the legal option farthest from existing pieces (max coverage)
            pick = max(legal_opts,
                       key=lambda c: min(_manhattan(c, p) for p in my_pieces))
            return cell_to_move[pick]

        # Fallback: any inner-box cell maximizing min-distance to existing pieces
        inner_legal = [c for c in legal_cells if c in INNER and c not in my_pieces]
        if inner_legal:
            pick = max(inner_legal,
                       key=lambda c: min(_manhattan(c, p) for p in my_pieces))
            return cell_to_move[pick]

        return _safe_fallback()

    # ── Ultimate fallback ─────────────────────────────────────
    return _safe_fallback()


def get_best_move(board: BoardState, root_color: int, is_red: bool, time_remaining: float, 
                  place_eval_func, move_eval_func, tt: dict) -> Action:
    if not time_remaining:
        time_limit_for_turn = 8
    else:
        time_limit_for_turn = min(time_remaining / 60.0, 1.5)
        is_placing = board.turn_count < 8

    if time_remaining < 5.0:
        time_limit_for_turn = 0.2

    # ── Placement phase: rule-based strategy ──────────────────
    # AB iterative deepening times out during placement (the search
    # crosses turn_count >= 8 at deep plies and explodes into full
    # move/cascade/eat trees). When AB times out, the fallback picks
    # fallback_moves[0] which is the first cell in board-index order
    # — often (0,0), the top-left corner. Rule-based placement is
    # both instant and structurally correct.
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
    cols 2-5). Only fall back to outside-box placements if every in-box
    cell is illegal.
    """
    INNER = frozenset(r * 8 + c for r in range(2, 6) for c in range(2, 6))
    in_box = []
    out_box = []
    for idx in range(64):
        if board.is_empty(idx):
            can_place = True
            for neighbor_idx in ADJACENT[idx].values():
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