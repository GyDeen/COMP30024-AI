"""Agent-specific evaluation profiles and strategy wrappers."""

from Agents.common.board import BoardState
from Agents.common.features.geometry import (
    ADJACENT, CENTER_WEIGHTS, EDGE_DIST, _ADJ_SET, _CENTER_4, _DIST_TO_CENTER,
)
from Agents.common.features.movement import (
    get_stack_maps,
    calc_material,
    calc_alignment,
)
from Agents.common.features.movement_variants import (
    calc_positioning_unbounded_height as calc_positioning,
    calc_efficiency_friendly_only as calc_efficiency,
    calc_safety_friendly_pushes as calc_safety,
    calc_hunting_all_pairs as calc_hunting,
)
from Agents.common.features.placement import (
    extract_placement_features as eval_place0,
)
from Agents.common.features.placement_variants import (
    detect_compact_l as _detect_compact_L,
)


RED = 1
BLUE = 0


PROFILES = {
    # The Phalanx Series (less emphasis on hunting, more on solid board presence)
    'phalanx_balanced':   {'material': 0.20, 'safety': 0.20, 'positioning': 0.20, 'alignment': 0.20, 'efficiency': 0.20, 'hunting': 0.00},
    'phalanx_aggressive': {'material': 0.40, 'safety': 0.25, 'positioning': 0.15, 'alignment': 0.15, 'efficiency': 0.10, 'hunting': 0.00},

    # The Hunt Series
    'hunt_aggressive':    {'material': 0.25, 'safety': 0.15, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15},
    'hunt_balanced':      {'material': 0.30, 'safety': 0.20, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15},
    'hunt_defensive':     {'material': 0.30, 'safety': 0.25, 'positioning': 0.20, 'alignment': 0.15, 'efficiency': 0.10, 'hunting': 0.00}
}


def evaluate_board(board: BoardState, root_color: int, is_current_turn_red: bool, profile_name: str) -> float:
    """The core evaluation runner that dynamically applies the requested profile"""
    weights = PROFILES.get(profile_name, PROFILES['phalanx_balanced'])
    is_my_turn = (root_color == 1 and is_current_turn_red) or (root_color == 0 and not is_current_turn_red)

    f_map, e_map = get_stack_maps(board, root_color)

    # Only calculate the feature if the weight is greater than 0, saving CPU time!
    total_score = 0.0
    if weights['material'] > 0:    total_score += calc_material(f_map, e_map) * weights['material']
    if weights['positioning'] > 0: total_score += calc_positioning(f_map, e_map) * weights['positioning']
    if weights['efficiency'] > 0:  total_score += calc_efficiency(f_map) * weights['efficiency']
    if weights['alignment'] > 0:   total_score += calc_alignment(f_map) * weights['alignment']
    if weights['safety'] > 0:      total_score += calc_safety(f_map, e_map, is_my_turn) * weights['safety']
    if weights['hunting'] > 0:     total_score += calc_hunting(f_map, e_map) * weights['hunting']

    return total_score


def eval0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_aggressive')


def eval1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_balanced')


def eval_place_co0(board: BoardState, root_color: int, is_red: bool) -> float:
    """Defensive evaluation for placing phase, focused on building a strong fortress."""
    mat, cen, sup, agg = eval_place0(board, root_color)

    # Calculate weighted final score
    score = (mat * 0.40) + (cen * 0.40) + (sup * 0.20) + (agg * 0.00)
    return score


def eval_place_co1(board: BoardState, root_color: int, is_red: bool) -> float:
    """Aggressive evaluation for placing phase, focused on threatening the opponent."""
    mat, cen, sup, agg = eval_place0(board, root_color)

    # Calculate weighted final score
    score = (mat * 0.40) + (cen * 0.10) + (sup * 0.00) + (agg * 0.50)
    return score


def eval_place_co2(board: BoardState, root_color: int, is_red: bool) -> float:
    """Balanced evaluation for placing phase, with moderate emphasis on both defense and aggression."""
    mat, cen, sup, agg = eval_place0(board, root_color)

    # Calculate weighted final score
    score = (mat * 0.40) + (cen * 0.25) + (sup * 0.20) + (agg * 0.15)
    return score


_MANHATTAN = [[0]*64 for _ in range(64)]


for _a in range(64):
    for _b in range(64):
        _MANHATTAN[_a][_b] = abs((_a >> 3) - (_b >> 3)) + abs((_a & 7) - (_b & 7))


def _scan_board(board, root_color):
    """Extract friendly/enemy positions and compute directional info."""
    friendly = []
    enemy = []
    for idx in range(64):
        if board.is_empty(idx):
            continue
        if (root_color == 1 and board.is_red(idx)) or \
           (root_color == 0 and board.is_blue(idx)):
            friendly.append(idx)
        else:
            enemy.append(idx)

    ctx = {
        'friendly': friendly,
        'enemy': enemy,
        'f_set': set(friendly),
        'e_set': set(enemy),
        'f_count': len(friendly),
        'e_count': len(enemy),
        'board': board,
    }

    # Direction vectors from board center toward/away from enemy
    cr, cc = 3.5, 3.5
    if enemy:
        e_avg_r = sum(e >> 3 for e in enemy) / ctx['e_count']
        e_avg_c = sum(e & 7 for e in enemy) / ctx['e_count']
        ctx['toward_r'] = e_avg_r - cr
        ctx['toward_c'] = e_avg_c - cc
    else:
        e_avg_r = e_avg_c = 3.5
        ctx['toward_r'] = ctx['toward_c'] = 0.0
    ctx['away_r'] = -ctx['toward_r']
    ctx['away_c'] = -ctx['toward_c']
    ctx['e_avg_r'] = e_avg_r
    ctx['e_avg_c'] = e_avg_c
    ctx['cr'] = cr
    ctx['cc'] = cc

    return ctx


def _score_centrality(friendly):
    """Score all friendly stacks by proximity to center-4 cells."""
    score = 0.0
    for idx in friendly:
        d = _DIST_TO_CENTER[idx]
        score += (6 - d) * 6.0
        if idx in _CENTER_4:
            score += 18.0
        if EDGE_DIST[idx] == 0:
            score -= 12.0
    return score


def _get_enemy_expansion(ctx):
    """Return set of empty cells adjacent to enemy stacks (where enemy wants to expand)."""
    e_exp = set()
    for e in ctx['enemy']:
        for nb in _ADJ_SET[e]:
            if nb not in ctx['e_set'] and nb not in ctx['f_set']:
                e_exp.add(nb)
    return e_exp


def _score_blocker(blocker_idx, ctx, toward_weight=8.0, toward_cap=None, block_weight=10.0):
    """
    Score a blocker stack: toward enemy, near center, blocks enemy expansion.

    Args:
        blocker_idx: cell index of the blocker
        ctx: context dict from _scan_board
        toward_weight: multiplier for toward-enemy dot product
        toward_cap: if set, cap the toward-enemy bonus at this value
        block_weight: reward per blocked enemy expansion cell
    """
    score = 0.0

    # Near center
    score += max(0, (4 - _DIST_TO_CENTER[blocker_idx]) * 4.0)

    # Toward enemy
    if ctx['enemy']:
        br, bc = blocker_idx >> 3, blocker_idx & 7
        dot = (br - ctx['cr']) * ctx['toward_r'] + (bc - ctx['cc']) * ctx['toward_c']
        bonus = dot * toward_weight
        if toward_cap is not None:
            bonus = min(bonus, toward_cap)
        score += bonus

    # Blocks enemy expansion
    if ctx['enemy']:
        blk_adj = _ADJ_SET[blocker_idx]
        e_exp = _get_enemy_expansion(ctx)
        score += len(e_exp & blk_adj) * block_weight

    return score


def _get_L_completions(a, b, f_set, e_set):
    """
    Find all cells that would complete a compact L from adjacent pair (a, b).
    Returns set of valid completion cell indices.
    """
    d_ab = b - a
    completions = set()
    for base in (a, b):
        for pd in (-8, 8, -1, 1):
            if pd == d_ab or pd == -d_ab:
                continue  # skip collinear
            cand = base + pd
            if cand < 0 or cand >= 64:
                continue
            if pd in (-1, 1) and (cand >> 3) != (base >> 3):
                continue  # horizontal wrap check
            if cand not in f_set and cand not in e_set:
                completions.add(cand)
    return completions


def _is_arm_extension(idx, elbow, arm1, arm2):
    """
    Check if placing at idx would just extend an L arm into a useless line.
    Returns True if idx extends past an arm tip OR through the elbow.
    """
    for arm in (arm1, arm2):
        d_elbow_to_arm = arm - elbow
        # Case A: extends past arm tip in same direction
        if _MANHATTAN[idx][arm] == 1 and (idx - arm) == d_elbow_to_arm:
            return True
        # Case B: extends through elbow in opposite direction
        if _MANHATTAN[idx][elbow] == 1 and (idx - elbow) == -d_elbow_to_arm:
            return True
    return False


def _phase2_center_blocker(ctx):
    """
    Phase 2 (f_count=2): Center + Blocker.
    The 2nd stack should be a BLOCKER: near center, toward enemy,
    NOT adjacent to our center stack. Penalize adjacency.
    """
    friendly = ctx['friendly']
    a, b = friendly[0], friendly[1]
    dist_ab = _MANHATTAN[a][b]

    # Identify which is center vs blocker
    a_cen = a in _CENTER_4
    b_cen = b in _CENTER_4
    if a_cen and not b_cen:
        center_idx, blocker_idx = a, b
    elif b_cen and not a_cen:
        center_idx, blocker_idx = b, a
    elif a_cen and b_cen:
        da = abs((a >> 3) - ctx['e_avg_r']) + abs((a & 7) - ctx['e_avg_c'])
        db = abs((b >> 3) - ctx['e_avg_r']) + abs((b & 7) - ctx['e_avg_c'])
        center_idx, blocker_idx = (b, a) if da < db else (a, b)
    else:
        center_idx, blocker_idx = (a, b) if _DIST_TO_CENTER[a] <= _DIST_TO_CENTER[b] else (b, a)

    score = 0.0

    # Penalize adjacency — don't start L yet
    if dist_ab == 1:
        score -= 20.0
    elif dist_ab == 2:
        score += 8.0
    elif dist_ab == 3:
        score += 4.0

    # Blocker quality
    score += _score_blocker(blocker_idx, ctx, toward_weight=8.0, block_weight=10.0)

    return score


def _phase3_adaptive(ctx):
    """
    Phase 3 (f_count=3): Adaptive — L-completion or blocker.

    If we already have an adjacent pair, the 3rd stack should ideally
    COMPLETE the L (perpendicular neighbor = +45 bonus).
    If not, it acts as a blocker (capped toward-enemy reward).
    """
    friendly = ctx['friendly']
    f_set = ctx['f_set']
    e_set = ctx['e_set']

    # Find all adjacent pairs among our 3 stacks
    pairs_found = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = friendly[i], friendly[j]
            if _MANHATTAN[a][b] == 1:
                third = friendly[3 - i - j]
                pairs_found.append((a, b, third))

    if not pairs_found:
        return -20.0  # no adjacent pair — bad placement

    best_cfg = -9999.0

    for a, b, third in pairs_found:
        cfg = 0.0

        # ── Pair quality ──
        if a in _CENTER_4 or b in _CENTER_4:
            cfg += 25.0
        cfg += max(0, 8.0 - _DIST_TO_CENTER[a] - _DIST_TO_CENTER[b])

        # ── Is 3rd stack an L-completion cell? ──
        d_ab = b - a
        is_completion = False
        for base in (a, b):
            for pd in (-8, 8, -1, 1):
                if pd == d_ab or pd == -d_ab:
                    continue
                if base + pd == third:
                    is_completion = True
                    break
            if is_completion:
                break

        if is_completion:
            cfg += _score_L_completion(a, b, third, ctx)
        else:
            cfg += _score_third_as_blocker(a, b, third, ctx)

        if cfg > best_cfg:
            best_cfg = cfg

    return best_cfg


def _score_L_completion(a, b, third, ctx):
    """Score the case where the 3rd stack completes a compact L with pair (a, b)."""
    score = 45.0  # huge L-completion bonus

    # L center-4 presence
    l_all = [a, b, third]
    score += sum(1 for s in l_all if s in _CENTER_4) * 10.0

    # L away from enemy
    if ctx['enemy']:
        l_avg_r = sum(s >> 3 for s in l_all) / 3.0
        l_avg_c = sum(s & 7 for s in l_all) / 3.0
        dot_L = (l_avg_r - ctx['cr']) * ctx['away_r'] + (l_avg_c - ctx['cc']) * ctx['away_c']
        score += dot_L * 5.0

    # Edge penalty
    if EDGE_DIST[third] == 0:
        score -= 10.0

    return score


def _score_third_as_blocker(a, b, third, ctx):
    """Score the case where the 3rd stack acts as a blocker (not L-completion)."""
    f_set = ctx['f_set']
    e_set = ctx['e_set']
    score = 0.0

    # Separation from pair
    min_d = min(_MANHATTAN[third][a], _MANHATTAN[third][b])
    if min_d >= 2:
        score += 8.0

    # Blocker quality (capped toward-enemy)
    score += _score_blocker(third, ctx, toward_weight=3.0, toward_cap=6.0, block_weight=4.0)

    # L-completion resilience: count safe paths for the pair
    completions = _get_L_completions(a, b, f_set, e_set)
    enemy_adj_zone = set()
    for e in ctx['enemy']:
        enemy_adj_zone.update(_ADJ_SET[e])
    safe = sum(1 for c in completions if c not in enemy_adj_zone)
    score += safe * 6.0
    if safe == 0 and completions:
        score -= 20.0

    return score


def _phase4_complete_L(ctx):
    """
    Phase 4 (f_count >= 4): Completed L + Blocker.
    Score the compact L shape and the separated blocker position.
    """
    friendly = ctx['friendly']
    board = ctx['board']
    score = 0.0

    # Detect compact L
    l_score, l_elbow, l_stacks = _detect_compact_L(friendly)
    score += l_score

    if l_stacks:
        l_set = set(l_stacks)
        elbow = l_stacks[0]
        arm1, arm2 = l_stacks[1], l_stacks[2]

        # L away from enemy
        if ctx['enemy']:
            l_avg_r = sum(s >> 3 for s in l_stacks) / 3.0
            l_avg_c = sum(s & 7 for s in l_stacks) / 3.0
            dot_L = (l_avg_r - ctx['cr']) * ctx['away_r'] + (l_avg_c - ctx['cc']) * ctx['away_c']
            score += dot_L * 5.0

        # Evaluate each non-L stack (the blocker)
        for idx in friendly:
            if idx in l_set:
                continue

            if _is_arm_extension(idx, elbow, arm1, arm2):
                score -= 35.0
            else:
                # Blocker quality (capped toward-enemy)
                score += _score_blocker(idx, ctx, toward_weight=4.0, toward_cap=8.0, block_weight=5.0)

                # Separation bonus
                min_d = min(_MANHATTAN[idx][l] for l in l_stacks)
                if min_d >= 2:
                    score += 8.0

    # Universal blocking
    if friendly and ctx['enemy']:
        our_adj = set()
        for f_idx in friendly:
            our_adj.update(_ADJ_SET[f_idx])
        enemy_wants = set()
        for e_idx in ctx['enemy']:
            for nb in _ADJ_SET[e_idx]:
                if board.is_empty(nb):
                    enemy_wants.add(nb)
        score += len(enemy_wants & our_adj) * 5.0

    # Anti-enemy-L
    if ctx['e_count'] >= 3:
        enemy_l, _, _ = _detect_compact_L(ctx['enemy'])
        score -= enemy_l * 0.5

    return score


def eval_place_L(board: BoardState, root_color: int, is_red: bool) -> float:
    """
    Phase-aware placement evaluation implementing the teacher's strategy:

      Stack #1: Center-4 grab
      Stack #2: BLOCKER — near center, TOWARD enemy, NOT adjacent to #1.
      Stack #3: L-start/complete — adjacent to #1, completing the L.
      Stack #4: L-complete or blocker.

    Dispatches to phase-specific functions based on f_count.
    """
    ctx = _scan_board(board, root_color)

    score = 0.0

    # Universal components
    score += (ctx['f_count'] - ctx['e_count']) * 15.0   # material
    score += _score_centrality(ctx['friendly'])          # centrality

    # Phase-specific components
    if ctx['f_count'] <= 1:
        pass  # Phase 1: centrality is enough
    elif ctx['f_count'] == 2:
        score += _phase2_center_blocker(ctx)
    elif ctx['f_count'] == 3:
        score += _phase3_adaptive(ctx)
    elif ctx['f_count'] >= 4:
        score += _phase4_complete_L(ctx)

    return score
