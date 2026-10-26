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


def calc_efficiency(friendly_map: dict) -> float:
    score = 0.0
    for h in friendly_map.values():
        if h == 1: score -= 10.0
        elif 2 <= h <= 3: score += 5.0
        elif 4 <= h <= 5: score += 15.0
        elif h == 6: score -= 15.0
        elif h > 6: score -= (30.0 + ((h - 7) * 10.0))
    return score


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


def eval_place_L(board: BoardState, root_color: int, is_red: bool) -> float:
    """
    Phase-aware placement evaluation implementing the teacher's strategy:

      Stack #1: Center-4 grab
      Stack #2: BLOCKER — near center, TOWARD enemy, NOT adjacent to #1.
                Restricts enemy expansion early.
      Stack #3: L-start — adjacent to #1, AWAY from enemy.
      Stack #4: L-complete — adjacent to #3, perpendicular to #1→#3.

    The L is stacks #1, #3, #4.  The blocker is stack #2.
    Both Red and Blue use the same strategy.

    Scoring branches on f_count to guide the correct placement ORDER:
      f_count=1: reward center-4
      f_count=2: reward separated blocker toward enemy, PENALIZE adjacency
      f_count=3: reward adjacent pair (L-start) + separated blocker
      f_count=4: reward completed compact L + good blocker position
    """
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

    f_count = len(friendly)
    e_count = len(enemy)
    f_set = set(friendly)
    e_set = set(enemy)

    score = 0.0

    # ── MATERIAL (universal) ────────────────────────────
    score += (f_count - e_count) * 15.0

    # ── CENTRALITY (universal) ──────────────────────────
    for idx in friendly:
        d = _DIST_TO_CENTER[idx]
        score += (6 - d) * 6.0
        if idx in _CENTER_4:
            score += 18.0
        if EDGE_DIST[idx] == 0:
            score -= 12.0

    # ── Enemy centroid (reused across phases) ───────────
    if enemy:
        e_avg_r = sum(e >> 3 for e in enemy) / e_count
        e_avg_c = sum(e & 7 for e in enemy) / e_count
        # Direction vectors from board center
        cr, cc = 3.5, 3.5
        toward_r = e_avg_r - cr   # toward enemy
        toward_c = e_avg_c - cc
        away_r = -toward_r        # away from enemy
        away_c = -toward_c
    else:
        cr, cc = 3.5, 3.5
        toward_r = toward_c = away_r = away_c = 0.0
        e_avg_r = e_avg_c = 3.5

    # ════════════════════════════════════════════════════
    # PHASE 1: Single stack — just centrality (already scored)
    # ════════════════════════════════════════════════════
    if f_count <= 1:
        pass

    # ════════════════════════════════════════════════════
    # PHASE 2: Center + Blocker
    # The 2nd stack should be the BLOCKER: near center, toward
    # enemy, NOT adjacent to our center stack.
    # PENALIZE adjacency — don't start L yet!
    # ════════════════════════════════════════════════════
    elif f_count == 2:
        a, b = friendly[0], friendly[1]
        dist_ab = _MANHATTAN[a][b]

        # Identify center stack vs blocker
        a_cen = a in _CENTER_4
        b_cen = b in _CENTER_4
        if a_cen and not b_cen:
            center_idx, blocker_idx = a, b
        elif b_cen and not a_cen:
            center_idx, blocker_idx = b, a
        elif a_cen and b_cen:
            # Both center — the one closer to enemy is the blocker
            da = abs((a >> 3) - e_avg_r) + abs((a & 7) - e_avg_c)
            db = abs((b >> 3) - e_avg_r) + abs((b & 7) - e_avg_c)
            if da < db:
                center_idx, blocker_idx = b, a
            else:
                center_idx, blocker_idx = a, b
        else:
            # Neither center — pick the more central as "center"
            if _DIST_TO_CENTER[a] <= _DIST_TO_CENTER[b]:
                center_idx, blocker_idx = a, b
            else:
                center_idx, blocker_idx = b, a

        # ── PENALIZE adjacency (don't start L yet) ──
        if dist_ab == 1:
            score -= 20.0
        elif dist_ab == 2:
            score += 8.0    # ideal blocker distance
        elif dist_ab == 3:
            score += 4.0

        # ── Reward blocker being TOWARD enemy ──
        if enemy:
            br, bc = blocker_idx >> 3, blocker_idx & 7
            dot_toward = (br - cr) * toward_r + (bc - cc) * toward_c
            score += dot_toward * 8.0

        # ── Reward blocker BLOCKING enemy expansion cells ──
        if enemy:
            blocker_adj = _ADJ_SET[blocker_idx]
            enemy_expansion = set()
            for e in enemy:
                for nb in _ADJ_SET[e]:
                    if nb not in e_set and nb not in f_set:
                        enemy_expansion.add(nb)
            blocked_cells = len(enemy_expansion & blocker_adj)
            score += blocked_cells * 10.0

    # ════════════════════════════════════════════════════
    # PHASE 3: L-start + Blocker
    # Two of the three stacks should be adjacent (L-start).
    # The third is the separated blocker.
    # The adjacent pair should be AWAY from enemy.
    # ════════════════════════════════════════════════════
    elif f_count == 3:
        # Try all ways to split 3 stacks into (pair, blocker)
        best_config_score = -9999.0

        for i in range(3):
            for j in range(i + 1, 3):
                a, b = friendly[i], friendly[j]
                if _MANHATTAN[a][b] != 1:
                    continue  # pair must be adjacent
                blocker = friendly[3 - i - j]

                cfg_score = 0.0

                # Pair should include a center-4 cell
                if a in _CENTER_4 or b in _CENTER_4:
                    cfg_score += 25.0

                # Pair near center
                cfg_score += max(0, 8.0 - _DIST_TO_CENTER[a] - _DIST_TO_CENTER[b])

                # Pair AWAY from enemy
                if enemy:
                    pair_r = ((a >> 3) + (b >> 3)) / 2.0
                    pair_c = ((a & 7) + (b & 7)) / 2.0
                    dot_away = (pair_r - cr) * away_r + (pair_c - cc) * away_c
                    cfg_score += dot_away * 6.0

                # Blocker should be SEPARATED from pair
                min_d_to_pair = min(_MANHATTAN[blocker][a], _MANHATTAN[blocker][b])
                if min_d_to_pair >= 2:
                    cfg_score += 12.0
                elif min_d_to_pair == 1:
                    cfg_score -= 5.0  # blocker too close, might form wrong shape

                # Blocker TOWARD enemy
                if enemy:
                    blk_r, blk_c = blocker >> 3, blocker & 7
                    dot_b = (blk_r - cr) * toward_r + (blk_c - cc) * toward_c
                    cfg_score += dot_b * 5.0

                # Blocker near center
                cfg_score += max(0, (4 - _DIST_TO_CENTER[blocker]) * 4.0)

                # Blocker blocks enemy expansion
                if enemy:
                    blk_adj = _ADJ_SET[blocker]
                    e_exp = set()
                    for e in enemy:
                        for nb in _ADJ_SET[e]:
                            if nb not in e_set and nb not in f_set:
                                e_exp.add(nb)
                    cfg_score += len(e_exp & blk_adj) * 6.0

                # L-completion resilience: how many safe cells can
                # complete the L from this pair?
                d_ab = b - a
                completions = set()
                for base in (a, b):
                    for pd in (-8, 8, -1, 1):
                        if pd == d_ab or pd == -d_ab:
                            continue
                        cand = base + pd
                        if cand < 0 or cand >= 64:
                            continue
                        if pd in (-1, 1) and (cand >> 3) != (base >> 3):
                            continue
                        if cand not in f_set and cand not in e_set:
                            completions.add(cand)
                enemy_adj_zone = set()
                for e in enemy:
                    enemy_adj_zone.update(_ADJ_SET[e])
                safe = sum(1 for c in completions if c not in enemy_adj_zone)
                cfg_score += safe * 8.0
                if safe == 0 and completions:
                    cfg_score -= 25.0

                if cfg_score > best_config_score:
                    best_config_score = cfg_score

        if best_config_score > -9999.0:
            score += best_config_score
        else:
            # No adjacent pair found at all — penalize
            score -= 20.0

    # ════════════════════════════════════════════════════
    # PHASE 4: Completed L + Blocker
    # Three stacks form compact L, one is the separated blocker.
    # ════════════════════════════════════════════════════
    elif f_count >= 4:
        l_score, l_elbow, l_stacks = _detect_compact_L(friendly)
        score += l_score

        if l_stacks:
            l_set = set(l_stacks)
            elbow = l_stacks[0]
            arm1, arm2 = l_stacks[1], l_stacks[2]

            # ── L should be AWAY from enemy ──
            if enemy:
                l_avg_r = sum(s >> 3 for s in l_stacks) / 3.0
                l_avg_c = sum(s & 7 for s in l_stacks) / 3.0
                dot_L = (l_avg_r - cr) * away_r + (l_avg_c - cc) * away_c
                score += dot_L * 5.0

            # ── Evaluate the blocker (4th stack) ──
            for idx in friendly:
                if idx in l_set:
                    continue

                # Arm extension detection (BAD)
                is_extending = False
                for arm in (arm1, arm2):
                    d_elbow_to_arm = arm - elbow
                    if _MANHATTAN[idx][arm] == 1:
                        if (idx - arm) == d_elbow_to_arm:
                            is_extending = True
                            break
                    if _MANHATTAN[idx][elbow] == 1:
                        if (idx - elbow) == -d_elbow_to_arm:
                            is_extending = True
                            break

                if is_extending:
                    score -= 35.0
                else:
                    # Blocker near center
                    score += max(0, (4 - _DIST_TO_CENTER[idx]) * 5.0)

                    # Blocker TOWARD enemy (the whole point)
                    if enemy:
                        br, bc = idx >> 3, idx & 7
                        dot_b = (br - cr) * toward_r + (bc - cc) * toward_c
                        score += dot_b * 5.0

                    # Blocker separated from L (dist >= 2)
                    min_d = min(_MANHATTAN[idx][l] for l in l_stacks)
                    if min_d >= 2:
                        score += 10.0

                    # Blocker blocks enemy expansion
                    if enemy:
                        blk_adj = _ADJ_SET[idx]
                        e_exp = set()
                        for e in enemy:
                            for nb in _ADJ_SET[e]:
                                if board.is_empty(nb):
                                    e_exp.add(nb)
                        score += len(e_exp & blk_adj) * 6.0

        # ── BLOCKING (universal) ────────────────────────
        if friendly and enemy:
            our_adj = set()
            for f_idx in friendly:
                our_adj.update(_ADJ_SET[f_idx])
            enemy_wants = set()
            for e_idx in enemy:
                for nb in _ADJ_SET[e_idx]:
                    if board.is_empty(nb):
                        enemy_wants.add(nb)
            score += len(enemy_wants & our_adj) * 5.0

        # ── Anti-enemy-L ────────────────────────────────
        if e_count >= 3:
            enemy_l, _, _ = _detect_compact_L(enemy)
            score -= enemy_l * 0.5

    return score
