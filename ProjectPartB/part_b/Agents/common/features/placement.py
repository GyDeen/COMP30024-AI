"""Unweighted placement features shared by the baseline and related agents."""

from Agents.common.board import BoardState
from .geometry import ADJACENT, CENTER_4, CENTER_WEIGHTS, EDGE_DIST, MANHATTAN


def get_place_maps(board: BoardState, root_color: int) -> tuple[list, list]:
    """Extracts positions of friendly and enemy tokens."""
    friendly = []
    enemies = []
    for idx in range(64):
        if board.is_empty(idx): continue
        if (root_color == 1 and board.is_red(idx)) or (root_color == 0 and board.is_blue(idx)):
            friendly.append(idx)
        else:
            enemies.append(idx)
    return friendly, enemies


def calc_place_centrality(friendly: list) -> float:
    """Rewards placing tokens toward the absolute center-4 squares."""
    score = 0.0
    for idx in friendly:
        score += CENTER_WEIGHTS[idx] * 6.0
    return score


def get_compact_l_score(stacks: list) -> float:
    """Helper: Detects and scores the best Compact L-shape in a list of stacks."""
    if len(stacks) < 3:
        return 0.0

    best_score = 0.0
    stack_set = set(stacks)

    for elbow in stacks:
        neighbors = [n for n in ADJACENT[elbow].values() if n in stack_set]
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                d1, d2 = n1 - elbow, n2 - elbow

                # Perpendicular check (one horizontal, one vertical offset)
                is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or (d1 in (-1, 1) and d2 in (-8, 8))
                if not is_perp:
                    continue

                score = 20.0
                center_score = sum(sc for sc in (CENTER_WEIGHTS[elbow] * 6.0, CENTER_WEIGHTS[n1] * 6.0, CENTER_WEIGHTS[n2] * 6.0))

                score += center_score

                if EDGE_DIST[n1] == 0: score -= CENTER_WEIGHTS[n1] * 6.0
                if EDGE_DIST[n2] == 0: score -= CENTER_WEIGHTS[n2] * 6.0

                if score > best_score:
                    best_score = score

    return best_score


def calc_place_synergy(friendly: list) -> float:
    """Phase-aware synergy. Dictates the structure based on current turn count."""
    f_count = len(friendly)
    if f_count < 2:
        return 0.0

    # Phase 2: Force the second piece to be a separated blocker
    if f_count == 2:
        dist = MANHATTAN[friendly[0]][friendly[1]]
        if dist == 1: return -20.0  # Heavily penalize early adjacency
        if dist == 2: return 8.0    # Perfect spacing
        return 4.0

    # Phase 3: Build the L-start (one pair adjacent, one piece separated)
    if f_count == 3:
        best = -999.0
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = friendly[i], friendly[j]
                if MANHATTAN[a][b] != 1: continue
                blocker = friendly[3 - i - j]

                score = 0.0
                if a in CENTER_4 or b in CENTER_4: score += CENTER_WEIGHTS[27] * 6.0
                score += sum(CENTER_WEIGHTS[x] for x in (a, b))

                # Ensure blocker is separated from the pair
                min_d = min(MANHATTAN[blocker][a], MANHATTAN[blocker][b])
                if min_d >= 2: score += 12.0
                elif min_d == 1: score -= 5.0

                if score > best: best = score
        return best if best > -999.0 else -20.0

    # Phase 4: Full Compact L established
    return get_compact_l_score(friendly)


def calc_place_denial(board: BoardState, friendly: list, enemies: list) -> float:
    """Evaluates how effectively the 'Blocker' piece disrupts the enemy."""
    score = 0.0
    if not friendly or not enemies:
        return score

    # Blocking Enemy Expansion Zones
    f_adj = set()
    for f in friendly:
        f_adj.update(ADJACENT[f].values())

    e_adjacent = set()
    for e in enemies:
        for nb in ADJACENT[e].values():
            if board.is_empty(nb):
                e_adjacent.add(nb)

    blocked = len(e_adjacent & f_adj)
    score += blocked * 6.0

    # Pushing the Blocker Toward Enemy Centroid
    cr, cc = 3.5, 3.5
    e_r = sum(e >> 3 for e in enemies) / len(enemies)
    e_c = sum(e & 7 for e in enemies) / len(enemies)
    toward_r, toward_c = e_r - cr, e_c - cc

    for f in friendly:
        fr, fc = f >> 3, f & 7
        dot = (fr - cr) * toward_r + (fc - cc) * toward_c
        score += dot * 4.0

    # Anti-Enemy L-Shape
    if len(enemies) >= 3:
        score -= get_compact_l_score(enemies) * 0.8

    return score


def extract_placement_features(board: BoardState, root_color: int) -> tuple[float, float, float, float]:
    """Returns (Material, Centrality, Support, Aggression) all scaled from 0 to 100."""
    friendly = []
    enemies = []
    center_raw = 0.0

    for idx in range(64):
        if board.is_empty(idx): continue
        if (root_color and board.is_red(idx)) or (not root_color and board.is_blue(idx)):
            friendly.append(idx)
            center_raw += CENTER_WEIGHTS[idx]
        else:
            enemies.append(idx)

    f_len = len(friendly)
    e_len = len(enemies)

    # Max difference in placement is 4. Scale from -4 to +4 into 0 to 100.
    material = ((f_len - e_len + 4.0) / 8.0) * 100.0

    # Average weight is between 1 (edges) and 4 (center).
    if f_len > 0:
        avg_center = center_raw / f_len
        centrality = ((avg_center - 1.0) / 3.0) * 100.0
    else:
        centrality = 0.0

    # Max theoretical pairs for 4 tokens is 6.
    support_pairs = 0
    for i in range(f_len):
        for j in range(i + 1, f_len):
            r1, c1 = divmod(friendly[i], 8)
            r2, c2 = divmod(friendly[j], 8)
            dist = abs(r1 - r2) + abs(c1 - c2)
            if 1 <= dist <= 3:
                support_pairs += 1
    support = (min(support_pairs, 6) / 6.0) * 100.0

    # How many of our tokens are exactly 2 steps from an enemy?
    threatening = 0
    for f_idx in friendly:
        fr, fc = divmod(f_idx, 8)
        for e_idx in enemies:
            er, ec = divmod(e_idx, 8)
            dist = abs(fr - er) + abs(fc - ec)
            if dist == 2:
                threatening += 1
                break # Count each friendly token max once
    aggression = (min(threatening, 4) / 4.0) * 100.0

    return material, centrality, support, aggression
