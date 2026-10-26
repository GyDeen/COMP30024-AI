"""Placement formulas used by the control-zone and compact-placement variants."""

from Agents.common.board import BoardState
from .geometry import (
    ADJACENT, CENTER_4, CENTER_WEIGHTS, EDGE_DIST, INNER_SQUARE, MANHATTAN,
    _ADJ_SET, _CENTER_4, _DIST_TO_CENTER,
)
from .placement import get_compact_l_score


def get_place_maps_with_control(board: BoardState, root_color: int) -> tuple[list, list, set, set]:
    """Scans the board once for placing phase features."""
    friendly = []
    enemies = []
    f_controlled = set()
    e_controlled = set()

    for idx in range(64):
        if board.is_empty(idx): continue

        is_friendly = (root_color == 1 and board.is_red(idx)) or \
                      (root_color == 0 and board.is_blue(idx))

        if is_friendly:
            friendly.append(idx)
            for adj in ADJACENT[idx].values():
                if board.is_empty(adj): f_controlled.add(adj)
        else:
            enemies.append(idx)
            for adj in ADJACENT[idx].values():
                if board.is_empty(adj): e_controlled.add(adj)

    return friendly, enemies, f_controlled, e_controlled


def calc_place_centrality_average(friendly: list) -> float:
    """A simple average of the center weights of our friendly stacks, rewarding central control."""
    if not friendly:
        return 0.0
    center_raw = sum(CENTER_WEIGHTS[idx] for idx in friendly)
    return center_raw / len(friendly) # likely to between 3.3 - 4.0


def calc_place_zone_control(f_controlled: set, e_controlled: set) -> float:
    # Every empty square adjacent to our stack is a square the enemy CANNOT place on.
    return float(len(f_controlled) - len(e_controlled)) # -4 to +4


def calc_place_synergy_proximity(friendly: list) -> float:
    """
    Rewards L-shapes (diagonal distance 2) which allow safe cascade-to-merge setups.
    """
    score = 0.0
    f_len = len(friendly)
    for i in range(f_len):
        for j in range(i + 1, f_len):
            r1, c1 = divmod(friendly[i], 8)
            r2, c2 = divmod(friendly[j], 8)

            # The True L-Shape Check:
            # Must be exactly 1 row apart AND 1 column apart (a perfect diagonal)
            if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                score += 15.0

    return score


def calc_place_centrality_inner_square(friendly: list) -> float:
    """
    Rewards placing tokens toward the absolute center-4 squares.

    Hard penalty if any token is outside the user-required 4x4 inner box
    (rows 2-5, cols 2-5). The penalty is large enough to overwhelm any
    denial dot-product reward, so corners and edges are effectively
    forbidden unless no inner-square placement is legal.
    """
    score = 0.0
    for idx in friendly:
        score += CENTER_WEIGHTS[idx] * 6.0
        if idx not in INNER_SQUARE:
            score -= 80.0  # stay-in-the-box penalty
    return score


def calc_place_synergy_compact_four(friendly: list) -> float:
    """
    Phase-aware synergy. Dictates the structure based on current turn count.

    Key fix vs. the previous version:
      * f_count == 4 used to return get_compact_l_score(friendly), which
        only scores the best L among 3 of the 4 pieces. The 4th piece
        contributed 0, so an isolated corner placement wasn't penalized
        and the denial dot-product reward could push it into a corner.
        Now the 4th piece is explicitly scored on how close it sits to
        the L formation. Adjacent (forms a 2x2 / extended L) is best,
        isolated is heavily penalized.
      * f_count == 2 and 3 also reject pieces that are too far from
        the existing tokens, so the whole formation stays compact.
    """
    f_count = len(friendly)
    if f_count < 2:
        return 0.0

    # Phase 2: Force the second piece to be a separated blocker — but
    # not too far away. The pair must stay within working range so the
    # eventual L can form.
    if f_count == 2:
        dist = MANHATTAN[friendly[0]][friendly[1]]
        if dist == 1: return -20.0  # too close, would adjacency-merge prematurely
        if dist == 2: return 8.0    # ideal spacing
        if dist == 3: return 4.0
        if dist == 4: return -2.0   # getting too far
        return -15.0                 # >=5: pieces drifting apart

    # Phase 3: Build the L-start (one pair adjacent, one piece nearby).
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

                # Blocker must be within reach of the pair (for L formation).
                min_d = min(MANHATTAN[blocker][a], MANHATTAN[blocker][b])
                if   min_d == 1: score -= 5.0   # would prematurely merge
                elif min_d == 2: score += 12.0  # perfect blocker spacing
                elif min_d == 3: score += 6.0
                elif min_d == 4: score -= 5.0
                else:            score -= 20.0  # blocker drifted too far

                if score > best: best = score
        return best if best > -999.0 else -20.0

    # Phase 4: Full Compact L + 4th piece must integrate.
    # Score = best 3-piece L score + integration bonus for the 4th piece.
    # Adjacent to the L (forms a 2x2 block or extends the L) is ideal;
    # isolated 4th piece is heavily penalized so it can't drift to a
    # corner just because the denial dot-product points there.
    if f_count == 4:
        best_total = -999.0
        for skip in range(4):
            sub = [friendly[i] for i in range(4) if i != skip]
            l_score = get_compact_l_score(sub)
            if l_score <= 0:
                continue

            fourth = friendly[skip]
            min_d = min(MANHATTAN[fourth][s] for s in sub)

            if   min_d == 1: integration = 18.0  # 2x2 block / extended L — ideal
            elif min_d == 2: integration = 10.0
            elif min_d == 3: integration = 0.0
            elif min_d == 4: integration = -15.0
            else:            integration = -30.0  # isolated 4th piece — reject

            total = l_score + integration
            if total > best_total:
                best_total = total

        if best_total <= -999.0:
            return -30.0  # no L formed at all
        return best_total

    return 0.0


def calc_place_denial_inner_square(board: BoardState, friendly: list, enemies: list) -> float:
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

    # Pushing the Blocker Toward Enemy Centroid.
    # Only reward this for tokens inside the inner square. Without this
    # restriction, a corner placement would get a huge dot-product reward
    # (cells far from board centre have large vector magnitudes), which
    # is exactly what was pulling the 4th token out of the box.
    cr, cc = 3.5, 3.5
    e_r = sum(e >> 3 for e in enemies) / len(enemies)
    e_c = sum(e & 7 for e in enemies) / len(enemies)
    toward_r, toward_c = e_r - cr, e_c - cc

    for f in friendly:
        if f not in INNER_SQUARE:
            continue  # outside-box pieces don't earn denial reward
        fr, fc = f >> 3, f & 7
        dot = (fr - cr) * toward_r + (fc - cc) * toward_c
        score += dot * 4.0

    # Anti-Enemy L-Shape
    if len(enemies) >= 3:
        score -= get_compact_l_score(enemies) * 0.8

    return score


def detect_compact_l(stacks):
    """
    Find the best COMPACT L among the given stack positions.

    A compact L is exactly: elbow + 1 adjacent stack in one cardinal
    direction + 1 adjacent stack in a perpendicular direction.
    That's 3 stacks forming a right-angle, all mutually within distance 2.

    Returns (score, elbow_idx, [elbow, arm1, arm2]).
    """
    if len(stacks) < 3:
        return 0.0, -1, []

    stack_set = set(stacks)
    best_score = 0.0
    best_elbow = -1
    best_L = []

    for elbow in stacks:
        neighbors = _ADJ_SET[elbow] & stack_set
        if len(neighbors) < 2:
            continue

        nlist = list(neighbors)
        for i in range(len(nlist)):
            for j in range(i + 1, len(nlist)):
                n1, n2 = nlist[i], nlist[j]
                d1 = n1 - elbow   # direction offset to n1
                d2 = n2 - elbow   # direction offset to n2

                # Must be perpendicular: one horizontal, one vertical
                is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or \
                          (d1 in (-1, 1) and d2 in (-8, 8))
                if not is_perp:
                    continue

                score = 40.0    # base compact-L bonus

                # ── Reward ANY of the 3 stacks being on center-4 ──
                # The key insight: whether the elbow or an arm tip sits
                # on center-4, the L is strategically valuable because
                # we can merge onto that center cell.
                center_count = sum(1 for s in (elbow, n1, n2)
                                   if s in _CENTER_4)
                if center_count >= 2:
                    score += 40.0   # two stacks on center-4 = amazing
                elif center_count == 1:
                    score += 28.0   # one stack on center-4 = great
                else:
                    # No center-4, score by average proximity
                    score += max(0, 15.0 - _DIST_TO_CENTER[elbow] * 5.0)

                # ── All 3 stacks near center (average proximity) ──
                avg_cd = (_DIST_TO_CENTER[elbow] +
                          _DIST_TO_CENTER[n1] +
                          _DIST_TO_CENTER[n2]) / 3.0
                score += max(0, 12.0 - avg_cd * 3.0)

                # ── Penalize if any arm tip is on the edge ──
                if EDGE_DIST[n1] == 0:
                    score -= 12.0
                if EDGE_DIST[n2] == 0:
                    score -= 12.0

                if score > best_score:
                    best_score = score
                    best_elbow = elbow
                    best_L = [elbow, n1, n2]

    return best_score, best_elbow, best_L
