"""
Feature extraction for TD-Leaf(λ) evaluation.

The evaluation is: score = dot(weights, features)

This is a LINEAR function of the features, which means:
  - Gradient ∇_w score = features  (trivial to compute)
  - TD-Leaf weight updates are simple: w += α * features * td_error
  - No neural network, no backprop — just a dot product

Each feature is computed from the board state from the perspective
of the player specified by is_red_maximizing.

Feature list (17 features):
  0: material_diff        — my_tokens - opp_tokens
  1: material_ratio       — (my - opp) / (my + opp)
  2: proximity            — height-weighted distance to eatable enemies
  3: my_threatened        — our tokens adjacent to strictly taller enemies
  4: opp_threatened       — their tokens adjacent to our taller stacks
  5: my_debris            — our h=1 stacks near enemies (cascade aftermath)
  6: opp_debris           — their h=1 stacks near our stacks
  7: dominance            — count of our stacks taller than opponent's max
  8: height_advantage     — max(our_tallest - their_tallest, 0)
  9: height_disadvantage  — max(their_tallest - our_tallest, 0)
  10: height_preservation — sum(h²) difference (rewards tall stacks)
  11: edge_trap_bonus     — our cascade push-off opportunities
  12: edge_trap_penalty   — their cascade push-off threats against us
  13: safety_diff         — positional safety (center vs edge)
  14: elim_pressure       — bonus when opponent has few stacks
  15: my_stack_count       — number of our stacks (mobility proxy)
  16: opp_stack_count      — number of their stacks
"""

import numpy as np
from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from .action import is_valid_coord, UP, DOWN, LEFT, RIGHT

OFFSETS = (UP, DOWN, LEFT, RIGHT)
NUM_FEATURES = 17

# Precomputed tables
EDGE_DIST = [0] * 64
for _r in range(8):
    for _c in range(8):
        EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)

FAR_EDGE_DIST = [0] * 256
for _i in range(64):
    _r, _c = _i >> 3, _i & 7
    FAR_EDGE_DIST[_i * 4 + 0] = _r
    FAR_EDGE_DIST[_i * 4 + 1] = 7 - _r
    FAR_EDGE_DIST[_i * 4 + 2] = _c
    FAR_EDGE_DIST[_i * 4 + 3] = 7 - _c

SCAN_OFFSETS = (DOWN, UP, RIGHT, LEFT)

ADJACENTS = [[] for _ in range(64)]
for _i in range(64):
    for _off in OFFSETS:
        if is_valid_coord(_i, _off):
            ADJACENTS[_i].append(_i + _off)
ADJACENTS = [tuple(a) for a in ADJACENTS]

MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        MANHATTAN[_a][_b] = abs((_a>>3)-(_b>>3)) + abs((_a&7)-(_b&7))

INF = 999999

# Default weights (hand-tuned starting point for TD-Leaf training)
DEFAULT_WEIGHTS = np.array([
    6.0,    # 0: material_diff
    12.0,   # 1: material_ratio
    1.0,    # 2: proximity (per unit — gets large in aggregate)
    -4.0,   # 3: my_threatened (negative = penalty)
    4.0,    # 4: opp_threatened
    -2.0,   # 5: my_debris (negative = penalty)
    2.0,    # 6: opp_debris
    3.0,    # 7: dominance
    5.0,    # 8: height_advantage
    -5.0,   # 9: height_disadvantage (negative = penalty)
    0.5,    # 10: height_preservation
    1.0,    # 11: edge_trap_bonus
    -1.0,   # 12: edge_trap_penalty (negative = penalty)
    0.15,   # 13: safety_diff
    1.0,    # 14: elim_pressure
    0.3,    # 15: my_stack_count
    -0.3,   # 16: opp_stack_count
], dtype=np.float64)


def extract_features(board: BoardState, is_red_maximizing: bool) -> np.ndarray:
    """
    Extract feature vector from board state.
    Returns numpy array of shape (NUM_FEATURES,).
    """
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens

    my_flag  = RED_FLAG if is_red_maximizing else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    state = board.state

    my_tokens  = red_tokens if is_red_maximizing else blue_tokens
    opp_tokens = blue_tokens if is_red_maximizing else red_tokens

    my_stacks = []
    opp_stacks = []
    for idx in range(64):
        val = state[idx]
        if val == 0: continue
        h = val & HEIGHT_MASK
        if val & my_flag: my_stacks.append((idx, h))
        else: opp_stacks.append((idx, h))

    features = np.zeros(NUM_FEATURES, dtype=np.float64)

    total = my_tokens + opp_tokens
    if total == 0:
        return features

    # 0: material_diff
    features[0] = my_tokens - opp_tokens

    # 1: material_ratio
    features[1] = (my_tokens - opp_tokens) / total

    # 2: proximity (height-weighted)
    prox = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            if mh >= oh:
                surplus = mh - oh
                d = MANHATTAN[mi][oi]
                prox += (1 + surplus) * max(0, 8 - d) * 0.3
    features[2] = prox

    # 3: my_threatened (strictly taller adjacent enemies)
    my_thr = 0.0
    for mi, mh in my_stacks:
        for nb in ADJACENTS[mi]:
            nb_val = state[nb]
            if nb_val and (nb_val & opp_flag) and (nb_val & HEIGHT_MASK) > mh:
                my_thr += mh
                break
    features[3] = my_thr

    # 4: opp_threatened
    opp_thr = 0.0
    for oi, oh in opp_stacks:
        for nb in ADJACENTS[oi]:
            nb_val = state[nb]
            if nb_val and (nb_val & my_flag) and (nb_val & HEIGHT_MASK) > oh:
                opp_thr += oh
                break
    features[4] = opp_thr

    # 5: my_debris (h=1 stacks near enemies)
    my_deb = 0.0
    for mi, mh in my_stacks:
        if mh > 1: continue
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d <= 1: my_deb += 3.0; break
            elif d <= 2: my_deb += 2.0; break
            elif d <= 3: my_deb += 1.0; break
    features[5] = my_deb

    # 6: opp_debris
    opp_deb = 0.0
    for oi, oh in opp_stacks:
        if oh > 1: continue
        for mi, mh in my_stacks:
            d = MANHATTAN[oi][mi]
            if d <= 1: opp_deb += 3.0; break
            elif d <= 2: opp_deb += 2.0; break
            elif d <= 3: opp_deb += 1.0; break
    features[6] = opp_deb

    # 7-10: height features
    opp_max_h = max((h for _, h in opp_stacks), default=0)
    my_max_h = max((h for _, h in my_stacks), default=0)

    features[7] = sum(1 for _, h in my_stacks if h > opp_max_h) - \
                  sum(1 for _, h in opp_stacks if h > my_max_h)
    features[8] = max(0, my_max_h - opp_max_h)
    features[9] = max(0, opp_max_h - my_max_h)
    features[10] = sum(h*h for _, h in my_stacks) - sum(h*h for _, h in opp_stacks)

    # 11-12: edge trap
    et_bonus = 0.0
    for oi, oh in opp_stacks:
        if EDGE_DIST[oi] > 2: continue
        for d_idx in range(4):
            opp_far = FAR_EDGE_DIST[oi * 4 + d_idx]
            if opp_far > 2: continue
            scan_off = SCAN_OFFSETS[d_idx]
            pos = oi
            while is_valid_coord(pos, scan_off):
                pos += scan_off
                sv = state[pos]
                if sv == 0: continue
                if sv & my_flag:
                    if (sv & HEIGHT_MASK) >= FAR_EDGE_DIST[pos * 4 + d_idx]:
                        et_bonus += oh * max(0, 6 - MANHATTAN[oi][pos])
                break
    features[11] = et_bonus

    et_penalty = 0.0
    for mi, mh in my_stacks:
        if EDGE_DIST[mi] > 2: continue
        for d_idx in range(4):
            my_far = FAR_EDGE_DIST[mi * 4 + d_idx]
            if my_far > 2: continue
            scan_off = SCAN_OFFSETS[d_idx]
            pos = mi
            while is_valid_coord(pos, scan_off):
                pos += scan_off
                sv = state[pos]
                if sv == 0: continue
                if sv & opp_flag:
                    opp_h = sv & HEIGHT_MASK
                    if mh > opp_h: break  # we're taller, no danger
                    if opp_h >= FAR_EDGE_DIST[pos * 4 + d_idx]:
                        et_penalty += mh * max(0, 6 - MANHATTAN[mi][pos])
                break
    features[12] = et_penalty

    # 13: safety_diff
    my_saf = sum(EDGE_DIST[i] * h for i, h in my_stacks)
    opp_saf = sum(EDGE_DIST[i] * h for i, h in opp_stacks)
    features[13] = my_saf - opp_saf

    # 14: elimination pressure
    mc, oc = len(my_stacks), len(opp_stacks)
    elim = 0.0
    if oc <= 1: elim += 5.0
    elif oc == 2: elim += 2.0
    if mc <= 1: elim -= 5.0
    elif mc == 2: elim -= 2.0
    features[14] = elim

    # 15-16: stack counts
    features[15] = mc
    features[16] = oc

    return features


def evaluate_linear(board: BoardState, is_red_maximizing: bool,
                    weights: np.ndarray = None) -> float:
    """
    Linear evaluation: score = dot(weights, features).
    Terminal states return ±INF.
    """
    if board.red_tokens == 0:
        return -INF if is_red_maximizing else INF
    if board.blue_tokens == 0:
        return INF if is_red_maximizing else -INF

    if weights is None:
        weights = DEFAULT_WEIGHTS

    features = extract_features(board, is_red_maximizing)
    return float(np.dot(weights, features))

