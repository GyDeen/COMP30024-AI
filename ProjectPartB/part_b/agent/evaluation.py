"""
Evaluation function for Cascade — v8: game-phase-aware.

GAME PHASE STRATEGY:
  Early game (both sides 8+ tokens): Slight survival bias.
    Don't sacrifice stacks for marginal gains. Build position.
  
  Mid game (5-10 tokens each): Balanced.
    Trade when profitable (sacrifice R1 to eat B6 = great).
    Avoid trades that are even or slightly positive.
  
  Late game winning (we have more tokens): Elimination focus.
    Actively seek captures and cascade push-offs. Close out.
  
  Late game losing (we have fewer tokens): Survival focus.
    Avoid trades. Keep stacks alive and tall. Look for reversals.

KEY FIXES FROM v7:
  1. Game-phase multipliers on survival vs elimination weights
  2. Much stronger penalty for h=1 stacks near enemies (anti-cascade-debris)
  3. Height preservation bonus (tall stacks are strategically superior)
"""
from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from .action import is_valid_coord, UP, DOWN, LEFT, RIGHT

INF = 999999
OFFSETS = (UP, DOWN, LEFT, RIGHT)

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


def evaluate(board: BoardState, is_red_maximizing: bool) -> float:
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens

    if red_tokens == 0: return -INF if is_red_maximizing else INF
    if blue_tokens == 0: return INF if is_red_maximizing else -INF

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

    # ── GAME PHASE ───────────────────────────────────────
    total = my_tokens + opp_tokens
    if total == 0: return 0.0

    # Who's winning? Positive = we're ahead
    advantage = (my_tokens - opp_tokens) / total  # range [-1, 1]

    # Phase: early (lots of tokens) vs late (few tokens)
    # total starts at 24 (12+12). Below 12 = late game.
    late_factor = max(0.0, 1.0 - total / 16.0)  # 0.0 early, ~0.5 mid, ~1.0 late

    # Survival multiplier: how much to weight keeping our stacks alive
    # Early game or losing: high. Late winning: low.
    if advantage < -0.1:
        # We're losing: survival is critical
        survival_weight = 1.5 + late_factor * 0.5  # 1.5 to 2.0
    elif advantage > 0.2:
        # We're winning: focus on elimination
        survival_weight = 0.8 - late_factor * 0.3  # 0.8 to 0.5
    else:
        # Close game: balanced
        survival_weight = 1.0

    # Elimination multiplier: how aggressively to pursue kills
    if advantage > 0.2:
        elimination_weight = 1.5 + late_factor * 0.5
    elif advantage < -0.1:
        elimination_weight = 0.6
    else:
        elimination_weight = 1.0

    # ── 1. MATERIAL ──────────────────────────────────────
    material_diff = my_tokens - opp_tokens
    material_ratio = material_diff / total

    # ── 2. HEIGHT-WEIGHTED PROXIMITY ─────────────────────
    proximity = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            if mh >= oh:
                surplus = mh - oh
                d = MANHATTAN[mi][oi]
                dist_bonus = max(0, 8 - d)
                proximity += (1 + surplus) * dist_bonus * 0.3

    # ── 3. IMMEDIATE THREATS ─────────────────────────────
    # Adjacent enemy that can eat us = imminent loss.
    # But EQUAL height is less dangerous — it's a mutual threat.
    # Strictly taller enemy is the real danger (they eat us, we can't eat back).
    my_threatened = 0.0
    opp_threatened = 0.0
    for mi, mh in my_stacks:
        worst_threat = 0  # 0=none, 1=equal, 2=taller
        for nb in ADJACENTS[mi]:
            nb_val = state[nb]
            if nb_val and (nb_val & opp_flag):
                nb_h = nb_val & HEIGHT_MASK
                if nb_h > mh:
                    worst_threat = 2; break  # strictly taller = real danger
                elif nb_h == mh and worst_threat < 1:
                    worst_threat = 1  # equal = mutual threat
        if worst_threat == 2:
            my_threatened += mh        # full penalty
        elif worst_threat == 1:
            my_threatened += mh * 0.3  # mild penalty (we can eat back)

    for oi, oh in opp_stacks:
        worst_threat = 0
        for nb in ADJACENTS[oi]:
            nb_val = state[nb]
            if nb_val and (nb_val & my_flag):
                nb_h = nb_val & HEIGHT_MASK
                if nb_h > oh:
                    worst_threat = 2; break
                elif nb_h == oh and worst_threat < 1:
                    worst_threat = 1
        if worst_threat == 2:
            opp_threatened += oh
        elif worst_threat == 1:
            opp_threatened += oh * 0.3

    # ── 4. CASCADE DEBRIS PENALTY ────────────────────────
    # h=1 stacks near ANY enemy (not just adjacent) are easy kills.
    # This is what makes cascading into enemy territory so bad:
    # each R1 created is a sitting duck for any nearby enemy.
    #
    # Penalty per h=1 stack: based on closest enemy distance.
    # Adjacent enemy: very high penalty (will be eaten next turn)
    # Enemy 2 cells away: moderate (eaten in 2 turns)
    # Enemy 3+ cells: minor
    my_debris_penalty = 0.0
    opp_debris_penalty = 0.0

    for mi, mh in my_stacks:
        if mh > 1: continue  # only h=1 stacks are "debris"
        closest_threat = 99
        for oi, oh in opp_stacks:
            if oh >= 1:  # anything can eat h=1
                d = MANHATTAN[mi][oi]
                if d < closest_threat:
                    closest_threat = d
        if closest_threat <= 1:
            my_debris_penalty += 3.0  # adjacent: dead next turn
        elif closest_threat <= 2:
            my_debris_penalty += 2.0  # 2 away: dead in 2 turns
        elif closest_threat <= 3:
            my_debris_penalty += 1.0  # 3 away: vulnerable

    for oi, oh in opp_stacks:
        if oh > 1: continue
        closest_threat = 99
        for mi, mh in my_stacks:
            if mh >= 1:
                d = MANHATTAN[oi][mi]
                if d < closest_threat:
                    closest_threat = d
        if closest_threat <= 1:
            opp_debris_penalty += 3.0
        elif closest_threat <= 2:
            opp_debris_penalty += 2.0
        elif closest_threat <= 3:
            opp_debris_penalty += 1.0

    # ── 5. HEIGHT DOMINANCE & PRESERVATION ───────────────
    opp_max_h = max((h for _, h in opp_stacks), default=0)
    my_max_h = max((h for _, h in my_stacks), default=0)

    my_dominant = sum(1 for _, h in my_stacks if h > opp_max_h)
    opp_dominant = sum(1 for _, h in opp_stacks if h > my_max_h)
    dominance = (my_dominant - opp_dominant) * 3.0
    
    # Height advantage: being strictly taller means we can safely eat them.
    # This is HUGE — R7 vs B6 means we can approach and eat without risk.
    # Weight 5.0: merging R6+R1→R7 gains +5 from this alone, making
    # the merge clearly worth the R1 investment.
    height_adv = max(0, my_max_h - opp_max_h) * 5.0
    height_disadv = max(0, opp_max_h - my_max_h) * 5.0

    # Height preservation: reward having tall stacks, penalize many small ones.
    # Sum of squares rewards concentration: one h=6 scores 36, six h=1 scores 6.
    # Weight 0.5: strong enough to prevent pointless cascades.
    # R3→3×R1: height_sq drops from 9 to 3, penalty = -6 × 0.5 = -3.0
    my_height_sq = sum(h * h for _, h in my_stacks)
    opp_height_sq = sum(h * h for _, h in opp_stacks)
    height_preservation = (my_height_sq - opp_height_sq) * 0.5

    # ── 6. EDGE-TRAP BONUS ───────────────────────────────
    edge_trap_bonus = 0.0
    edge_trap_penalty = 0.0

    for oi, oh in opp_stacks:
        if EDGE_DIST[oi] > 2: continue
        for d_idx in range(4):
            opp_far = FAR_EDGE_DIST[oi * 4 + d_idx]
            if opp_far > 2: continue
            scan_off = SCAN_OFFSETS[d_idx]
            pos = oi; dist_from_opp = 0
            while is_valid_coord(pos, scan_off):
                pos += scan_off; dist_from_opp += 1
                sv = state[pos]
                if sv == 0: continue
                if sv & my_flag:
                    our_far = FAR_EDGE_DIST[pos * 4 + d_idx]
                    if (sv & HEIGHT_MASK) >= our_far:
                        urgency = max(0, 6 - dist_from_opp)
                        edge_trap_bonus += oh * urgency * 1.0
                break

    for mi, mh in my_stacks:
        if EDGE_DIST[mi] > 2: continue
        for d_idx in range(4):
            my_far = FAR_EDGE_DIST[mi * 4 + d_idx]
            if my_far > 2: continue
            scan_off = SCAN_OFFSETS[d_idx]
            pos = mi; dist_from_me = 0
            while is_valid_coord(pos, scan_off):
                pos += scan_off; dist_from_me += 1
                sv = state[pos]
                if sv == 0: continue
                if sv & opp_flag:
                    opp_h = sv & HEIGHT_MASK
                    # SKIP if we're taller — we eat them before they cascade us
                    if mh > opp_h:
                        break
                    opp_far = FAR_EDGE_DIST[pos * 4 + d_idx]
                    if opp_h >= opp_far:
                        urgency = max(0, 6 - dist_from_me)
                        edge_trap_penalty += mh * urgency * 1.0
                break

    # ── 7. POSITIONAL SAFETY ─────────────────────────────
    my_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in my_stacks)
    opp_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in opp_stacks)

    # ── 8. STRUCTURE ─────────────────────────────────────
    mc, oc = len(my_stacks), len(opp_stacks)

    elim = 0.0
    if oc <= 1: elim += 5.0
    elif oc == 2: elim += 2.0
    if mc <= 1: elim -= 5.0
    elif mc == 2: elim -= 2.0

    # ── COMBINE WITH PHASE WEIGHTS ───────────────────────
    score = (
        # Material (always important)
        material_diff * 6.0
        + material_ratio * 12.0

        # Approach (scaled by elimination weight)
        + proximity * elimination_weight

        # Survival (scaled by survival weight)
        - my_threatened * 4.0 * survival_weight
        + opp_threatened * 4.0 * elimination_weight

        # Anti-cascade-debris (scaled by survival weight)
        - my_debris_penalty * survival_weight
        + opp_debris_penalty * elimination_weight

        # Height (always important — tall stacks are better)
        + dominance
        + height_adv - height_disadv
        + height_preservation

        # Edge traps (scaled by elimination weight)
        + edge_trap_bonus * elimination_weight
        - edge_trap_penalty * survival_weight

        # Position and structure
        + elim * elimination_weight
        + (my_safety - opp_safety) * survival_weight
    )
    return score