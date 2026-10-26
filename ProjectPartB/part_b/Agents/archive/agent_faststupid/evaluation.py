"""
Evaluation function for Cascade — v11: quiescence + anti-retreat.

CRITICAL FIXES:
  1. QUIESCENCE: If the side to move has an available eat, evaluate
     the POST-EAT position. This gives MCTS an implicit extra ply
     for tactical positions. Without this, MCTS can't see "I eat R3
     next turn" because it only evaluates static positions.

  2. ANTI-RETREAT: When we're the tallest stack, moving away from
     enemies is TERRIBLE. We should always be closing distance.
     The "convergence" bonus is now massive — strong enough to
     override any safety concerns.

  3. CORNER DEATH: Being pushed into a corner with enemies nearby
     is a death sentence (cascade push-off). Explicit huge penalty.

  4. MOBILITY CRISIS: Having very few moves = about to be cornered
     and killed. Severe penalty scales with how few options remain.

  5. All the good stuff from v10 (initiative, vulnerability, 
     stack-count-aware threats, last-stand logic).
"""
from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import is_valid_coord, UP, DOWN, LEFT, RIGHT, MOVE, EAT, CASCADE
from Agents.common.action import move, eat

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


def _quiescence_eval(board, is_red_maximizing):
    """
    If the side to move has a safe eat available, evaluate the
    post-eat position instead. This gives a 1-ply tactical extension.
    
    "Safe eat" = we can eat an enemy and no remaining enemy can eat
    us back at the destination.
    
    This is THE critical fix: without it, MCTS evaluates "B9 adjacent
    to R3" the same as "B9 far from R3" — it can't see the eat.
    With quiescence, "B9 adjacent to R3" evaluates as "B9 has eaten
    R3 → B9 vs R6, we're winning."
    """
    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    state = board.state

    best_eat_board = None
    best_eat_gain = 0

    for idx in range(64):
        val = state[idx]
        if not (val & my_flag): continue
        h = val & HEIGHT_MASK
        for offset in OFFSETS:
            if not is_valid_coord(idx, offset): continue
            tv = state[idx + offset]
            if not (tv & opp_flag): continue
            th = tv & HEIGHT_MASK
            if h < th: continue

            # We can eat. Check if safe (no enemy recapture).
            dest = idx + offset
            nb = eat(board, idx, offset)

            # Check recapture
            safe = True
            for check_off in OFFSETS:
                if not is_valid_coord(dest, check_off): continue
                cv = nb.state[dest + check_off]
                if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= h:
                    safe = False
                    break

            if safe and th > best_eat_gain:
                best_eat_gain = th
                best_eat_board = nb

    return best_eat_board


def evaluate(board: BoardState, is_red_maximizing: bool) -> float:
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens

    if red_tokens == 0: return -INF if is_red_maximizing else INF
    if blue_tokens == 0: return INF if is_red_maximizing else -INF

    # ── QUIESCENCE: check for available safe eat ─────────
    # If we have a safe eat, evaluate as if we took it.
    # This makes positions like "B9 adjacent to R3" evaluate as
    # "B9 has eaten R3 → material advantage" instead of "static equal."
    q_board = _quiescence_eval(board, is_red_maximizing)
    if q_board is not None:
        # Evaluate post-eat position from OPPONENT's perspective,
        # then negate (since after our eat, it's their turn).
        # But we want the score from OUR perspective, so we evaluate
        # the resulting board for us, with a small bonus for having
        # the capture available (tempo).
        return _static_eval(q_board, is_red_maximizing) + 3.0

    return _static_eval(board, is_red_maximizing)


def _static_eval(board: BoardState, is_red_maximizing: bool) -> float:
    """Core static evaluation — called after quiescence resolution."""
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

    my_stack_n = len(my_stacks)
    opp_stack_n = len(opp_stacks)

    if not my_stacks: return -INF
    if not opp_stacks: return INF

    total = my_tokens + opp_tokens
    if total == 0: return 0.0

    advantage = (my_tokens - opp_tokens) / total
    late_factor = max(0.0, 1.0 - total / 16.0)

    # ── GAME PHASE WEIGHTS ───────────────────────────────
    if my_stack_n <= 1:
        survival_weight = 2.5
        elimination_weight = 0.5
    elif my_stack_n == 2 and advantage < 0:
        survival_weight = 2.0
        elimination_weight = 0.5
    elif advantage < -0.1:
        survival_weight = 1.5 + late_factor * 0.5
        elimination_weight = 0.6
    elif advantage > 0.2:
        survival_weight = 0.8 - late_factor * 0.3
        elimination_weight = 1.5 + late_factor * 0.5
    else:
        survival_weight = 1.0
        elimination_weight = 1.0

    # ── Precompute commonly needed values ────────────────
    my_max_h = max(h for _, h in my_stacks)
    opp_max_h = max(h for _, h in opp_stacks)
    we_are_tallest = my_max_h > opp_max_h

    # ── 1. MATERIAL ──────────────────────────────────────
    material_diff = my_tokens - opp_tokens
    material_ratio = material_diff / total

    # ── 2. PROXIMITY (vulnerability-aware) ───────────────
    proximity = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d > 8: continue

            if mh > oh:
                surplus = mh - oh
                dist_bonus = max(0, 8 - d)
                # Vulnerability: isolated small stacks are prime targets
                vuln = 1.0
                if oh <= 2:
                    vuln = 2.0
                    isolated = True
                    for oi2, oh2 in opp_stacks:
                        if oi2 != oi and MANHATTAN[oi][oi2] <= 2:
                            isolated = False; break
                    if isolated:
                        vuln = 3.0
                proximity += (1 + surplus) * dist_bonus * vuln * 0.25

            elif mh == oh and advantage > 0.1:
                dist_bonus = max(0, 8 - d)
                proximity += dist_bonus * 0.08

    # ── 3. IMMEDIATE THREATS (stack-count aware) ─────────
    my_threatened = 0.0
    opp_threatened = 0.0
    for mi, mh in my_stacks:
        worst_threat = 0
        for nb in ADJACENTS[mi]:
            nb_val = state[nb]
            if nb_val and (nb_val & opp_flag):
                nb_h = nb_val & HEIGHT_MASK
                if nb_h > mh: worst_threat = 2; break
                elif nb_h >= mh and worst_threat < 1: worst_threat = 1
        if worst_threat == 2:
            my_threatened += mh
        elif worst_threat == 1:
            if my_stack_n <= 1: my_threatened += mh * 1.5
            elif my_stack_n == 2: my_threatened += mh * 0.8
            else: my_threatened += mh * 0.3

    for oi, oh in opp_stacks:
        worst_threat = 0
        for nb in ADJACENTS[oi]:
            nb_val = state[nb]
            if nb_val and (nb_val & my_flag):
                nb_h = nb_val & HEIGHT_MASK
                if nb_h > oh: worst_threat = 2; break
                elif nb_h >= oh and worst_threat < 1: worst_threat = 1
        if worst_threat == 2:
            opp_threatened += oh
        elif worst_threat == 1:
            if opp_stack_n <= 1: opp_threatened += oh * 1.5
            elif opp_stack_n == 2: opp_threatened += oh * 0.8
            else: opp_threatened += oh * 0.3

    # ── 4. INITIATIVE ────────────────────────────────────
    my_threats = 0.0
    opp_threats = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d > 3: continue
            if mh > oh:
                if d <= 1:   my_threats += oh * 3.0
                elif d == 2: my_threats += oh * 1.5
                elif d == 3: my_threats += oh * 0.5
    for oi, oh in opp_stacks:
        for mi, mh in my_stacks:
            d = MANHATTAN[oi][mi]
            if d > 3: continue
            if oh > mh:
                if d <= 1:   opp_threats += mh * 3.0
                elif d == 2: opp_threats += mh * 1.5
                elif d == 3: opp_threats += mh * 0.5
    initiative = my_threats - opp_threats

    # ── 5. CASCADE DEBRIS PENALTY ────────────────────────
    my_debris_penalty = 0.0
    opp_debris_penalty = 0.0
    for mi, mh in my_stacks:
        if mh > 1: continue
        closest = 99
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d < closest: closest = d
        if   closest <= 1: my_debris_penalty += 3.0
        elif closest <= 2: my_debris_penalty += 2.0
        elif closest <= 3: my_debris_penalty += 1.0

    for oi, oh in opp_stacks:
        if oh > 1: continue
        closest = 99
        for mi, mh in my_stacks:
            d = MANHATTAN[oi][mi]
            if d < closest: closest = d
        if   closest <= 1: opp_debris_penalty += 3.0
        elif closest <= 2: opp_debris_penalty += 2.0
        elif closest <= 3: opp_debris_penalty += 1.0

    # ── 6. HEIGHT DOMINANCE & PRESERVATION ───────────────
    my_dominant = sum(1 for _, h in my_stacks if h > opp_max_h)
    opp_dominant = sum(1 for _, h in opp_stacks if h > my_max_h)
    dominance = (my_dominant - opp_dominant) * 3.0

    height_adv = max(0, my_max_h - opp_max_h) * 5.0
    height_disadv = max(0, opp_max_h - my_max_h) * 5.0

    my_height_sq = sum(h * h for _, h in my_stacks)
    opp_height_sq = sum(h * h for _, h in opp_stacks)
    height_preservation = (my_height_sq - opp_height_sq) * 0.5

    # ── 7. EDGE-TRAP BONUS ───────────────────────────────
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
                    # BUG FIX: Do NOT skip when we're taller!
                    # Being taller does NOT protect from cascade push-off.
                    # R4 cascading right pushes B6 off the edge even though
                    # B6 height 6 > R4 height 4.
                    #
                    # Only skip if we're adjacent (dist=1) AND strictly
                    # taller — we can eat them before they cascade. Even
                    # this is risky (it might be their turn), so use a
                    # reduced penalty instead of skipping entirely.
                    opp_far = FAR_EDGE_DIST[pos * 4 + d_idx]
                    if opp_h >= opp_far:
                        urgency = max(0, 6 - dist_from_me)
                        if dist_from_me == 1 and mh > opp_h:
                            # Adjacent and taller: reduced threat (we might eat first)
                            edge_trap_penalty += mh * urgency * 0.3
                        else:
                            # Full threat: cascade can push us off
                            edge_trap_penalty += mh * urgency * 1.5
                break

    # ── 8. POSITIONAL SAFETY ─────────────────────────────
    my_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in my_stacks)
    opp_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in opp_stacks)

    # ── 9. MOBILITY ──────────────────────────────────────
    my_mobility = 0
    opp_mobility = 0
    for mi, mh in my_stacks:
        for nb in ADJACENTS[mi]:
            nb_val = state[nb]
            if nb_val == 0:      my_mobility += 1
            elif nb_val & my_flag: my_mobility += 1
            elif nb_val & opp_flag:
                if mh >= (nb_val & HEIGHT_MASK): my_mobility += 2
    for oi, oh in opp_stacks:
        for nb in ADJACENTS[oi]:
            nb_val = state[nb]
            if nb_val == 0:       opp_mobility += 1
            elif nb_val & opp_flag: opp_mobility += 1
            elif nb_val & my_flag:
                if oh >= (nb_val & HEIGHT_MASK): opp_mobility += 2
    mobility_diff = (my_mobility - opp_mobility) * 0.4

    # ── 10. EDGE/CORNER DEATH PENALTY ───────────────────
    # When we're the SHORTER stack, being near edges is catastrophic:
    #   - Fewer escape directions (corner=2, edge=3, center=4)
    #   - Vulnerable to cascade push-off
    #   - One wrong step = cornered and dead
    #
    # This is NOT just a "corner" problem. ANY edge cell is dangerous
    # when fleeing. The penalty scales with:
    #   - How close to edge (ed=0 worst, ed=3 safe)
    #   - Stack height (losing B9 on edge = game over)
    #   - How close the nearest enemy is (edge + nearby enemy = death)
    #
    # When we're TALLER, edges are fine (we're chasing, not fleeing).
    # Reward ENEMY being on edges instead.
    edge_death_penalty = 0.0
    edge_death_bonus = 0.0

    for mi, mh in my_stacks:
        ed = EDGE_DIST[mi]
        if ed >= 3: continue  # center is safe

        # Find nearest enemy
        min_enemy_dist = 99
        nearest_enemy_h = 0
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d < min_enemy_dist:
                min_enemy_dist = d
                nearest_enemy_h = oh

        # Penalty depends on whether we're shorter (fleeing)
        if nearest_enemy_h > mh:
            # We're SHORTER — edge is a death trap
            # ed=0 (corner): penalty = mh * 12 * proximity_factor
            # ed=1 (edge):   penalty = mh * 8 * proximity_factor
            # ed=2 (near):   penalty = mh * 4 * proximity_factor
            edge_danger = (3 - ed) * 4.0
            # Closer enemy = more urgent
            if min_enemy_dist <= 2:
                prox_factor = 3.0   # imminent death
            elif min_enemy_dist <= 4:
                prox_factor = 2.0   # approaching danger
            elif min_enemy_dist <= 6:
                prox_factor = 1.0   # distant but still bad
            else:
                prox_factor = 0.5   # far away, mild penalty
            edge_death_penalty += mh * edge_danger * prox_factor
        elif nearest_enemy_h == mh:
            # Equal height on edge: also bad (they can eat us)
            edge_danger = (3 - ed) * 2.0
            if min_enemy_dist <= 3:
                edge_death_penalty += mh * edge_danger * 1.5

    for oi, oh in opp_stacks:
        ed = EDGE_DIST[oi]
        if ed >= 3: continue

        min_enemy_dist = 99
        nearest_my_h = 0
        for mi, mh in my_stacks:
            d = MANHATTAN[mi][oi]
            if d < min_enemy_dist:
                min_enemy_dist = d
                nearest_my_h = mh

        if nearest_my_h > oh:
            # Enemy is shorter and on edge — great for us
            edge_danger = (3 - ed) * 4.0
            if min_enemy_dist <= 2:   prox_factor = 3.0
            elif min_enemy_dist <= 4: prox_factor = 2.0
            elif min_enemy_dist <= 6: prox_factor = 1.0
            else:                     prox_factor = 0.5
            edge_death_bonus += oh * edge_danger * prox_factor
        elif nearest_my_h == oh:
            edge_danger = (3 - ed) * 2.0
            if min_enemy_dist <= 3:
                edge_death_bonus += oh * edge_danger * 1.5

    # ── 11. CONVERGENCE (anti-stall + smart flee) ────────
    # When tallest: REWARD closing distance to killable enemies.
    # When shortest: REWARD distance from enemies AND being centered.
    #
    # The CENTER-SEEKING is critical. "Run from enemy" without
    # "run TOWARD center" causes corner death. The agent must
    # prefer (3,3) dist=4 over (0,7) dist=5 because (3,3) has
    # 4 escape directions while (0,7) has 2.
    convergence = 0.0
    if my_stacks and opp_stacks:
        if we_are_tallest:
            min_kill_dist = 99
            for mi, mh in my_stacks:
                if mh != my_max_h: continue
                for oi, oh in opp_stacks:
                    if mh > oh:
                        d = MANHATTAN[mi][oi]
                        if d < min_kill_dist:
                            min_kill_dist = d
            if min_kill_dist < 99:
                convergence = max(0, 8 - min_kill_dist) * 3.0
        elif opp_max_h > my_max_h:
            # FLEE MODE: reward distance AND centrality.
            # 
            # Formula: distance_score + center_score
            #   distance_score: how far from the nearest taller enemy
            #   center_score: EDGE_DIST of our position (0=corner, 3=center)
            #
            # Center score is weighted HEAVILY because:
            #   B3 at (0,7) dist=5 from R5: distance=5, center=0 → 5+0=5
            #   B3 at (3,4) dist=4 from R5: distance=4, center=3 → 4+9=13
            # The center position wins despite being closer.
            for mi, mh in my_stacks:
                min_threat_dist = 99
                for oi, oh in opp_stacks:
                    if oh > mh:
                        d = MANHATTAN[mi][oi]
                        if d < min_threat_dist:
                            min_threat_dist = d
                if min_threat_dist < 99:
                    ed = EDGE_DIST[mi]
                    # Distance from enemy: capped at 7
                    dist_score = min(min_threat_dist, 7) * 1.5
                    # Center bonus: ed=3 → +9, ed=2 → +6, ed=1 → +3, ed=0 → 0
                    # This is THE key weight. Must be strong enough that
                    # center at dist=4 beats corner at dist=6.
                    center_score = ed * 3.0
                    convergence += dist_score + center_score

    # ── 12. STRUCTURE ────────────────────────────────────
    elim = 0.0
    if opp_stack_n <= 1: elim += 5.0
    elif opp_stack_n == 2: elim += 2.0
    if my_stack_n <= 1: elim -= 5.0
    elif my_stack_n == 2: elim -= 2.0

    # ── COMBINE ──────────────────────────────────────────
    score = (
        material_diff * 6.0
        + material_ratio * 12.0

        + proximity * elimination_weight

        - my_threatened * 4.0 * survival_weight
        + opp_threatened * 4.0 * elimination_weight

        + initiative * 2.0

        - my_debris_penalty * survival_weight
        + opp_debris_penalty * elimination_weight

        + dominance
        + height_adv - height_disadv
        + height_preservation

        + edge_trap_bonus * elimination_weight
        - edge_trap_penalty * survival_weight

        + mobility_diff

        - edge_death_penalty * survival_weight
        + edge_death_bonus * elimination_weight

        + convergence

        + elim * elimination_weight
        + (my_safety - opp_safety) * survival_weight
    )
    return score