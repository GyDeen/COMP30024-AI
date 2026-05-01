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
from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from .action import is_valid_coord, UP, DOWN, LEFT, RIGHT, MOVE, EAT, CASCADE
from .action import move, eat, cascade

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

            dest = idx + offset
            nb = eat(board, idx, offset)

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


def _opponent_cascade_threat(board, is_red_maximizing):
    """
    Check if the OPPONENT has a cascade that destroys significant tokens.
    
    This is the critical missing piece: without it, MCTS can't see
    "if I put B6 at (4,5), Red cascades R6 right and I'm dead."
    
    Returns (worst_board, our_loss) where worst_board is the post-cascade
    position and our_loss is how many of our tokens get destroyed.
    Returns (None, 0) if no significant cascade threat exists.
    """
    # Check from opponent's perspective
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    state = board.state
    
    my_tokens = board.red_tokens if is_red_maximizing else board.blue_tokens
    opp_tokens = board.blue_tokens if is_red_maximizing else board.red_tokens
    
    worst_board = None
    worst_loss = 0  # most tokens WE lose
    
    for idx in range(64):
        val = state[idx]
        if not (val & opp_flag): continue
        h = val & HEIGHT_MASK
        if h < 2: continue  # need height >= 2 to cascade
        
        for offset in OFFSETS:
            nb = cascade(board, idx, offset)
            
            # How many of OUR tokens got destroyed?
            nb_my = nb.red_tokens if is_red_maximizing else nb.blue_tokens
            my_lost = my_tokens - nb_my
            
            # How many of THEIR tokens got destroyed? (off-edge cost)
            nb_opp = nb.blue_tokens if is_red_maximizing else nb.red_tokens
            opp_lost = opp_tokens - nb_opp
            
            # Net damage to us (their gain)
            net_damage = my_lost - opp_lost
            
            if net_damage > worst_loss:
                worst_loss = net_damage
                worst_board = nb
            
            # Also check instant kill
            if nb_my == 0:
                return nb, my_tokens
    
    return worst_board, worst_loss


def evaluate(board: BoardState, is_red_maximizing: bool) -> float:
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens

    if red_tokens == 0: return -INF if is_red_maximizing else INF
    if blue_tokens == 0: return INF if is_red_maximizing else -INF

    # ── QUIESCENCE PHASE 1: Can we eat safely? ───────────
    q_board = _quiescence_eval(board, is_red_maximizing)
    if q_board is not None:
        return _static_eval(q_board, is_red_maximizing) + 3.0

    # ── QUIESCENCE PHASE 2: Can opponent cascade-kill us? ─
    # Check if any opponent stack can cascade (height >= 2).
    # BUG FIX: old check (our stacks near edge?) was WRONG.
    # R9 at center can be cascade-pushed through 4+ cells off the edge.
    opp_flag_check = BLUE_FLAG if is_red_maximizing else RED_FLAG
    any_opp_can_cascade = False
    for idx in range(64):
        v = board.state[idx]
        if v and (v & opp_flag_check) and (v & HEIGHT_MASK) >= 2:
            any_opp_can_cascade = True
            break

    if any_opp_can_cascade:
        cascade_board, cascade_loss = _opponent_cascade_threat(
            board, is_red_maximizing)
        
        if cascade_loss > 0 and cascade_board is not None:
            static_score = _static_eval(board, is_red_maximizing)
            
            # Instant death: return near -INF directly
            cb_my = cascade_board.red_tokens if is_red_maximizing else cascade_board.blue_tokens
            if cb_my == 0:
                return -INF + 100  # slightly above -INF to allow comparison
            
            post_cascade_score = _static_eval(cascade_board, is_red_maximizing)
            
            if cascade_loss >= 3:
                return post_cascade_score * 0.7 + static_score * 0.3
            elif cascade_loss >= 1:
                return post_cascade_score * 0.4 + static_score * 0.6

    # ── QUIESCENCE PHASE 3: Can WE cascade profitably? ──
    # Check if the side to move has a cascade that destroys significant
    # opponent tokens. This catches "B6 cascade LEFT kills R9" — 
    # the position is great for Blue, not neutral.
    #
    # Without this, MCTS evaluates "R9(3,4) next to B6(3,5)" from
    # Blue's perspective as "Red has material advantage" (static eval).
    # But Blue can cascade and kill R9! The position should score
    # near +INF for Blue, making MCTS see it as terrible for Red.
    my_flag_q3 = RED_FLAG if is_red_maximizing else BLUE_FLAG
    any_we_can_cascade = False
    for idx in range(64):
        v = board.state[idx]
        if v and (v & my_flag_q3) and (v & HEIGHT_MASK) >= 2:
            any_we_can_cascade = True
            break

    if any_we_can_cascade:
        # Check our cascades that destroy opponent tokens
        opp_flag_q3 = BLUE_FLAG if is_red_maximizing else RED_FLAG
        my_tokens_q3 = board.red_tokens if is_red_maximizing else board.blue_tokens
        opp_tokens_q3 = board.blue_tokens if is_red_maximizing else board.red_tokens
        
        best_our_cascade = None
        best_net_gain = 0
        
        for idx in range(64):
            val = board.state[idx]
            if not (val & my_flag_q3): continue
            h = val & HEIGHT_MASK
            if h < 2: continue
            
            for offset in OFFSETS:
                nb = cascade(board, idx, offset)
                nb_opp = nb.blue_tokens if is_red_maximizing else nb.red_tokens
                nb_my = nb.red_tokens if is_red_maximizing else nb.blue_tokens
                
                opp_destroyed = opp_tokens_q3 - nb_opp
                my_lost = my_tokens_q3 - nb_my
                net = opp_destroyed - my_lost
                
                if nb_opp == 0 and nb_my > 0:
                    # We can cascade to win! This is near +INF
                    return INF - 100
                
                if net > best_net_gain:
                    best_net_gain = net
                    best_our_cascade = nb
        
        if best_our_cascade is not None and best_net_gain >= 3:
            post_score = _static_eval(best_our_cascade, is_red_maximizing)
            static_score = _static_eval(board, is_red_maximizing)
            # Blend: we'll probably take this cascade
            return post_score * 0.6 + static_score * 0.4

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
        elif my_max_h == opp_max_h:
            # EQUAL HEIGHT: approach smaller enemy stacks we can eat.
            # This prevents the B6-vs-R6 stalemate where the agent
            # shuffles endlessly because neither "approach" (needs
            # strictly taller) nor "flee" (needs strictly shorter) fires.
            #
            # B6 should eat R1/R2 debris to become B7/B8 and gain
            # height advantage. Convergence rewards closing distance
            # to any enemy shorter than us.
            min_eat_dist = 99
            for mi, mh in my_stacks:
                for oi, oh in opp_stacks:
                    if mh > oh:
                        d = MANHATTAN[mi][oi]
                        if d < min_eat_dist:
                            min_eat_dist = d
            if min_eat_dist < 99:
                convergence = max(0, 8 - min_eat_dist) * 2.0
            # Also: stay centered (equal height = mutual danger)
            for mi, mh in my_stacks:
                convergence += EDGE_DIST[mi] * 0.5

        elif opp_max_h > my_max_h:
            # FLEE MODE: reward distance AND centrality.
            for mi, mh in my_stacks:
                min_threat_dist = 99
                for oi, oh in opp_stacks:
                    if oh > mh:
                        d = MANHATTAN[mi][oi]
                        if d < min_threat_dist:
                            min_threat_dist = d
                if min_threat_dist < 99:
                    ed = EDGE_DIST[mi]
                    dist_score = min(min_threat_dist, 7) * 1.5
                    center_score = ed * 3.0
                    convergence += dist_score + center_score

    # ── 12. STRUCTURE ────────────────────────────────────
    elim = 0.0
    if opp_stack_n <= 1: elim += 5.0
    elif opp_stack_n == 2: elim += 2.0
    if my_stack_n <= 1: elim -= 5.0
    elif my_stack_n == 2: elim -= 2.0

    # ── 13. PINCER TRAP ───────────────────────────────────
    # Two of our stacks on the same row/col with an enemy BETWEEN them.
    # The enemy is trapped: moving along the line toward either of our
    # stacks gets it eaten/cascaded. Moving perpendicular only works
    # if it doesn't hit an edge.
    #
    # Example: B3(3,3), B3(5,3), R3(4,3). R3 is pinched:
    #   Move up → B3(3,3) eats. Move down → B3(5,3) eats.
    #   Move left/right → might hit edge, or we follow.
    #
    # Score: based on how trapped the enemy is.
    #   - Both our stacks >= enemy height: deadly (we eat either way)
    #   - Enemy near edge perpendicular to the line: even worse
    #   - Close pincer (our stacks close together): more urgent
    pincer_bonus = 0.0
    pincer_penalty = 0.0

    # Check our pincers on enemy
    for i, (mi1, mh1) in enumerate(my_stacks):
        for mi2, mh2 in my_stacks[i+1:]:
            mr1, mc1 = mi1 >> 3, mi1 & 7
            mr2, mc2 = mi2 >> 3, mi2 & 7

            # Same row?
            if mr1 == mr2:
                lo_c = min(mc1, mc2)
                hi_c = max(mc1, mc2)
                for oi, oh in opp_stacks:
                    or_, oc = oi >> 3, oi & 7
                    if or_ == mr1 and lo_c < oc < hi_c:
                        # Enemy is between us on same row!
                        gap = hi_c - lo_c - 1  # cells between our stacks
                        can_eat = (mh1 >= oh) and (mh2 >= oh)
                        # Perpendicular escape: how close to edge?
                        perp_escape = min(or_, 7 - or_)  # 0=edge, 3=center

                        score_p = oh * 2.0  # base: enemy height at risk
                        if can_eat:
                            score_p += oh * 3.0  # deadly: eaten either way
                        if gap <= 2:
                            score_p += oh * 2.0  # tight pincer
                        if perp_escape <= 1:
                            score_p += oh * 2.0  # edge perpendicular = very trapped
                        pincer_bonus += score_p

            # Same column?
            if mc1 == mc2:
                lo_r = min(mr1, mr2)
                hi_r = max(mr1, mr2)
                for oi, oh in opp_stacks:
                    or_, oc = oi >> 3, oi & 7
                    if oc == mc1 and lo_r < or_ < hi_r:
                        gap = hi_r - lo_r - 1
                        can_eat = (mh1 >= oh) and (mh2 >= oh)
                        perp_escape = min(oc, 7 - oc)

                        score_p = oh * 2.0
                        if can_eat:
                            score_p += oh * 3.0
                        if gap <= 2:
                            score_p += oh * 2.0
                        if perp_escape <= 1:
                            score_p += oh * 2.0
                        pincer_bonus += score_p

    # Check enemy pincers on us
    for i, (oi1, oh1) in enumerate(opp_stacks):
        for oi2, oh2 in opp_stacks[i+1:]:
            or1, oc1 = oi1 >> 3, oi1 & 7
            or2, oc2 = oi2 >> 3, oi2 & 7

            if or1 == or2:
                lo_c = min(oc1, oc2)
                hi_c = max(oc1, oc2)
                for mi, mh in my_stacks:
                    mr, mc = mi >> 3, mi & 7
                    if mr == or1 and lo_c < mc < hi_c:
                        gap = hi_c - lo_c - 1
                        can_eat = (oh1 >= mh) and (oh2 >= mh)
                        perp_escape = min(mr, 7 - mr)

                        score_p = mh * 2.0
                        if can_eat:
                            score_p += mh * 3.0
                        if gap <= 2:
                            score_p += mh * 2.0
                        if perp_escape <= 1:
                            score_p += mh * 2.0
                        pincer_penalty += score_p

            if oc1 == oc2:
                lo_r = min(or1, or2)
                hi_r = max(or1, or2)
                for mi, mh in my_stacks:
                    mr, mc = mi >> 3, mi & 7
                    if mc == oc1 and lo_r < mr < hi_r:
                        gap = hi_r - lo_r - 1
                        can_eat = (oh1 >= mh) and (oh2 >= mh)
                        perp_escape = min(mc, 7 - mc)

                        score_p = mh * 2.0
                        if can_eat:
                            score_p += mh * 3.0
                        if gap <= 2:
                            score_p += mh * 2.0
                        if perp_escape <= 1:
                            score_p += mh * 2.0
                        pincer_penalty += score_p

    # ── 14. CORNER TRAP ──────────────────────────────────
    # Enemy in corner (or 1 step from corner) with our stack positioned
    # to cascade it off. This is an instant-win setup:
    #   R6 at (0,0), B4 at (1,1): no matter what R6 does, B4 cascades.
    #   R6 moves to (0,1) → B4 cascade left pushes R6 off.
    #   R6 moves to (1,0) → B4 cascade up pushes R6 off.
    #
    # Score: massive when we can cascade in MULTIPLE directions to
    # cover all escape routes. The more exits we cover, the more trapped.
    CORNER_CELLS = [(0,0), (0,7), (7,0), (7,7)]

    corner_trap_bonus = 0.0
    corner_trap_penalty = 0.0

    for oi, oh in opp_stacks:
        or_, oc = oi >> 3, oi & 7
        opp_ed = EDGE_DIST[oi]
        if opp_ed > 1: continue  # only corner / near-corner

        # Count how many cascade-attack angles we have on this enemy
        # A cascade attack angle: our stack on the same row/col,
        # with enough height to reach the enemy's position and push it off
        attack_angles = 0
        total_attack_score = 0.0

        for mi, mh in my_stacks:
            mr, mc = mi >> 3, mi & 7

            # Same row?
            if mr == or_:
                dist = abs(mc - oc)
                if dist >= 1 and mh >= dist:
                    # We can cascade along this row and reach the enemy
                    # Check: does the cascade push enemy off the edge?
                    # Enemy at column oc, we're at mc.
                    # If mc < oc: cascade right → pushes enemy right → off if oc + (mh-dist) >= 8
                    # If mc > oc: cascade left → pushes enemy left → off if oc - (mh-dist) < 0
                    if mc < oc:
                        push_dist = mh - dist  # remaining tokens after reaching enemy
                        if oc + push_dist >= 8:
                            attack_angles += 1
                            total_attack_score += oh * 3.0
                    else:
                        push_dist = mh - dist
                        if oc - push_dist < 0:
                            attack_angles += 1
                            total_attack_score += oh * 3.0

            # Same column?
            if mc == oc:
                dist = abs(mr - or_)
                if dist >= 1 and mh >= dist:
                    if mr < or_:
                        push_dist = mh - dist
                        if or_ + push_dist >= 8:
                            attack_angles += 1
                            total_attack_score += oh * 3.0
                    else:
                        push_dist = mh - dist
                        if or_ - push_dist < 0:
                            attack_angles += 1
                            total_attack_score += oh * 3.0

            # DIAGONAL CAGE: diagonally adjacent to a corner enemy.
            # B4(1,1) vs R6(0,0): R6 can only go to (0,1) or (1,0).
            # After R6→(0,1): B4 is on same column → cascade up pushes off.
            # After R6→(1,0): B4 is on same row → cascade left pushes off.
            # So diagonal adjacency to a corner = 2 attack angles!
            if opp_ed == 0 and MANHATTAN[mi][oi] == 2:
                # Check if we're diagonally adjacent (dr=1, dc=1)
                dr = abs(mr - or_)
                dc = abs(mc - oc)
                if dr == 1 and dc == 1 and mh >= 1:
                    # Count both escape routes we cover
                    # Escape 1: enemy moves to (or_, mc) — same col as us
                    esc1_r, esc1_c = or_, mc
                    if 0 <= esc1_r < 8 and 0 <= esc1_c < 8:
                        esc1_ed_r = min(esc1_r, 7 - esc1_r)
                        esc1_ed_c = min(esc1_c, 7 - esc1_c)
                        if esc1_ed_r == 0 or esc1_ed_c == 0:
                            # We can cascade to push enemy off from this position
                            attack_angles += 1
                            total_attack_score += oh * 3.0
                    # Escape 2: enemy moves to (mr, oc) — same row as us
                    esc2_r, esc2_c = mr, oc
                    if 0 <= esc2_r < 8 and 0 <= esc2_c < 8:
                        esc2_ed_r = min(esc2_r, 7 - esc2_r)
                        esc2_ed_c = min(esc2_c, 7 - esc2_c)
                        if esc2_ed_r == 0 or esc2_ed_c == 0:
                            attack_angles += 1
                            total_attack_score += oh * 3.0

        if attack_angles >= 2:
            # Multiple attack angles = enemy can't escape either direction
            corner_trap_bonus += total_attack_score * 2.0
        elif attack_angles == 1:
            corner_trap_bonus += total_attack_score * 0.8

    # Check if WE are corner-trapped
    for mi, mh in my_stacks:
        mr, mc = mi >> 3, mi & 7
        my_ed = EDGE_DIST[mi]
        if my_ed > 1: continue

        attack_angles = 0
        total_attack_score = 0.0

        for oi, oh in opp_stacks:
            or_, oc = oi >> 3, oi & 7

            if or_ == mr:
                dist = abs(oc - mc)
                if dist >= 1 and oh >= dist:
                    if oc < mc:
                        push_dist = oh - dist
                        if mc + push_dist >= 8:
                            attack_angles += 1
                            total_attack_score += mh * 3.0
                    else:
                        push_dist = oh - dist
                        if mc - push_dist < 0:
                            attack_angles += 1
                            total_attack_score += mh * 3.0

            if oc == mc:
                dist = abs(or_ - mr)
                if dist >= 1 and oh >= dist:
                    if or_ < mr:
                        push_dist = oh - dist
                        if mr + push_dist >= 8:
                            attack_angles += 1
                            total_attack_score += mh * 3.0
                    else:
                        push_dist = oh - dist
                        if mr - push_dist < 0:
                            attack_angles += 1
                            total_attack_score += mh * 3.0

            # Diagonal cage (enemy diagonally adjacent to us in corner)
            if my_ed == 0 and MANHATTAN[oi][mi] == 2:
                dr = abs(or_ - mr)
                dc = abs(oc - mc)
                if dr == 1 and dc == 1 and oh >= 1:
                    esc1_r, esc1_c = mr, oc
                    if 0 <= esc1_r < 8 and 0 <= esc1_c < 8:
                        if min(esc1_r, 7-esc1_r) == 0 or min(esc1_c, 7-esc1_c) == 0:
                            attack_angles += 1
                            total_attack_score += mh * 3.0
                    esc2_r, esc2_c = or_, mc
                    if 0 <= esc2_r < 8 and 0 <= esc2_c < 8:
                        if min(esc2_r, 7-esc2_r) == 0 or min(esc2_c, 7-esc2_c) == 0:
                            attack_angles += 1
                            total_attack_score += mh * 3.0

        if attack_angles >= 2:
            corner_trap_penalty += total_attack_score * 2.0
        elif attack_angles == 1:
            corner_trap_penalty += total_attack_score * 0.8

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

        # Pincer trap
        + pincer_bonus * elimination_weight
        - pincer_penalty * survival_weight

        # Corner trap
        + corner_trap_bonus * elimination_weight
        - corner_trap_penalty * survival_weight
    )
    return score