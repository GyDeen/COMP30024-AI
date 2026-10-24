"""
Evaluation function for Cascade — v12: central anchor strategy.

CHANGES OVER v11:
  * ANCHOR BONUS: large reward for owning a height-4+ stack on one of
    the 4 center squares (3,3), (3,4), (4,3), (4,4). This is the
    strategic core — a tall central stack threatens the entire ring
    of cells around it and is hardest to corner or push off.
  * CENTRIPETAL PULL: when no anchor exists, h>=2 stacks get a bonus
    proportional to their proximity to a center square, weighted by
    height. Pulls our pieces inward to build the anchor.
  * HUNTER-CONVERGENCE: the existing convergence term used to reward
    only the tallest stack closing distance to enemies. Now, when we
    have an anchor, the anchor is EXCLUDED from convergence (it should
    stay put) and non-anchor "hunters" do the closing. This stops MCTS
    from wanting to drag the anchor across the board.
  * Quiescence (eat/cascade lookahead) and all v11 terms preserved.
"""
from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import is_valid_coord, UP, DOWN, LEFT, RIGHT, MOVE, EAT, CASCADE
from Agents.common.action import move, eat, cascade

INF = 999999
OFFSETS = (UP, DOWN, LEFT, RIGHT)

# 4 center cells: (3,3)=27, (3,4)=28, (4,3)=35, (4,4)=36
CENTER_4 = frozenset({27, 28, 35, 36})
ANCHOR_MIN_H = 4

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

# Distance from each cell to its nearest CENTER_4 cell
DIST_TO_CENTER = [0] * 64
for _i in range(64):
    DIST_TO_CENTER[_i] = min(MANHATTAN[_i][_c] for _c in CENTER_4)


def _quiescence_eval(board, is_red_maximizing):
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
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    state = board.state
    
    my_tokens = board.red_tokens if is_red_maximizing else board.blue_tokens
    opp_tokens = board.blue_tokens if is_red_maximizing else board.red_tokens
    
    worst_board = None
    worst_loss = 0
    
    for idx in range(64):
        val = state[idx]
        if not (val & opp_flag): continue
        h = val & HEIGHT_MASK
        if h < 2: continue
        
        for offset in OFFSETS:
            nb = cascade(board, idx, offset)
            
            nb_my = nb.red_tokens if is_red_maximizing else nb.blue_tokens
            my_lost = my_tokens - nb_my
            
            nb_opp = nb.blue_tokens if is_red_maximizing else nb.red_tokens
            opp_lost = opp_tokens - nb_opp
            
            net_damage = my_lost - opp_lost
            
            if net_damage > worst_loss:
                worst_loss = net_damage
                worst_board = nb
            
            if nb_my == 0:
                return nb, my_tokens
    
    return worst_board, worst_loss


def evaluate(board: BoardState, is_red_maximizing: bool) -> float:
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens

    if red_tokens == 0: return -INF if is_red_maximizing else INF
    if blue_tokens == 0: return INF if is_red_maximizing else -INF

    # Quiescence: if we have a safe eat, score the post-eat position
    q_board = _quiescence_eval(board, is_red_maximizing)
    if q_board is not None:
        return _static_eval(q_board, is_red_maximizing) + 3.0

    # Quiescence: opponent cascade threat
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
            
            cb_my = cascade_board.red_tokens if is_red_maximizing else cascade_board.blue_tokens
            if cb_my == 0:
                return -INF + 100
            
            post_cascade_score = _static_eval(cascade_board, is_red_maximizing)
            
            if cascade_loss >= 3:
                return post_cascade_score * 0.7 + static_score * 0.3
            elif cascade_loss >= 1:
                return post_cascade_score * 0.4 + static_score * 0.6

    # Quiescence: our profitable cascade
    my_flag_q3 = RED_FLAG if is_red_maximizing else BLUE_FLAG
    any_we_can_cascade = False
    for idx in range(64):
        v = board.state[idx]
        if v and (v & my_flag_q3) and (v & HEIGHT_MASK) >= 2:
            any_we_can_cascade = True
            break

    if any_we_can_cascade:
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
                    return INF - 100
                
                if net > best_net_gain:
                    best_net_gain = net
                    best_our_cascade = nb
        
        if best_our_cascade is not None and best_net_gain >= 3:
            post_score = _static_eval(best_our_cascade, is_red_maximizing)
            static_score = _static_eval(board, is_red_maximizing)
            return post_score * 0.6 + static_score * 0.4

    return _static_eval(board, is_red_maximizing)


def _static_eval(board: BoardState, is_red_maximizing: bool) -> float:
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

    # Game phase weights
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

    my_max_h = max(h for _, h in my_stacks)
    opp_max_h = max(h for _, h in opp_stacks)
    we_are_tallest = my_max_h > opp_max_h

    # ── ANCHOR BONUS ─────────────────────────────────────
    # Strategic core: tall stack on one of the 4 center cells.
    # Threatens the surrounding 8-cell ring, hardest to corner.
    # Reward escalates with height up to h6 (diminishing returns).
    anchor_bonus = 0.0
    my_has_anchor = False
    opp_has_anchor = False

    for mi, mh in my_stacks:
        if mi in CENTER_4:
            if mh >= ANCHOR_MIN_H:
                anchor_bonus += 22.0 + min(mh - 4, 2) * 3.0  # h4=22, h5=25, h6+=28
                my_has_anchor = True
            elif mh == 3:
                anchor_bonus += 10.0  # almost an anchor
            elif mh == 2:
                anchor_bonus += 4.0
            else:
                anchor_bonus += 1.5

    for oi, oh in opp_stacks:
        if oi in CENTER_4:
            if oh >= ANCHOR_MIN_H:
                anchor_bonus -= 22.0 + min(oh - 4, 2) * 3.0
                opp_has_anchor = True
            elif oh == 3:
                anchor_bonus -= 10.0
            elif oh == 2:
                anchor_bonus -= 4.0
            else:
                anchor_bonus -= 1.5

    # ── CENTRIPETAL PULL ─────────────────────────────────
    # When we don't have an anchor yet, pull h>=2 stacks toward center.
    # max(0, 5 - dist_to_center) means cells at dist 0–4 contribute.
    # Weighted by height so taller stacks getting central matters more.
    centripetal = 0.0
    if not my_has_anchor:
        for mi, mh in my_stacks:
            if mh >= 2:
                centripetal += mh * max(0, 5 - DIST_TO_CENTER[mi]) * 0.7
    if not opp_has_anchor:
        for oi, oh in opp_stacks:
            if oh >= 2:
                centripetal -= oh * max(0, 5 - DIST_TO_CENTER[oi]) * 0.7

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

    # ── 3. IMMEDIATE THREATS ─────────────────────────────
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
                    opp_far = FAR_EDGE_DIST[pos * 4 + d_idx]
                    if opp_h >= opp_far:
                        urgency = max(0, 6 - dist_from_me)
                        if dist_from_me == 1 and mh > opp_h:
                            edge_trap_penalty += mh * urgency * 0.3
                        else:
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
    edge_death_penalty = 0.0
    edge_death_bonus = 0.0

    for mi, mh in my_stacks:
        ed = EDGE_DIST[mi]
        if ed >= 3: continue

        min_enemy_dist = 99
        nearest_enemy_h = 0
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d < min_enemy_dist:
                min_enemy_dist = d
                nearest_enemy_h = oh

        if nearest_enemy_h > mh:
            edge_danger = (3 - ed) * 4.0
            if min_enemy_dist <= 2:
                prox_factor = 3.0
            elif min_enemy_dist <= 4:
                prox_factor = 2.0
            elif min_enemy_dist <= 6:
                prox_factor = 1.0
            else:
                prox_factor = 0.5
            edge_death_penalty += mh * edge_danger * prox_factor
        elif nearest_enemy_h == mh:
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

    # ── 11. CONVERGENCE (anchor-aware) ───────────────────
    # When we have an anchor, EXCLUDE it from convergence — anchors
    # stay put. Only non-anchor "hunters" should be rewarded for
    # closing distance to enemies.
    convergence = 0.0
    if my_stacks and opp_stacks:
        if we_are_tallest:
            min_kill_dist = 99
            for mi, mh in my_stacks:
                # Skip the anchor: it threatens statically, not by chasing
                if my_has_anchor and mi in CENTER_4 and mh >= ANCHOR_MIN_H:
                    continue
                # Original heuristic: tallest-height stack only.
                # Without an anchor that's our max stack; with an anchor
                # we need any stack that can kill an enemy.
                if not my_has_anchor and mh != my_max_h:
                    continue
                for oi, oh in opp_stacks:
                    if mh > oh:
                        d = MANHATTAN[mi][oi]
                        if d < min_kill_dist:
                            min_kill_dist = d
            if min_kill_dist < 99:
                convergence = max(0, 8 - min_kill_dist) * 3.0

        elif my_max_h == opp_max_h:
            min_eat_dist = 99
            for mi, mh in my_stacks:
                if my_has_anchor and mi in CENTER_4 and mh >= ANCHOR_MIN_H:
                    continue
                for oi, oh in opp_stacks:
                    if mh > oh:
                        d = MANHATTAN[mi][oi]
                        if d < min_eat_dist:
                            min_eat_dist = d
            if min_eat_dist < 99:
                convergence = max(0, 8 - min_eat_dist) * 2.0
            for mi, mh in my_stacks:
                convergence += EDGE_DIST[mi] * 0.5

        elif opp_max_h > my_max_h:
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
    pincer_bonus = 0.0
    pincer_penalty = 0.0

    for i, (mi1, mh1) in enumerate(my_stacks):
        for mi2, mh2 in my_stacks[i+1:]:
            mr1, mc1 = mi1 >> 3, mi1 & 7
            mr2, mc2 = mi2 >> 3, mi2 & 7

            if mr1 == mr2:
                lo_c = min(mc1, mc2)
                hi_c = max(mc1, mc2)
                for oi, oh in opp_stacks:
                    or_, oc = oi >> 3, oi & 7
                    if or_ == mr1 and lo_c < oc < hi_c:
                        gap = hi_c - lo_c - 1
                        can_eat = (mh1 >= oh) and (mh2 >= oh)
                        perp_escape = min(or_, 7 - or_)

                        score_p = oh * 2.0
                        if can_eat:
                            score_p += oh * 3.0
                        if gap <= 2:
                            score_p += oh * 2.0
                        if perp_escape <= 1:
                            score_p += oh * 2.0
                        pincer_bonus += score_p

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
    corner_trap_bonus = 0.0
    corner_trap_penalty = 0.0

    for oi, oh in opp_stacks:
        or_, oc = oi >> 3, oi & 7
        opp_ed = EDGE_DIST[oi]
        if opp_ed > 1: continue

        attack_angles = 0
        total_attack_score = 0.0

        for mi, mh in my_stacks:
            mr, mc = mi >> 3, mi & 7

            if mr == or_:
                dist = abs(mc - oc)
                if dist >= 1 and mh >= dist:
                    if mc < oc:
                        push_dist = mh - dist
                        if oc + push_dist >= 8:
                            attack_angles += 1
                            total_attack_score += oh * 3.0
                    else:
                        push_dist = mh - dist
                        if oc - push_dist < 0:
                            attack_angles += 1
                            total_attack_score += oh * 3.0

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

            if opp_ed == 0 and MANHATTAN[mi][oi] == 2:
                dr = abs(mr - or_)
                dc = abs(mc - oc)
                if dr == 1 and dc == 1 and mh >= 1:
                    esc1_r, esc1_c = or_, mc
                    if 0 <= esc1_r < 8 and 0 <= esc1_c < 8:
                        esc1_ed_r = min(esc1_r, 7 - esc1_r)
                        esc1_ed_c = min(esc1_c, 7 - esc1_c)
                        if esc1_ed_r == 0 or esc1_ed_c == 0:
                            attack_angles += 1
                            total_attack_score += oh * 3.0
                    esc2_r, esc2_c = mr, oc
                    if 0 <= esc2_r < 8 and 0 <= esc2_c < 8:
                        esc2_ed_r = min(esc2_r, 7 - esc2_r)
                        esc2_ed_c = min(esc2_c, 7 - esc2_c)
                        if esc2_ed_r == 0 or esc2_ed_c == 0:
                            attack_angles += 1
                            total_attack_score += oh * 3.0

        if attack_angles >= 2:
            corner_trap_bonus += total_attack_score * 2.0
        elif attack_angles == 1:
            corner_trap_bonus += total_attack_score * 0.8

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

        + pincer_bonus * elimination_weight
        - pincer_penalty * survival_weight

        + corner_trap_bonus * elimination_weight
        - corner_trap_penalty * survival_weight

        # ★ NEW: central anchor strategy
        + anchor_bonus
        + centripetal
    )
    return score