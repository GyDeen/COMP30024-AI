"""
Heuristic evaluation for Cascade.

Movement phase:
    - quiescence safe-eat extension
    - opponent cascade-threat detection
    - anti-retreat / convergence behaviour
    - edge and corner death penalties
    - mobility, initiative, capped height dominance, centre-cascade strategy

Placement phase:
    - latest L-shape + blocker logic
    - prefers inner 4x4 zone
    - first token prefers centre
    - builds compact L-shape
    - uses extra stack as blocker toward enemy expansion
"""

from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from .action import is_valid_coord, UP, DOWN, LEFT, RIGHT
from .action import eat, cascade


INF = 999999
OFFSETS = (UP, DOWN, LEFT, RIGHT)


# =============================================================================
# Shared lookup tables
# =============================================================================

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


MANHATTAN = [[0] * 64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        MANHATTAN[_a][_b] = abs((_a >> 3) - (_b >> 3)) + abs((_a & 7) - (_b & 7))


# Centre/corner strategic zones for movement phase.
# 27=(3,3), 28=(3,4), 35=(4,3), 36=(4,4).
MOVEMENT_CENTER_4 = frozenset({27, 28, 35, 36})
MOVEMENT_CENTER_16 = frozenset({
    18, 19, 20, 21,
    26, 27, 28, 29,
    34, 35, 36, 37,
    42, 43, 44, 45,
})

CORNERS = frozenset({0, 7, 56, 63})
CORNER_RING = frozenset({
    0, 1, 8, 9,
    6, 7, 14, 15,
    48, 49, 56, 57,
    54, 55, 62, 63,
})


def _dist_to_center(idx):
    return min(
        MANHATTAN[idx][27], MANHATTAN[idx][28],
        MANHATTAN[idx][35], MANHATTAN[idx][36],
    )


def _closest_dist(idx, stacks):
    if not stacks:
        return 99
    return min(MANHATTAN[idx][j] for j, _ in stacks)


def _path_after_cascade(start, offset, height):
    cells = []
    pos = start
    for _ in range(height):
        if not is_valid_coord(pos, offset):
            break
        pos += offset
        cells.append(pos)
    return cells


def _cascade_is_contact_escape(board, opp_source, offset, is_red_maximizing):
    """
    Only consider an opponent cascade as an urgent escape-stage threat when it
    is already local to our stack. This prevents the evaluator from panicking
    about long-distance theoretical cascades.
    """
    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    state = board.state

    for nb in ADJACENTS[opp_source]:
        if state[nb] and (state[nb] & my_flag):
            return True

    val = state[opp_source]
    h = val & HEIGHT_MASK
    touched = _path_after_cascade(opp_source, offset, h)

    for cell in touched[:2]:
        for nb in ADJACENTS[cell]:
            if state[nb] and (state[nb] & my_flag):
                return True

    return False


def _capped_height_value(h):
    """Reward concentration up to h4, but punish giant h6+ stacks."""
    if h <= 4:
        return h * h
    if h == 5:
        return 14
    return 14 - (h - 5) * 8


# =============================================================================
# Movement phase heuristic
# =============================================================================

def _quiescence_eval(board, is_red_maximizing):
    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    state = board.state
    best_eat_board = None
    best_eat_gain = 0
    for idx in range(64):
        val = state[idx]
        if not (val & my_flag):
            continue
        h = val & HEIGHT_MASK
        for offset in OFFSETS:
            if not is_valid_coord(idx, offset):
                continue
            target_idx = idx + offset
            tv = state[target_idx]
            if not (tv & opp_flag):
                continue
            th = tv & HEIGHT_MASK
            if h < th:
                continue
            nb = eat(board, idx, offset)
            safe = True
            for check_off in OFFSETS:
                if not is_valid_coord(target_idx, check_off):
                    continue
                cv = nb.state[target_idx + check_off]
                if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= h:
                    safe = False
                    break
            if safe and th > best_eat_gain:
                best_eat_gain = th
                best_eat_board = nb
    return best_eat_board


def _opponent_cascade_threat(board, is_red_maximizing):
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    state = board.state
    my_tokens = board.red_tokens if is_red_maximizing else board.blue_tokens
    opp_tokens = board.blue_tokens if is_red_maximizing else board.red_tokens
    worst_board = None
    worst_loss = 0
    for idx in range(64):
        val = state[idx]
        if not (val & opp_flag):
            continue
        h = val & HEIGHT_MASK
        if h < 2:
            continue
        for offset in OFFSETS:
            # Escape-stage fix: only react to adjacent/local cascade danger.
            if not _cascade_is_contact_escape(board, idx, offset, is_red_maximizing):
                continue

            nb = cascade(board, idx, offset)
            nb_my = nb.red_tokens if is_red_maximizing else nb.blue_tokens
            nb_opp = nb.blue_tokens if is_red_maximizing else nb.red_tokens
            my_lost = my_tokens - nb_my
            opp_lost = opp_tokens - nb_opp
            net_damage = my_lost - opp_lost
            if net_damage > worst_loss:
                worst_loss = net_damage
                worst_board = nb
            if nb_my == 0:
                return nb, my_tokens
    return worst_board, worst_loss


def evaluate(board, is_red_maximizing):
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens
    if red_tokens == 0:
        return -INF if is_red_maximizing else INF
    if blue_tokens == 0:
        return INF if is_red_maximizing else -INF

    q_board = _quiescence_eval(board, is_red_maximizing)
    if q_board is not None:
        return _static_eval(q_board, is_red_maximizing) + 3.0

    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    any_near_edge = False
    for idx in range(64):
        val = board.state[idx]
        if val and (val & my_flag) and EDGE_DIST[idx] <= 2:
            any_near_edge = True
            break

    if any_near_edge:
        cascade_board, cascade_loss = _opponent_cascade_threat(board, is_red_maximizing)
        if cascade_loss > 0 and cascade_board is not None:
            static_score = _static_eval(board, is_red_maximizing)
            cb_my = cascade_board.red_tokens if is_red_maximizing else cascade_board.blue_tokens
            if cb_my == 0:
                return -INF + 100
            post_cascade_score = _static_eval(cascade_board, is_red_maximizing)
            if cascade_loss >= 3:
                return post_cascade_score * 0.7 + static_score * 0.3
            if cascade_loss >= 1:
                return post_cascade_score * 0.4 + static_score * 0.6

    return _static_eval(board, is_red_maximizing)


def _static_eval(board, is_red_maximizing):
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens
    if red_tokens == 0:
        return -INF if is_red_maximizing else INF
    if blue_tokens == 0:
        return INF if is_red_maximizing else -INF

    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    state = board.state
    my_tokens = red_tokens if is_red_maximizing else blue_tokens
    opp_tokens = blue_tokens if is_red_maximizing else red_tokens

    my_stacks = []
    opp_stacks = []
    for idx in range(64):
        val = state[idx]
        if val == 0:
            continue
        h = val & HEIGHT_MASK
        if val & my_flag:
            my_stacks.append((idx, h))
        else:
            opp_stacks.append((idx, h))

    my_stack_n = len(my_stacks)
    opp_stack_n = len(opp_stacks)
    if not my_stacks:
        return -INF
    if not opp_stacks:
        return INF

    total = my_tokens + opp_tokens
    if total == 0:
        return 0.0

    advantage = (my_tokens - opp_tokens) / total
    late_factor = max(0.0, 1.0 - total / 16.0)

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

    # h4 is the ideal weapon. h5 is only acceptable when we are clearly
    # winning and the opponent is nearly finished. h6+ is overstacking.
    final_stage = late_factor > 0.5 and advantage > 0.15 and opp_stack_n <= 1
    max_height_penalty = 0.0
    opp_overstack_bonus = 0.0

    for _, mh in my_stacks:
        if mh <= 4:
            continue
        if mh == 5:
            if not final_stage:
                max_height_penalty += 25.0
        else:
            max_height_penalty += 45.0 + (mh - 5) * 35.0

    for _, oh in opp_stacks:
        if oh <= 4:
            continue
        if oh == 5:
            if not final_stage:
                opp_overstack_bonus += 18.0
        else:
            opp_overstack_bonus += 35.0 + (oh - 5) * 25.0

    material_diff = my_tokens - opp_tokens
    material_ratio = material_diff / total

    proximity = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d > 8:
                continue
            if mh > oh:
                surplus = mh - oh
                dist_bonus = max(0, 8 - d)
                vuln = 1.0
                if oh <= 2:
                    vuln = 2.0
                    isolated = True
                    for oi2, _ in opp_stacks:
                        if oi2 != oi and MANHATTAN[oi][oi2] <= 2:
                            isolated = False
                            break
                    if isolated:
                        vuln = 3.0
                proximity += (1 + surplus) * dist_bonus * vuln * 0.25
            elif mh == oh and advantage > 0.1:
                dist_bonus = max(0, 8 - d)
                proximity += dist_bonus * 0.08

    my_threatened = 0.0
    opp_threatened = 0.0
    for mi, mh in my_stacks:
        worst_threat = 0
        for nb in ADJACENTS[mi]:
            nb_val = state[nb]
            if nb_val and (nb_val & opp_flag):
                nb_h = nb_val & HEIGHT_MASK
                if nb_h > mh:
                    worst_threat = 2
                    break
                if nb_h >= mh and worst_threat < 1:
                    worst_threat = 1
        if worst_threat == 2:
            my_threatened += mh
        elif worst_threat == 1:
            if my_stack_n <= 1:
                my_threatened += mh * 1.5
            elif my_stack_n == 2:
                my_threatened += mh * 0.8
            else:
                my_threatened += mh * 0.3

    for oi, oh in opp_stacks:
        worst_threat = 0
        for nb in ADJACENTS[oi]:
            nb_val = state[nb]
            if nb_val and (nb_val & my_flag):
                nb_h = nb_val & HEIGHT_MASK
                if nb_h > oh:
                    worst_threat = 2
                    break
                if nb_h >= oh and worst_threat < 1:
                    worst_threat = 1
        if worst_threat == 2:
            opp_threatened += oh
        elif worst_threat == 1:
            if opp_stack_n <= 1:
                opp_threatened += oh * 1.5
            elif opp_stack_n == 2:
                opp_threatened += oh * 0.8
            else:
                opp_threatened += oh * 0.3

    my_threats = 0.0
    opp_threats = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d > 3:
                continue
            if mh > oh:
                if d <= 1:
                    my_threats += oh * 3.0
                elif d == 2:
                    my_threats += oh * 1.5
                elif d == 3:
                    my_threats += oh * 0.5
    for oi, oh in opp_stacks:
        for mi, mh in my_stacks:
            d = MANHATTAN[oi][mi]
            if d > 3:
                continue
            if oh > mh:
                if d <= 1:
                    opp_threats += mh * 3.0
                elif d == 2:
                    opp_threats += mh * 1.5
                elif d == 3:
                    opp_threats += mh * 0.5
    initiative = my_threats - opp_threats

    my_debris_penalty = 0.0
    opp_debris_penalty = 0.0
    for mi, mh in my_stacks:
        if mh > 1:
            continue
        closest = 99
        for oi, _ in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d < closest:
                closest = d
        if closest <= 1:
            my_debris_penalty += 3.0
        elif closest <= 2:
            my_debris_penalty += 2.0
        elif closest <= 3:
            my_debris_penalty += 1.0
    for oi, oh in opp_stacks:
        if oh > 1:
            continue
        closest = 99
        for mi, _ in my_stacks:
            d = MANHATTAN[oi][mi]
            if d < closest:
                closest = d
        if closest <= 1:
            opp_debris_penalty += 3.0
        elif closest <= 2:
            opp_debris_penalty += 2.0
        elif closest <= 3:
            opp_debris_penalty += 1.0

    my_dominant = sum(1 for _, h in my_stacks if h > opp_max_h)
    opp_dominant = sum(1 for _, h in opp_stacks if h > my_max_h)
    dominance = (my_dominant - opp_dominant) * 3.0
    height_adv = max(0, my_max_h - opp_max_h) * 5.0
    height_disadv = max(0, opp_max_h - my_max_h) * 5.0
    my_height_sq = sum(_capped_height_value(h) for _, h in my_stacks)
    opp_height_sq = sum(_capped_height_value(h) for _, h in opp_stacks)
    height_preservation = (my_height_sq - opp_height_sq) * 0.5

    edge_trap_bonus = 0.0
    edge_trap_penalty = 0.0
    for oi, oh in opp_stacks:
        if EDGE_DIST[oi] > 2:
            continue
        for d_idx in range(4):
            opp_far = FAR_EDGE_DIST[oi * 4 + d_idx]
            if opp_far > 2:
                continue
            scan_off = SCAN_OFFSETS[d_idx]
            pos = oi
            dist_from_opp = 0
            while is_valid_coord(pos, scan_off):
                pos += scan_off
                dist_from_opp += 1
                sv = state[pos]
                if sv == 0:
                    continue
                if sv & my_flag:
                    our_far = FAR_EDGE_DIST[pos * 4 + d_idx]
                    if (sv & HEIGHT_MASK) >= our_far:
                        urgency = max(0, 6 - dist_from_opp)
                        edge_trap_bonus += oh * urgency * 1.0
                break
    for mi, mh in my_stacks:
        if EDGE_DIST[mi] > 2:
            continue
        for d_idx in range(4):
            my_far = FAR_EDGE_DIST[mi * 4 + d_idx]
            if my_far > 2:
                continue
            scan_off = SCAN_OFFSETS[d_idx]
            pos = mi
            dist_from_me = 0
            while is_valid_coord(pos, scan_off):
                pos += scan_off
                dist_from_me += 1
                sv = state[pos]
                if sv == 0:
                    continue
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

    my_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in my_stacks)
    opp_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in opp_stacks)

    my_mobility = 0
    opp_mobility = 0
    for mi, mh in my_stacks:
        for nb in ADJACENTS[mi]:
            nb_val = state[nb]
            if nb_val == 0:
                my_mobility += 1
            elif nb_val & my_flag:
                my_mobility += 1
            elif nb_val & opp_flag:
                if mh >= (nb_val & HEIGHT_MASK):
                    my_mobility += 2
    for oi, oh in opp_stacks:
        for nb in ADJACENTS[oi]:
            nb_val = state[nb]
            if nb_val == 0:
                opp_mobility += 1
            elif nb_val & opp_flag:
                opp_mobility += 1
            elif nb_val & my_flag:
                if oh >= (nb_val & HEIGHT_MASK):
                    opp_mobility += 2
    mobility_diff = (my_mobility - opp_mobility) * 0.4

    edge_death_penalty = 0.0
    edge_death_bonus = 0.0
    for mi, mh in my_stacks:
        ed = EDGE_DIST[mi]
        if ed >= 3:
            continue
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
        if ed >= 3:
            continue
        min_enemy_dist = 99
        nearest_my_h = 0
        for mi, mh in my_stacks:
            d = MANHATTAN[mi][oi]
            if d < min_enemy_dist:
                min_enemy_dist = d
                nearest_my_h = mh
        if nearest_my_h > oh:
            edge_danger = (3 - ed) * 4.0
            if min_enemy_dist <= 2:
                prox_factor = 3.0
            elif min_enemy_dist <= 4:
                prox_factor = 2.0
            elif min_enemy_dist <= 6:
                prox_factor = 1.0
            else:
                prox_factor = 0.5
            edge_death_bonus += oh * edge_danger * prox_factor
        elif nearest_my_h == oh:
            edge_danger = (3 - ed) * 2.0
            if min_enemy_dist <= 3:
                edge_death_bonus += oh * edge_danger * 1.5

    convergence = 0.0
    if my_stacks and opp_stacks:
        if we_are_tallest:
            min_kill_dist = 99
            for mi, mh in my_stacks:
                if mh != my_max_h:
                    continue
                for oi, oh in opp_stacks:
                    if mh > oh:
                        d = MANHATTAN[mi][oi]
                        if d < min_kill_dist:
                            min_kill_dist = d
            if min_kill_dist < 99:
                convergence = max(0, 8 - min_kill_dist) * 3.0
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

    # ── CENTRE CASCADE / MERGE PLAN ─────────────────────
    # Early/mid game: prefer h3/h4 around centre, then use h4 as a
    # cascade-control weapon. Do not let this create h8/h9/h10 monsters.
    early_factor = max(0.0, min(1.0, (total - 10.0) / 14.0))

    centre_power = 0.0
    centre_merge_potential = 0.0
    centre_debris_penalty = 0.0
    central_friendly = [(i, h) for i, h in my_stacks if i in MOVEMENT_CENTER_16]

    for mi, mh in my_stacks:
        cd = _dist_to_center(mi)

        if mi in MOVEMENT_CENTER_4:
            if mh == 3:
                centre_power += 16.0
            elif mh == 4:
                centre_power += 34.0
            elif mh == 5:
                centre_power += 18.0 if final_stage else -18.0
            elif mh > 5:
                centre_power -= (mh - 5) * 35.0 + mh * 6.0
            else:
                centre_power += mh * 2.0

        elif mi in MOVEMENT_CENTER_16:
            if mh == 3:
                centre_power += 10.0
            elif mh == 4:
                centre_power += 24.0
            elif mh == 5:
                centre_power += 10.0 if final_stage else -22.0
            elif mh > 5:
                centre_power -= (mh - 5) * 40.0 + mh * 8.0
            else:
                centre_power += max(0.0, 4.0 - cd) * mh

        else:
            if early_factor > 0.0 and mh >= 3:
                centre_power -= cd * 1.2

        # h1 debris can help merge, but should not be parked next to enemies.
        if mh == 1 and mi in MOVEMENT_CENTER_16:
            enemy_d = _closest_dist(mi, opp_stacks)
            if enemy_d <= 1:
                centre_debris_penalty += 7.0
            elif enemy_d <= 2:
                centre_debris_penalty += 3.0
            else:
                centre_merge_potential += 1.5

    for a_i in range(len(central_friendly)):
        ai, ah = central_friendly[a_i]
        for b_i in range(a_i + 1, len(central_friendly)):
            bi, bh = central_friendly[b_i]
            d = MANHATTAN[ai][bi]
            if ah >= 3 and bh >= 3 and d <= 2:
                centre_merge_potential += (3 - d) * 8.0
            elif max(ah, bh) >= 3 and min(ah, bh) == 1 and d <= 2:
                centre_merge_potential += (3 - d) * 3.0

    # Late/cascade finishing: if we have a tall stack, spend that height to
    # pressure edge/corner enemies instead of hoarding B8/B9/B10.
    cascade_finish = 0.0
    central_killers = [(i, h) for i, h in my_stacks if h >= 4]

    if central_killers:
        for oi, oh in opp_stacks:
            edge_pressure = max(0, 3 - EDGE_DIST[oi])
            corner_pressure = 2.0 if oi in CORNER_RING else 0.0
            for ki, kh in central_killers:
                d = MANHATTAN[ki][oi]
                if d <= kh + 1 and kh >= oh:
                    cascade_finish += edge_pressure * 8.0
                    cascade_finish += corner_pressure * 10.0
                    cascade_finish += max(0, 8 - d) * 2.0
                    if kh > 5:
                        cascade_finish += (kh - 5) * 30.0
                    if opp_stack_n <= 2:
                        cascade_finish += oh * 8.0
                elif d <= kh + 2 and kh > oh:
                    cascade_finish += edge_pressure * 2.0

    centre_strategy = (
        centre_power * (0.7 + early_factor * 0.8)
        + centre_merge_potential * (0.8 + early_factor * 0.7)
        - centre_debris_penalty * survival_weight
        + cascade_finish * (0.8 + late_factor * 1.4) * elimination_weight
    )

    elim = 0.0
    if opp_stack_n <= 1:
        elim += 5.0
    elif opp_stack_n == 2:
        elim += 2.0
    if my_stack_n <= 1:
        elim -= 5.0
    elif my_stack_n == 2:
        elim -= 2.0

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
        + height_adv
        - height_disadv
        + height_preservation
        - max_height_penalty
        + opp_overstack_bonus
        + edge_trap_bonus * elimination_weight
        - edge_trap_penalty * survival_weight
        + mobility_diff
        - edge_death_penalty * survival_weight
        + edge_death_bonus * elimination_weight
        + convergence
        + centre_strategy
        + elim * elimination_weight
        + (my_safety - opp_safety) * survival_weight
    )
    return score


# =============================================================================
# Placement phase heuristic
# =============================================================================

_ZONE = frozenset({
    18, 19, 20, 21,   # (2,2) (2,3) (2,4) (2,5)
    26, 27, 28, 29,   # (3,2) (3,3) (3,4) (3,5)
    34, 35, 36, 37,   # (4,2) (4,3) (4,4) (4,5)
    42, 43, 44, 45,   # (5,2) (5,3) (5,4) (5,5)
})

_CENTER_4 = frozenset({27, 28, 35, 36})

_DIST_TO_CENTER = [0] * 64
for _i in range(64):
    _r, _c = divmod(_i, 8)
    _DIST_TO_CENTER[_i] = min(
        abs(_r - 3) + abs(_c - 3), abs(_r - 3) + abs(_c - 4),
        abs(_r - 4) + abs(_c - 3), abs(_r - 4) + abs(_c - 4),
    )

_ADJ_SET = [set() for _ in range(64)]
for _i in range(64):
    _r, _c = divmod(_i, 8)
    if _r > 0: _ADJ_SET[_i].add(_i - 8)
    if _r < 7: _ADJ_SET[_i].add(_i + 8)
    if _c > 0: _ADJ_SET[_i].add(_i - 1)
    if _c < 7: _ADJ_SET[_i].add(_i + 1)


def _detect_compact_L(stacks):
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
                d1, d2 = n1 - elbow, n2 - elbow
                is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or \
                          (d1 in (-1, 1) and d2 in (-8, 8))
                if not is_perp:
                    continue
                score = 40.0
                cc = sum(1 for s in (elbow, n1, n2) if s in _CENTER_4)
                if cc >= 2:   score += 40.0
                elif cc == 1: score += 28.0
                else:         score += max(0, 15.0 - _DIST_TO_CENTER[elbow] * 5.0)
                avg_cd = (_DIST_TO_CENTER[elbow] + _DIST_TO_CENTER[n1] + _DIST_TO_CENTER[n2]) / 3.0
                score += max(0, 12.0 - avg_cd * 3.0)
                if score > best_score:
                    best_score = score
                    best_elbow = elbow
                    best_L = [elbow, n1, n2]
    return best_score, best_elbow, best_L


def _is_L_completion(a, b, candidate):
    d_ab = b - a
    for base in (a, b):
        for pd in (-8, 8, -1, 1):
            if pd == d_ab or pd == -d_ab:
                continue
            if base + pd == candidate:
                return True
    return False


def _count_blocked_enemy_cells(idx, enemy, f_set, e_set):
    my_adj = _ADJ_SET[idx]
    enemy_wants = set()
    for e in enemy:
        for nb in _ADJ_SET[e]:
            if nb not in e_set and nb not in f_set:
                enemy_wants.add(nb)
    return len(enemy_wants & my_adj)


def eval_place_L(board: BoardState, root_color: int, is_red: bool) -> float:
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

    # ── ZONE ENFORCEMENT ──
    for idx in friendly:
        if idx in _ZONE:
            score += 10.0
            if idx in _CENTER_4:
                score += 20.0
            score += (2 - _DIST_TO_CENTER[idx]) * 5.0
        else:
            score -= 50.0

    score += (f_count - e_count) * 10.0

    cr, cc = 3.5, 3.5
    if enemy:
        e_avg_r = sum(e >> 3 for e in enemy) / e_count
        e_avg_c = sum(e & 7 for e in enemy) / e_count
        toward_r, toward_c = e_avg_r - cr, e_avg_c - cc
        away_r, away_c = -toward_r, -toward_c
    else:
        toward_r = toward_c = away_r = away_c = 0.0

    # ── PHASE 1 ──
    if f_count <= 1:
        pass

    # ── PHASE 2: center + blocker ──
    # FIX: penalize dist >= 4 (blocker too far = isolated),
    #      tighter toward-enemy cap (was 10, now 6)
    elif f_count == 2:
        a, b = friendly[0], friendly[1]
        dist_ab = MANHATTAN[a][b]

        if dist_ab == 1:
            score -= 20.0       # adjacent = bad, don't start L yet
        elif dist_ab == 2:
            score += 8.0        # ideal blocker distance
        elif dist_ab == 3:
            score += 2.0        # acceptable
        else:
            score -= 15.0       # ★ FIX: dist >= 4, will be isolated

        if a in _CENTER_4 and b not in _CENTER_4:
            blocker = b
        elif b in _CENTER_4 and a not in _CENTER_4:
            blocker = a
        else:
            blocker = b if _DIST_TO_CENTER[a] < _DIST_TO_CENTER[b] else a

        if enemy:
            br, bc = blocker >> 3, blocker & 7
            dot = (br - cr) * toward_r + (bc - cc) * toward_c
            score += min(dot * 5.0, 6.0)  # ★ FIX: cap 10 → 6

            score += _count_blocked_enemy_cells(blocker, enemy, f_set, e_set) * 8.0

    # ── PHASE 3: complete L or place blocker ──
    elif f_count == 3:
        pairs = []
        for i in range(3):
            for j in range(i + 1, 3):
                if MANHATTAN[friendly[i]][friendly[j]] == 1:
                    third = friendly[3 - i - j]
                    pairs.append((friendly[i], friendly[j], third))

        if not pairs:
            score -= 25.0
        else:
            best = -9999.0
            for a, b, third in pairs:
                cfg = 0.0
                if a in _CENTER_4 or b in _CENTER_4:
                    cfg += 25.0

                if _is_L_completion(a, b, third):
                    cfg += 50.0
                    l_all = [a, b, third]
                    cfg += sum(1 for s in l_all if s in _CENTER_4) * 10.0
                    if enemy:
                        lr = sum(s >> 3 for s in l_all) / 3.0
                        lc = sum(s & 7 for s in l_all) / 3.0
                        cfg += ((lr - cr) * away_r + (lc - cc) * away_c) * 5.0
                else:
                    cfg += 5.0
                    if enemy:
                        tr, tc2 = third >> 3, third & 7
                        dot = (tr - cr) * toward_r + (tc2 - cc) * toward_c
                        cfg += min(dot * 3.0, 6.0)
                    if enemy:
                        cfg += _count_blocked_enemy_cells(third, enemy, f_set, e_set) * 4.0
                    if min(MANHATTAN[third][a], MANHATTAN[third][b]) >= 2:
                        cfg += 6.0

                if cfg > best:
                    best = cfg
            score += best

    # ── PHASE 4: full L + blocker ──
    elif f_count >= 4:
        l_score, _, l_stacks = _detect_compact_L(friendly)
        score += l_score

        if l_stacks:
            l_set = set(l_stacks)
            elbow, arm1, arm2 = l_stacks[0], l_stacks[1], l_stacks[2]

            if enemy:
                lr = sum(s >> 3 for s in l_stacks) / 3.0
                lc = sum(s & 7 for s in l_stacks) / 3.0
                score += ((lr - cr) * away_r + (lc - cc) * away_c) * 5.0

            for idx in friendly:
                if idx in l_set:
                    continue

                extending = False
                for arm in (arm1, arm2):
                    d = arm - elbow
                    if MANHATTAN[idx][arm] == 1 and (idx - arm) == d:
                        extending = True; break
                    if MANHATTAN[idx][elbow] == 1 and (idx - elbow) == -d:
                        extending = True; break

                if extending:
                    score -= 30.0
                else:
                    if enemy:
                        br, bc = idx >> 3, idx & 7
                        dot = (br - cr) * toward_r + (bc - cc) * toward_c
                        score += min(dot * 4.0, 8.0)

                    if enemy:
                        score += _count_blocked_enemy_cells(idx, enemy, f_set, e_set) * 5.0

                    min_d_to_L = min(MANHATTAN[idx][s] for s in l_stacks)
                    if min_d_to_L >= 2:
                        score += 8.0
                    if min_d_to_L >= 4:
                        score -= 12.0   # ★ FIX: blocker too far from L

        if friendly and enemy:
            our_adj = set()
            for f in friendly:
                our_adj.update(_ADJ_SET[f])
            ew = set()
            for e in enemy:
                for nb in _ADJ_SET[e]:
                    if board.is_empty(nb):
                        ew.add(nb)
            score += len(ew & our_adj) * 4.0

        if e_count >= 3:
            el, _, _ = _detect_compact_L(enemy)
            score -= el * 0.5

    return score


# =============================================================================
# Compatibility aliases
# =============================================================================

def eval_place_co0(board, root_color, is_red=True):
    return eval_place_L(board, root_color, is_red)

def eval_place_co1(board, root_color, is_red=True):
    return eval_place_L(board, root_color, is_red)

def eval_place_co2(board, root_color, is_red=True):
    return eval_place_L(board, root_color, is_red)

def eval0(board, root_color, is_red):
    return evaluate(board, root_color == 1)

def eval1(board, root_color, is_red):
    return evaluate(board, root_color == 1)