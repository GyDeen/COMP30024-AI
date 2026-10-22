"""
Cascade v11 evaluation wired for agent_ab1 minimax.

search.py calls eval_function(board, root_color, is_red_to_move).
This wrapper scores every leaf from the root agent's colour, which is the
correct perspective for alpha-beta minimax.
"""
from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from .action import is_valid_coord, UP, DOWN, LEFT, RIGHT, eat, cascade

INF = 999999
RED = 1
BLUE = 0
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

MANHATTAN = [[0] * 64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        MANHATTAN[_a][_b] = abs((_a >> 3) - (_b >> 3)) + abs((_a & 7) - (_b & 7))


def eval_v11(board: BoardState, root_color: int, is_red_to_move: bool) -> float:
    return evaluate(board, root_color == RED)


def _quiescence_eval(board: BoardState, is_red_maximizing: bool):
    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    best_eat_board = None
    best_eat_gain = 0

    for idx in range(64):
        val = board.state[idx]
        if not (val & my_flag):
            continue
        h = val & HEIGHT_MASK
        for offset in OFFSETS:
            if not is_valid_coord(idx, offset):
                continue
            tv = board.state[idx + offset]
            if not (tv & opp_flag):
                continue
            th = tv & HEIGHT_MASK
            if h < th:
                continue
            dest = idx + offset
            nb = eat(board, idx, offset)
            safe = True
            for check_off in OFFSETS:
                if not is_valid_coord(dest, check_off):
                    continue
                cv = nb.state[dest + check_off]
                if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= h:
                    safe = False
                    break
            if safe and th > best_eat_gain:
                best_eat_gain = th
                best_eat_board = nb
    return best_eat_board


def _opponent_cascade_threat(board: BoardState, is_red_maximizing: bool):
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    my_tokens = board.red_tokens if is_red_maximizing else board.blue_tokens
    opp_tokens = board.blue_tokens if is_red_maximizing else board.red_tokens
    worst_board = None
    worst_loss = 0

    for idx in range(64):
        val = board.state[idx]
        if not (val & opp_flag):
            continue
        if (val & HEIGHT_MASK) < 2:
            continue
        for offset in OFFSETS:
            nb = cascade(board, idx, offset)
            nb_my = nb.red_tokens if is_red_maximizing else nb.blue_tokens
            nb_opp = nb.blue_tokens if is_red_maximizing else nb.red_tokens
            net_damage = (my_tokens - nb_my) - (opp_tokens - nb_opp)
            if net_damage > worst_loss:
                worst_loss = net_damage
                worst_board = nb
            if nb_my == 0:
                return nb, my_tokens
    return worst_board, worst_loss


def evaluate(board: BoardState, is_red_maximizing: bool) -> float:
    if board.red_tokens == 0:
        return -INF if is_red_maximizing else INF
    if board.blue_tokens == 0:
        return INF if is_red_maximizing else -INF

    q_board = _quiescence_eval(board, is_red_maximizing)
    if q_board is not None:
        return _static_eval(q_board, is_red_maximizing) + 3.0

    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    if any(v and (v & opp_flag) and (v & HEIGHT_MASK) >= 2 for v in board.state):
        cascade_board, cascade_loss = _opponent_cascade_threat(board, is_red_maximizing)
        if cascade_loss > 0 and cascade_board is not None:
            static_score = _static_eval(board, is_red_maximizing)
            cb_my = cascade_board.red_tokens if is_red_maximizing else cascade_board.blue_tokens
            if cb_my == 0:
                return -INF + 100
            post_score = _static_eval(cascade_board, is_red_maximizing)
            if cascade_loss >= 3:
                return post_score * 0.7 + static_score * 0.3
            return post_score * 0.4 + static_score * 0.6

    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    if any(v and (v & my_flag) and (v & HEIGHT_MASK) >= 2 for v in board.state):
        my_tokens = board.red_tokens if is_red_maximizing else board.blue_tokens
        opp_tokens = board.blue_tokens if is_red_maximizing else board.red_tokens
        best_board = None
        best_net_gain = 0
        for idx in range(64):
            val = board.state[idx]
            if not (val & my_flag) or (val & HEIGHT_MASK) < 2:
                continue
            for offset in OFFSETS:
                nb = cascade(board, idx, offset)
                nb_opp = nb.blue_tokens if is_red_maximizing else nb.red_tokens
                nb_my = nb.red_tokens if is_red_maximizing else nb.blue_tokens
                net = (opp_tokens - nb_opp) - (my_tokens - nb_my)
                if nb_opp == 0 and nb_my > 0:
                    return INF - 100
                if net > best_net_gain:
                    best_net_gain = net
                    best_board = nb
        if best_board is not None and best_net_gain >= 3:
            return _static_eval(best_board, is_red_maximizing) * 0.6 + _static_eval(board, is_red_maximizing) * 0.4

    return _static_eval(board, is_red_maximizing)


def _pincer_score(attackers, victims):
    score = 0.0
    for i, (a1, h1) in enumerate(attackers):
        for a2, h2 in attackers[i + 1:]:
            r1, c1 = a1 >> 3, a1 & 7
            r2, c2 = a2 >> 3, a2 & 7
            if r1 == r2:
                lo, hi = min(c1, c2), max(c1, c2)
                for bi, bh in victims:
                    br, bc = bi >> 3, bi & 7
                    if br == r1 and lo < bc < hi:
                        gap = hi - lo - 1
                        s = bh * 2.0
                        if h1 >= bh and h2 >= bh:
                            s += bh * 3.0
                        if gap <= 2:
                            s += bh * 2.0
                        if min(br, 7 - br) <= 1:
                            s += bh * 2.0
                        score += s
            if c1 == c2:
                lo, hi = min(r1, r2), max(r1, r2)
                for bi, bh in victims:
                    br, bc = bi >> 3, bi & 7
                    if bc == c1 and lo < br < hi:
                        gap = hi - lo - 1
                        s = bh * 2.0
                        if h1 >= bh and h2 >= bh:
                            s += bh * 3.0
                        if gap <= 2:
                            s += bh * 2.0
                        if min(bc, 7 - bc) <= 1:
                            s += bh * 2.0
                        score += s
    return score


def _corner_trap_score(attackers, victims):
    score = 0.0
    for vi, vh in victims:
        vr, vc = vi >> 3, vi & 7
        victim_ed = EDGE_DIST[vi]
        if victim_ed > 1:
            continue
        attack_angles = 0
        total_attack_score = 0.0
        for ai, ah in attackers:
            ar, ac = ai >> 3, ai & 7
            if ar == vr:
                dist = abs(ac - vc)
                if dist >= 1 and ah >= dist:
                    push_dist = ah - dist
                    if (ac < vc and vc + push_dist >= 8) or (ac > vc and vc - push_dist < 0):
                        attack_angles += 1
                        total_attack_score += vh * 3.0
            if ac == vc:
                dist = abs(ar - vr)
                if dist >= 1 and ah >= dist:
                    push_dist = ah - dist
                    if (ar < vr and vr + push_dist >= 8) or (ar > vr and vr - push_dist < 0):
                        attack_angles += 1
                        total_attack_score += vh * 3.0
            if victim_ed == 0 and MANHATTAN[ai][vi] == 2:
                dr = abs(ar - vr)
                dc = abs(ac - vc)
                if dr == 1 and dc == 1 and ah >= 1:
                    attack_angles += 2
                    total_attack_score += vh * 6.0
        if attack_angles >= 2:
            score += total_attack_score * 2.0
        elif attack_angles == 1:
            score += total_attack_score * 0.8
    return score


def _static_eval(board: BoardState, is_red_maximizing: bool) -> float:
    red_tokens = board.red_tokens
    blue_tokens = board.blue_tokens
    if red_tokens == 0:
        return -INF if is_red_maximizing else INF
    if blue_tokens == 0:
        return INF if is_red_maximizing else -INF

    my_flag = RED_FLAG if is_red_maximizing else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
    my_tokens = red_tokens if is_red_maximizing else blue_tokens
    opp_tokens = blue_tokens if is_red_maximizing else red_tokens

    my_stacks = []
    opp_stacks = []
    for idx, val in enumerate(board.state):
        if val == 0:
            continue
        h = val & HEIGHT_MASK
        if val & my_flag:
            my_stacks.append((idx, h))
        else:
            opp_stacks.append((idx, h))

    if not my_stacks:
        return -INF
    if not opp_stacks:
        return INF

    my_stack_n = len(my_stacks)
    opp_stack_n = len(opp_stacks)
    total = my_tokens + opp_tokens
    advantage = (my_tokens - opp_tokens) / total
    late_factor = max(0.0, 1.0 - total / 16.0)

    if my_stack_n <= 1:
        survival_weight, elimination_weight = 2.5, 0.5
    elif my_stack_n == 2 and advantage < 0:
        survival_weight, elimination_weight = 2.0, 0.5
    elif advantage < -0.1:
        survival_weight, elimination_weight = 1.5 + late_factor * 0.5, 0.6
    elif advantage > 0.2:
        survival_weight, elimination_weight = 0.8 - late_factor * 0.3, 1.5 + late_factor * 0.5
    else:
        survival_weight, elimination_weight = 1.0, 1.0

    my_max_h = max(h for _, h in my_stacks)
    opp_max_h = max(h for _, h in opp_stacks)
    we_are_tallest = my_max_h > opp_max_h

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
                vuln = 1.0
                if oh <= 2:
                    vuln = 3.0 if all(oi2 == oi or MANHATTAN[oi][oi2] > 2 for oi2, _ in opp_stacks) else 2.0
                proximity += (1 + surplus) * max(0, 8 - d) * vuln * 0.25
            elif mh == oh and advantage > 0.1:
                proximity += max(0, 8 - d) * 0.08

    my_threatened = 0.0
    opp_threatened = 0.0
    for mi, mh in my_stacks:
        worst = 0
        for nb in ADJACENTS[mi]:
            v = board.state[nb]
            if v and (v & opp_flag):
                nh = v & HEIGHT_MASK
                if nh > mh:
                    worst = 2
                    break
                if nh >= mh:
                    worst = max(worst, 1)
        if worst == 2:
            my_threatened += mh
        elif worst == 1:
            my_threatened += mh * (1.5 if my_stack_n <= 1 else 0.8 if my_stack_n == 2 else 0.3)
    for oi, oh in opp_stacks:
        worst = 0
        for nb in ADJACENTS[oi]:
            v = board.state[nb]
            if v and (v & my_flag):
                nh = v & HEIGHT_MASK
                if nh > oh:
                    worst = 2
                    break
                if nh >= oh:
                    worst = max(worst, 1)
        if worst == 2:
            opp_threatened += oh
        elif worst == 1:
            opp_threatened += oh * (1.5 if opp_stack_n <= 1 else 0.8 if opp_stack_n == 2 else 0.3)

    my_threats = 0.0
    opp_threats = 0.0
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            d = MANHATTAN[mi][oi]
            if d <= 3 and mh > oh:
                my_threats += oh * (3.0 if d <= 1 else 1.5 if d == 2 else 0.5)
    for oi, oh in opp_stacks:
        for mi, mh in my_stacks:
            d = MANHATTAN[oi][mi]
            if d <= 3 and oh > mh:
                opp_threats += mh * (3.0 if d <= 1 else 1.5 if d == 2 else 0.5)
    initiative = my_threats - opp_threats

    my_debris_penalty = 0.0
    opp_debris_penalty = 0.0
    for mi, mh in my_stacks:
        if mh <= 1:
            closest = min(MANHATTAN[mi][oi] for oi, _ in opp_stacks)
            my_debris_penalty += 3.0 if closest <= 1 else 2.0 if closest <= 2 else 1.0 if closest <= 3 else 0.0
    for oi, oh in opp_stacks:
        if oh <= 1:
            closest = min(MANHATTAN[oi][mi] for mi, _ in my_stacks)
            opp_debris_penalty += 3.0 if closest <= 1 else 2.0 if closest <= 2 else 1.0 if closest <= 3 else 0.0

    dominance = (sum(1 for _, h in my_stacks if h > opp_max_h) - sum(1 for _, h in opp_stacks if h > my_max_h)) * 3.0
    height_adv = max(0, my_max_h - opp_max_h) * 5.0
    height_disadv = max(0, opp_max_h - my_max_h) * 5.0
    height_preservation = (sum(h * h for _, h in my_stacks) - sum(h * h for _, h in opp_stacks)) * 0.5

    edge_trap_bonus = 0.0
    edge_trap_penalty = 0.0
    for oi, oh in opp_stacks:
        if EDGE_DIST[oi] > 2:
            continue
        for d_idx, scan_off in enumerate(SCAN_OFFSETS):
            if FAR_EDGE_DIST[oi * 4 + d_idx] > 2:
                continue
            pos, dist_from_opp = oi, 0
            while is_valid_coord(pos, scan_off):
                pos += scan_off
                dist_from_opp += 1
                sv = board.state[pos]
                if sv == 0:
                    continue
                if sv & my_flag and (sv & HEIGHT_MASK) >= FAR_EDGE_DIST[pos * 4 + d_idx]:
                    edge_trap_bonus += oh * max(0, 6 - dist_from_opp)
                break
    for mi, mh in my_stacks:
        if EDGE_DIST[mi] > 2:
            continue
        for d_idx, scan_off in enumerate(SCAN_OFFSETS):
            if FAR_EDGE_DIST[mi * 4 + d_idx] > 2:
                continue
            pos, dist_from_me = mi, 0
            while is_valid_coord(pos, scan_off):
                pos += scan_off
                dist_from_me += 1
                sv = board.state[pos]
                if sv == 0:
                    continue
                if sv & opp_flag:
                    opp_h = sv & HEIGHT_MASK
                    if opp_h >= FAR_EDGE_DIST[pos * 4 + d_idx]:
                        urgency = max(0, 6 - dist_from_me)
                        edge_trap_penalty += mh * urgency * (0.3 if dist_from_me == 1 and mh > opp_h else 1.5)
                break

    my_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in my_stacks)
    opp_safety = sum(EDGE_DIST[i] * h * 0.15 for i, h in opp_stacks)

    my_mobility = 0
    opp_mobility = 0
    for mi, mh in my_stacks:
        for nb in ADJACENTS[mi]:
            v = board.state[nb]
            if v == 0 or (v & my_flag):
                my_mobility += 1
            elif v & opp_flag and mh >= (v & HEIGHT_MASK):
                my_mobility += 2
    for oi, oh in opp_stacks:
        for nb in ADJACENTS[oi]:
            v = board.state[nb]
            if v == 0 or (v & opp_flag):
                opp_mobility += 1
            elif v & my_flag and oh >= (v & HEIGHT_MASK):
                opp_mobility += 2
    mobility_diff = (my_mobility - opp_mobility) * 0.4

    edge_death_penalty = 0.0
    edge_death_bonus = 0.0
    for mi, mh in my_stacks:
        ed = EDGE_DIST[mi]
        if ed >= 3:
            continue
        min_enemy_dist, nearest_enemy_h = min((MANHATTAN[mi][oi], oh) for oi, oh in opp_stacks)
        if nearest_enemy_h > mh:
            prox_factor = 3.0 if min_enemy_dist <= 2 else 2.0 if min_enemy_dist <= 4 else 1.0 if min_enemy_dist <= 6 else 0.5
            edge_death_penalty += mh * (3 - ed) * 4.0 * prox_factor
        elif nearest_enemy_h == mh and min_enemy_dist <= 3:
            edge_death_penalty += mh * (3 - ed) * 2.0 * 1.5
    for oi, oh in opp_stacks:
        ed = EDGE_DIST[oi]
        if ed >= 3:
            continue
        min_enemy_dist, nearest_my_h = min((MANHATTAN[oi][mi], mh) for mi, mh in my_stacks)
        if nearest_my_h > oh:
            prox_factor = 3.0 if min_enemy_dist <= 2 else 2.0 if min_enemy_dist <= 4 else 1.0 if min_enemy_dist <= 6 else 0.5
            edge_death_bonus += oh * (3 - ed) * 4.0 * prox_factor
        elif nearest_my_h == oh and min_enemy_dist <= 3:
            edge_death_bonus += oh * (3 - ed) * 2.0 * 1.5

    convergence = 0.0
    if we_are_tallest:
        min_kill_dist = min((MANHATTAN[mi][oi] for mi, mh in my_stacks if mh == my_max_h for oi, oh in opp_stacks if mh > oh), default=99)
        if min_kill_dist < 99:
            convergence = max(0, 8 - min_kill_dist) * 3.0
    elif my_max_h == opp_max_h:
        min_eat_dist = min((MANHATTAN[mi][oi] for mi, mh in my_stacks for oi, oh in opp_stacks if mh > oh), default=99)
        if min_eat_dist < 99:
            convergence = max(0, 8 - min_eat_dist) * 2.0
        convergence += sum(EDGE_DIST[mi] * 0.5 for mi, _ in my_stacks)
    else:
        for mi, mh in my_stacks:
            min_threat_dist = min((MANHATTAN[mi][oi] for oi, oh in opp_stacks if oh > mh), default=99)
            if min_threat_dist < 99:
                convergence += min(min_threat_dist, 7) * 1.5 + EDGE_DIST[mi] * 3.0

    elim = (5.0 if opp_stack_n <= 1 else 2.0 if opp_stack_n == 2 else 0.0) - (5.0 if my_stack_n <= 1 else 2.0 if my_stack_n == 2 else 0.0)
    pincer_bonus = _pincer_score(my_stacks, opp_stacks)
    pincer_penalty = _pincer_score(opp_stacks, my_stacks)
    corner_trap_bonus = _corner_trap_score(my_stacks, opp_stacks)
    corner_trap_penalty = _corner_trap_score(opp_stacks, my_stacks)

    return (
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
    )
 