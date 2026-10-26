"""Distinct movement formulas shared by subsets of the historical agents.

Agents import the appropriate implementation under their existing local name.
These variants deliberately retain their original scores and argument lists.
"""

from .geometry import CENTER_WEIGHTS


def calc_positioning_unbounded_height(friendly_map: dict, enemy_map: dict) -> float:
    score = 0.0
    # Friendly positioning (plus Artillery bonus)
    for idx, h in friendly_map.items():
        weight = CENTER_WEIGHTS[idx]
        score += (weight * h / 2.0)
        if h >= 4 and weight >= 3:
            score += (h * weight * 1.5)

    # Enemy positioning (plus Artillery penalty)
    for idx, h in enemy_map.items():
        weight = CENTER_WEIGHTS[idx]
        score -= (weight * h / 2.0)
        if h >= 4 and weight >= 3:
            score -= (h * weight * 1.5)
    return score


def calc_efficiency_height_difference(friendly_map: dict, enemy_map: dict) -> float:
    f_score = 0.0
    for h in friendly_map.values():
        # if h == 1: f_score -= 3.0
        if 2 <= h <= 3: f_score += 3.0
        elif 4 <= h <= 5: f_score += 15.0
        elif h == 6: f_score -= 10.0
        elif h > 6: f_score -= (30.0 + ((h - 6) * 10.0))

    e_score = 0.0
    for h in enemy_map.values():
        # if h == 1: e_score -= 3.0
        if 2 <= h <= 3: e_score += 3.0
        elif 4 <= h <= 5: e_score += 15.0
        elif h == 6: e_score -= 10.0
        elif h > 6: e_score -= (30.0 + ((h - 6) * 10.0))
    return f_score - e_score


def calc_efficiency_with_single_penalty(friendly_map: dict, enemy_map: dict) -> float:
    f_score = 0.0
    for h in friendly_map.values():
        if h == 1: f_score -= 3.0
        elif 2 <= h <= 3: f_score += 3.0
        elif 4 <= h <= 5: f_score += 15.0
        elif h == 6: f_score -= 10.0
        elif h > 6: f_score -= (30.0 + ((h - 6) * 10.0))

    e_score = 0.0
    for h in enemy_map.values():
        if h == 1: e_score -= 3.0
        elif 2 <= h <= 3: e_score += 3.0
        elif 4 <= h <= 5: e_score += 15.0
        elif h == 6: e_score -= 10.0
        elif h > 6: e_score -= (30.0 + ((h - 6) * 10.0))
    return f_score - e_score


def calc_efficiency_friendly_only(friendly_map: dict) -> float:
    score = 0.0
    for h in friendly_map.values():
        if h == 1: score -= 10.0
        elif 2 <= h <= 3: score += 5.0
        elif 4 <= h <= 6: score += 15.0
        elif h == 7: score -= 15.0
        elif h > 7: score -= (30.0 + ((h - 7) * 10.0))
    return score


def calc_safety_all_heights(friendly_map: dict, enemy_map: dict, is_my_turn: bool) -> float:
    score = 0.0
    for e_idx, e_h in enemy_map.items():
        er, ec = divmod(e_idx, 8)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Threat A: Direct EAT
            adj_r, adj_c = er + dr, ec + dc
            if 0 <= adj_r < 8 and 0 <= adj_c < 8:
                adj_idx = adj_r * 8 + adj_c
                if adj_idx in friendly_map and e_h >= friendly_map[adj_idx]:
                    threat_h = friendly_map[adj_idx]
                    score -= (threat_h * 2.0) if is_my_turn else (threat_h * 10.0)

            # Threat B: CASCADE Push
            f_in_line = 0
            for step in range(1, e_h + 1):
                tr, tc = er + (dr * step), ec + (dc * step)
                if not (0 <= tr < 8 and 0 <= tc < 8): break

                t_idx = tr * 8 + tc
                if t_idx in friendly_map or t_idx in enemy_map:
                    f_in_line += 1
                    push_r, push_c = er + (dr * (e_h + f_in_line)), ec + (dc * (e_h + f_in_line))
                    if not (0 <= push_r < 8 and 0 <= push_c < 8):
                        if t_idx in friendly_map:
                            threat_h = friendly_map[t_idx]
                            score -= (threat_h * 2.0) if is_my_turn else (threat_h * 10.0)
    return score


def calc_safety_friendly_pushes(friendly_map: dict, enemy_map: dict, is_my_turn: bool) -> float:
    score = 0.0
    for e_idx, e_h in enemy_map.items():
        er, ec = divmod(e_idx, 8)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Threat A: Direct EAT
            adj_r, adj_c = er + dr, ec + dc
            if 0 <= adj_r < 8 and 0 <= adj_c < 8:
                adj_idx = adj_r * 8 + adj_c
                if adj_idx in friendly_map and e_h >= friendly_map[adj_idx]:
                    threat_h = friendly_map[adj_idx]
                    score -= (threat_h * 2.0) if is_my_turn else (threat_h * 10.0)

            # Threat B: CASCADE Push
            f_in_line = 0
            for step in range(1, e_h + 1):
                tr, tc = er + (dr * step), ec + (dc * step)
                if not (0 <= tr < 8 and 0 <= tc < 8): break

                t_idx = tr * 8 + tc
                if t_idx in friendly_map:
                    f_in_line += 1
                    push_r, push_c = er + (dr * (e_h + f_in_line)), ec + (dc * (e_h + f_in_line))
                    if not (0 <= push_r < 8 and 0 <= push_c < 8):
                        threat_h = friendly_map[t_idx]
                        score -= (threat_h * 2.0) if is_my_turn else (threat_h * 10.0)
    return score


def calc_hunting_unweighted_fear(friendly_map: dict, enemy_map: dict) -> float:
    score = 0.0
    MAX_DIST = 14.0

    for f_idx, f_h in friendly_map.items():
        f_row, f_col = divmod(f_idx, 8)

        # Track the best target to chase, and the scariest threat to flee
        best_hunt = 0.0
        worst_fear = 0.0

        for e_idx, e_h in enemy_map.items():
            e_row, e_col = divmod(e_idx, 8)
            dist = abs(f_row - e_row) + abs(f_col - e_col)
            size_diff = f_h - e_h

            # PREDATOR LOGIC (We are bigger)
            if size_diff > 0:
                effective_diff = min(size_diff, 1)
                # Scaled down to prevent out-weighing the Material score
                hunt_val = ((MAX_DIST - dist) / 5.0) * effective_diff
                if hunt_val > best_hunt:
                    best_hunt = hunt_val

            # PINNING LOGIC (We are equal)
            elif size_diff == 0:
                if dist == 2:
                    dx, dy = abs(f_row - e_row), abs(f_col - e_col)
                    if dx == 1 and dy == 1:
                        # Scaled down hard diagonal pin
                        hunt_val = ((MAX_DIST - dist) / 5.0) * 1.5
                        if hunt_val > best_hunt:
                            best_hunt = hunt_val

            # PREY LOGIC (We are smaller)
            elif size_diff < 0:
                # 1-height tokens are ammo, they don't feel fear.
                if f_h > 1:
                    # The closer the big enemy is, the higher the fear value
                    fear_val = ((MAX_DIST - dist) / 5.0)
                    if fear_val > worst_fear:
                        worst_fear = fear_val

        # Add the desire to chase, subtract the desire to flee
        score += best_hunt
        score -= worst_fear

    return score


def calc_hunting_all_pairs(friendly_map: dict, enemy_map: dict) -> float:
    score = 0.0
    MAX_DIST = 14.0
    for f_idx, f_h in friendly_map.items():
        fr, fc = divmod(f_idx, 8)
        for e_idx, e_h in enemy_map.items():
            er, ec = divmod(e_idx, 8)
            dist = abs(fr - er) + abs(fc - ec)
            size_diff = f_h - e_h

            if size_diff > 0:
                score += (MAX_DIST - dist) * size_diff
            elif size_diff == 0:
                if dist == 1:
                    score -= 30.0
                elif dist == 2:
                    # User update: ONLY reward the hard diagonal pin.
                    dx, dy = abs(fr - er), abs(fc - ec)
                    if dx == 1 and dy == 1:
                        score += (MAX_DIST - dist) * 3.0
            elif size_diff < 0:
                score -= (MAX_DIST - dist) * abs(size_diff)
    return score
