"""Closeout features shared by the two closeout-oriented quiescence agents."""

from .geometry import ADJACENT, CENTER_CELLS, CORNERS, EDGE_DIST, MANHATTAN


def _dist_to_center(idx: int) -> int:
    return min(MANHATTAN[idx][c] for c in CENTER_CELLS)


def _dist_to_corner(idx: int) -> int:
    return min(MANHATTAN[idx][c] for c in CORNERS)


def _is_attacked_by(friendly_map: dict, cell: int, target_h: int) -> bool:
    """True if a friendly stack can immediately eat a target of height target_h on cell."""
    for nb in ADJACENT[cell].values():
        if nb in friendly_map and friendly_map[nb] >= target_h:
            return True
    return False


def _safe_escape_count_for_enemy(enemy_idx: int, enemy_h: int,
                                 friendly_map: dict, enemy_map: dict) -> int:
    """
    Approximate how many useful safe moves the enemy stack still has.
    Lower is better for us. Escapes back toward centre / away from edge count more.
    """
    current_center = _dist_to_center(enemy_idx)
    current_edge = EDGE_DIST[enemy_idx]
    count = 0

    for nb in ADJACENT[enemy_idx].values():
        # Enemy cannot safely move into our occupied cell.
        if nb in friendly_map:
            continue

        # If we can immediately eat them after they move there, not safe.
        if _is_attacked_by(friendly_map, nb, enemy_h):
            continue

        next_center = _dist_to_center(nb)
        next_edge = EDGE_DIST[nb]

        # Returning to centre / away from edge is the escape we most want to deny.
        if next_center < current_center:
            count += 2
        elif next_edge > current_edge:
            count += 2
        else:
            count += 1

    return count


def calc_closeout(friendly_map: dict, enemy_map: dict) -> float:
    """
    Anti-draw closeout score.
    When ahead, reward positions that cut off enemy safe moves and force edge/corner traps.
    """
    if not friendly_map or not enemy_map:
        return 0.0

    my_tokens = sum(friendly_map.values())
    enemy_tokens = sum(enemy_map.values())
    total = my_tokens + enemy_tokens
    if total <= 0:
        return 0.0

    advantage = (my_tokens - enemy_tokens) / total
    late_factor = max(0.0, 1.0 - total / 18.0)
    closeout_mode = advantage > 0.05 and (late_factor > 0.15 or len(enemy_map) <= 3 or enemy_tokens <= 7)
    if not closeout_mode:
        return 0.0

    score = 0.0
    my_max_h = max(friendly_map.values())

    for e_idx, e_h in enemy_map.items():
        safe = _safe_escape_count_for_enemy(e_idx, e_h, friendly_map, enemy_map)
        edge_pressure = max(0, 3 - EDGE_DIST[e_idx])
        corner_pressure = max(0, 4 - _dist_to_corner(e_idx))
        center_escape = _dist_to_center(e_idx)

        # Fewer safe escapes means closer to a forced kill.
        score += (5 - safe) * e_h * 5.0

        if safe <= 1:
            score += e_h * 25.0
        elif safe == 2:
            score += e_h * 10.0

        # Edge/corner pressure matters most when we have a nearby stack that can threaten.
        nearest_threat_dist = 99
        nearest_threat_h = 0
        for f_idx, f_h in friendly_map.items():
            if f_h >= e_h:
                d = MANHATTAN[f_idx][e_idx]
                if d < nearest_threat_dist:
                    nearest_threat_dist = d
                    nearest_threat_h = f_h

        if nearest_threat_dist < 99:
            score += edge_pressure * e_h * 4.0
            score += corner_pressure * e_h * 6.0
            score += max(0, 8 - nearest_threat_dist) * 2.5

            # Stand between enemy and centre: this prevents endless running.
            for f_idx, f_h in friendly_map.items():
                if f_h < e_h:
                    continue
                if _dist_to_center(f_idx) <= _dist_to_center(e_idx) and MANHATTAN[f_idx][e_idx] <= 3:
                    score += e_h * 5.0

            if nearest_threat_h >= e_h + 2:
                score += e_h * 8.0

        # Enemy far from centre while we are ahead is good; keep pushing outward.
        score += center_escape * e_h * 1.5

    # As the game gets later / opponent has fewer stacks, scale closeout harder.
    scale = 1.0 + late_factor * 1.5
    if len(enemy_map) <= 2:
        scale += 0.8
    if len(enemy_map) <= 1:
        scale += 1.0

    return score * scale


def calc_max_height_control(friendly_map: dict, enemy_map: dict) -> float:
    """
    Keep the intended height policy:
      h4 = ideal weapon
      h5 = okay only in final winning stage
      h6+ = overstacking unless it is actively finishing.
    Positive score is good for us.
    """
    my_tokens = sum(friendly_map.values())
    enemy_tokens = sum(enemy_map.values())
    total = my_tokens + enemy_tokens
    if total <= 0:
        return 0.0

    advantage = (my_tokens - enemy_tokens) / total
    late_factor = max(0.0, 1.0 - total / 18.0)
    final_stage = advantage > 0.15 and late_factor > 0.35 and len(enemy_map) <= 1

    score = 0.0
    for h in friendly_map.values():
        if h <= 4:
            continue
        if h == 5:
            score += 8.0 if final_stage else -22.0
        else:
            score -= 45.0 + (h - 5) * 35.0

    for h in enemy_map.values():
        if h <= 4:
            continue
        if h == 5:
            score += 12.0 if not final_stage else -5.0
        else:
            score += 35.0 + (h - 5) * 25.0

    return score
