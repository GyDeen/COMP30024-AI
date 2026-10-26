"""Agent-specific evaluation profiles and strategy wrappers."""

from Agents.common.board import BoardState
from Agents.common.features.geometry import (
    ADJACENT, CENTER_4, CENTER_WEIGHTS, EDGE_DIST, MANHATTAN,
)
from Agents.common.features.movement import (
    get_stack_maps,
    calc_material,
    calc_alignment,
)
from Agents.common.features.movement_variants import (
    calc_positioning_unbounded_height as calc_positioning,
    calc_safety_friendly_pushes as calc_safety,
    calc_hunting_all_pairs as calc_hunting,
)
from Agents.common.features.placement_variants import (
    get_place_maps_with_control as get_place_maps,
    calc_place_centrality_average as calc_place_centrality,
    calc_place_zone_control,
    calc_place_synergy_proximity as calc_place_synergy,
)


RED = 1
BLUE = 0


def calc_efficiency(friendly_map: dict) -> float:
    score = 0.0
    for h in friendly_map.values():
        if h == 1: score -= 10.0
        elif h == 2: score += 4.0
        elif h == 3: score += 8.0
        elif h == 4: score += 18.0
        elif h == 5: score -= 8.0
        elif h > 5: score -= (35.0 + ((h - 5) * 25.0))
    return score


CENTER_DIST = [0] * 64


CORNER_DIST = [0] * 64


for _i in range(64):
    CENTER_DIST[_i] = min(MANHATTAN[_i][27], MANHATTAN[_i][28], MANHATTAN[_i][35], MANHATTAN[_i][36])
    CORNER_DIST[_i] = min(MANHATTAN[_i][0], MANHATTAN[_i][7], MANHATTAN[_i][56], MANHATTAN[_i][63])


def _is_attacked_by_stack_map(cell: int, target_h: int, attacker_map: dict) -> bool:
    """True if one of the attacker's adjacent stacks can eat target_h on cell."""
    for nb in ADJACENT[cell].values():
        h = attacker_map.get(nb)
        if h is not None and h >= target_h:
            return True
    return False


def _safe_escape_count_for_enemy(enemy_idx: int, enemy_h: int, friendly_map: dict, enemy_map: dict) -> int:
    """
    Approximate how many useful safe adjacent moves the enemy has.
    Lower is better for us. Moves back toward centre / away from edge count extra
    because those are the moves that normally create draws.
    """
    count = 0
    cur_edge = EDGE_DIST[enemy_idx]
    cur_center = CENTER_DIST[enemy_idx]

    for nb in ADJACENT[enemy_idx].values():
        # Enemy cannot safely move into our occupied square.
        if nb in friendly_map:
            continue

        # If after moving there, our adjacent stack can immediately eat it,
        # this is not a safe escape.
        if _is_attacked_by_stack_map(nb, enemy_h, friendly_map):
            continue

        next_edge = EDGE_DIST[nb]
        next_center = CENTER_DIST[nb]

        # Moving toward centre / away from edge is the escape we really want to deny.
        if next_center < cur_center:
            count += 2
        elif next_edge > cur_edge:
            count += 2
        else:
            count += 1

    return count


def _height_shape_value(h: int, final_stage: bool) -> float:
    """
    Height policy:
      h3 = builder
      h4 = ideal weapon
      h5 = only okay in final winning stage
      h6+ = overstacked; should be used/cascaded, not preserved
    """
    if h == 1:
        return -8.0
    if h == 2:
        return 4.0
    if h == 3:
        return 9.0
    if h == 4:
        return 20.0
    if h == 5:
        return 12.0 if final_stage else -18.0
    return -35.0 - (h - 6) * 25.0


def calc_height_control(friendly_map: dict, enemy_map: dict, final_stage: bool) -> float:
    """Replaces the old 'h4-h6 all good' idea with h4 as the preferred weapon."""
    score = 0.0
    for h in friendly_map.values():
        score += _height_shape_value(h, final_stage)
    for h in enemy_map.values():
        score -= _height_shape_value(h, final_stage) * 0.7
    return score


def calc_closeout_pressure(friendly_map: dict, enemy_map: dict) -> float:
    """
    Anti-draw feature: when ahead, reward reducing enemy safe escapes and pushing
    enemy stacks from centre -> edge -> corner.
    """
    if not friendly_map or not enemy_map:
        return 0.0

    friendly_tokens = sum(friendly_map.values())
    enemy_tokens = sum(enemy_map.values())
    total = friendly_tokens + enemy_tokens
    if total <= 0:
        return 0.0

    advantage = (friendly_tokens - enemy_tokens) / total
    late_factor = max(0.0, 1.0 - total / 16.0)
    enemy_stack_n = len(enemy_map)

    closeout_mode = advantage > 0.08 and (late_factor > 0.20 or enemy_stack_n <= 3 or enemy_tokens <= 7)
    if not closeout_mode:
        return 0.0

    score = 0.0
    for e_idx, e_h in enemy_map.items():
        safe_escapes = _safe_escape_count_for_enemy(e_idx, e_h, friendly_map, enemy_map)
        edge_pressure = max(0, 3 - EDGE_DIST[e_idx])
        corner_pressure = max(0, 4 - CORNER_DIST[e_idx])

        nearest_threat_dist = 99
        nearest_threat_h = 0
        nearest_any_dist = 99

        for f_idx, f_h in friendly_map.items():
            d = MANHATTAN[f_idx][e_idx]
            if d < nearest_any_dist:
                nearest_any_dist = d
            if f_h >= e_h and d < nearest_threat_dist:
                nearest_threat_dist = d
                nearest_threat_h = f_h

        # The core anti-draw signal: fewer safe escapes is good.
        score += (5 - safe_escapes) * e_h * 6.0

        # But only score edge/corner pressure strongly if we actually have a stack
        # that can threaten this enemy.
        if nearest_threat_dist < 99:
            score += edge_pressure * e_h * 5.0
            score += corner_pressure * e_h * 7.0
            score += max(0, 8 - nearest_threat_dist) * 3.0

            if safe_escapes <= 1:
                score += e_h * 24.0
            if safe_escapes == 0:
                score += e_h * 40.0
            if nearest_threat_h >= e_h + 2:
                score += e_h * 10.0
        else:
            # We are ahead but not threatening: chase, but avoid chasing from behind forever.
            score += max(0, 8 - nearest_any_dist) * 1.5

        # If the opponent is nearly dead, stop preserving shape and force contact.
        if enemy_stack_n <= 2:
            score += max(0, 10 - nearest_any_dist) * 4.0
            score += edge_pressure * 8.0 + corner_pressure * 10.0

    # More closeout aggression when the material lead is clearer.
    return score * (1.0 + max(0.0, advantage) * 1.5 + late_factor)


PROFILES = {
    # The Phalanx Series (less emphasis on hunting, more on solid board presence)
    'phalanx_balanced':   {'material': 0.20, 'safety': 0.20, 'positioning': 0.20, 'alignment': 0.20, 'efficiency': 0.20, 'hunting': 0.00},
    'phalanx_aggressive': {'material': 0.40, 'safety': 0.25, 'positioning': 0.15, 'alignment': 0.15, 'efficiency': 0.10, 'hunting': 0.00},

    # The Hunt Series
    'hunt_aggressive':    {'material': 0.25, 'safety': 0.15, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15},
    'hunt_balanced':      {'material': 0.30, 'safety': 0.20, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15},
    'hunt_defensive':     {'material': 0.30, 'safety': 0.25, 'positioning': 0.20, 'alignment': 0.15, 'efficiency': 0.10, 'hunting': 0.00}
}


def evaluate_board(board: BoardState, root_color: int, is_current_turn_red: bool, profile_name: str) -> float:
    """The core evaluation runner that dynamically applies the requested profile."""
    weights = PROFILES.get(profile_name, PROFILES['phalanx_balanced'])
    is_my_turn = (root_color == 1 and is_current_turn_red) or (root_color == 0 and not is_current_turn_red)

    f_map, e_map = get_stack_maps(board, root_color)

    friendly_tokens = sum(f_map.values())
    enemy_tokens = sum(e_map.values())

    if enemy_tokens == 0:
        return 999999.0
    if friendly_tokens == 0:
        return -999999.0

    total = friendly_tokens + enemy_tokens
    advantage = (friendly_tokens - enemy_tokens) / total if total else 0.0
    late_factor = max(0.0, 1.0 - total / 16.0)
    final_stage = late_factor > 0.45 and advantage > 0.15 and len(e_map) <= 1

    # Only calculate the feature if the weight is greater than 0, saving CPU time!
    total_score = 0.0
    if weights['material'] > 0:    total_score += calc_material(f_map, e_map) * weights['material']
    if weights['positioning'] > 0: total_score += calc_positioning(f_map, e_map) * weights['positioning']
    if weights['efficiency'] > 0:  total_score += calc_efficiency(f_map) * weights['efficiency']
    if weights['alignment'] > 0:   total_score += calc_alignment(f_map) * weights['alignment']
    if weights['safety'] > 0:      total_score += calc_safety(f_map, e_map, is_my_turn) * weights['safety']
    if weights['hunting'] > 0:     total_score += calc_hunting(f_map, e_map) * weights['hunting']

    # New phase-aware features. These are deliberately added outside the profile
    # weights so every movement profile learns how to finish instead of drawing.
    total_score += calc_height_control(f_map, e_map, final_stage) * 0.45

    closeout = calc_closeout_pressure(f_map, e_map)
    if closeout:
        # When we are ahead, safety should not dominate forever. Convert the lead.
        total_score += closeout * (0.9 + max(0.0, advantage) * 1.4)

    return total_score


def eval0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_aggressive')


def eval1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'phalanx_aggressive')


PLACE_PROFILES = {
    # Tries to maximize valid squares for us while minimizing them for the enemy
    'place_territory': {'centrality': 0.20, 'zone_control': 0.50, 'synergy': 0.20},

    # Focuses on creating L-shapes and tight clusters for massive height-6 merges
    'place_phalanx':   {'centrality': 0.30, 'zone_control': 0.10, 'synergy': 0.50},

    # Actively seeks to place distance-2 traps to bait the opponent into bad trades
    'place_trapper':   {'centrality': 0.20, 'zone_control': 0.10, 'synergy': 0.30},

    # A balanced approach for general testing
    'place_balanced':  {'centrality': 0.25, 'zone_control': 0.25, 'synergy': 0.25}
}


def evaluate_place_board(board: BoardState, root_color: int, is_current_turn_red: bool, profile_name: str) -> float:
    """The core placement evaluation runner that dynamically applies the requested profile"""
    weights = PLACE_PROFILES.get(profile_name, PLACE_PROFILES['place_balanced'])
    friendly, enemies, f_controlled, e_controlled = get_place_maps(board, root_color)

    total_score = 0.0

    # Only calculate the feature if the weight is greater than 0
    if weights['centrality'] > 0:
        total_score += calc_place_centrality(friendly) * weights['centrality']

    if weights['zone_control'] > 0:
        total_score += calc_place_zone_control(f_controlled, e_controlled) * weights['zone_control']

    if weights['synergy'] > 0:
        total_score += calc_place_synergy(friendly) * weights['synergy']

    return total_score


def eval_place_co0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_territory')


def eval_place_co1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_trapper')


def eval_place_co2(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_phalanx')
