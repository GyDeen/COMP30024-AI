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
    calc_efficiency_friendly_only as calc_efficiency,
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
    """The core evaluation runner that dynamically applies the requested profile"""
    weights = PROFILES.get(profile_name, PROFILES['phalanx_balanced'])
    is_my_turn = (root_color == 1 and is_current_turn_red) or (root_color == 0 and not is_current_turn_red)

    f_map, e_map = get_stack_maps(board, root_color)

    # Only calculate the feature if the weight is greater than 0, saving CPU time!
    total_score = 0.0
    if weights['material'] > 0:    total_score += calc_material(f_map, e_map) * weights['material']
    if weights['positioning'] > 0: total_score += calc_positioning(f_map, e_map) * weights['positioning']
    if weights['efficiency'] > 0:  total_score += calc_efficiency(f_map) * weights['efficiency']
    if weights['alignment'] > 0:   total_score += calc_alignment(f_map) * weights['alignment']
    if weights['safety'] > 0:      total_score += calc_safety(f_map, e_map, is_my_turn) * weights['safety']
    if weights['hunting'] > 0:     total_score += calc_hunting(f_map, e_map) * weights['hunting']

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
