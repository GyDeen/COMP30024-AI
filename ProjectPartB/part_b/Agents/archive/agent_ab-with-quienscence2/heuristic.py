"""Agent-specific evaluation profiles and strategy wrappers."""

from Agents.common.board import BoardState
from Agents.common.features.geometry import (
    ADJACENT, CENTER_4, CENTER_CELLS, CENTER_WEIGHTS, CORNERS, EDGE_DIST, MANHATTAN,
)
from Agents.common.features.movement import (
    get_stack_maps,
    calc_material,
    calc_positioning,
    calc_alignment,
    calc_control,
)
from Agents.common.features.movement_variants import (
    calc_efficiency_with_single_penalty as calc_efficiency,
    calc_safety_all_heights as calc_safety,
    calc_hunting_unweighted_fear as calc_hunting,
)
from Agents.common.features.closeout import (
    _dist_to_center,
    _dist_to_corner,
    _is_attacked_by,
    _safe_escape_count_for_enemy,
    calc_closeout,
    calc_max_height_control,
)
from Agents.common.features.placement import (
    extract_placement_features as eval_place0,
    get_place_maps,
    calc_place_centrality,
    get_compact_l_score,
    calc_place_synergy,
    calc_place_denial,
)


RED = 1
BLUE = 0


PROFILES = {
    # The Phalanx Series (less emphasis on hunting, more on solid board presence)
    'phalanx_balanced':   {'material': 0.40, 'safety': 0.20, 'positioning': 0.15, 'alignment': 0.00, 'efficiency': 0.30, 'hunting': 0.10, 'control': 0.10, 'closeout': 0.55, 'max_height': 1.00},
    'phalanx_aggressive': {'material': 0.40, 'safety': 0.20, 'positioning': 0.12, 'alignment': 0.00, 'efficiency': 0.08, 'hunting': 0.18, 'control': 0.20, 'closeout': 0.90, 'max_height': 1.00},

    # The Hunt Series
    'hunt_aggressive':    {'material': 0.25, 'safety': 0.15, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15, 'control': 0.10, 'closeout': 0.80, 'max_height': 1.00},
    'hunt_balanced':      {'material': 0.30, 'safety': 0.18, 'positioning': 0.10, 'alignment': 0.15, 'efficiency': 0.12, 'hunting': 0.18, 'control': 0.18, 'closeout': 0.85, 'max_height': 1.00},
    'hunt_defensive':     {'material': 0.30, 'safety': 0.25, 'positioning': 0.20, 'alignment': 0.15, 'efficiency': 0.10, 'hunting': 0.00, 'control': 0.10, 'closeout': 0.35, 'max_height': 1.00},


    # The endgame profile, focused on eliminating enemy pieces and securing the win
    'endgame':            {'material': 0.45, 'safety': 0.18, 'positioning': 0.03, 'alignment': 0.00, 'efficiency': 0.06, 'hunting': 0.25, 'control': 0.25, 'closeout': 1.20, 'max_height': 1.00},
}


def evaluate_board(board: BoardState, root_color: int, is_current_turn_red: bool, profile_name: str) -> float:
    """The core evaluation runner that dynamically applies the requested profile."""
    weights = PROFILES.get(profile_name, PROFILES['phalanx_balanced'])
    is_my_turn = (root_color == 1 and is_current_turn_red) or (root_color == 0 and not is_current_turn_red)

    f_map, e_map = get_stack_maps(board, root_color)

    # Terminal states must dominate everything else.
    if not f_map:
        return -999999.0
    if not e_map:
        return 999999.0

    # Only calculate the feature if the weight is greater than 0, saving CPU time.
    total_score = 0.0
    if weights['material'] > 0:    total_score += calc_material(f_map, e_map) * weights['material']
    if weights['positioning'] > 0: total_score += calc_positioning(f_map, e_map) * weights['positioning']
    if weights['efficiency'] > 0:  total_score += calc_efficiency(f_map, e_map) * weights['efficiency']
    if weights['alignment'] > 0:   total_score += calc_alignment(f_map) * weights['alignment']
    if weights['safety'] > 0:      total_score += calc_safety(f_map, e_map, is_my_turn) * weights['safety']
    if weights['hunting'] > 0:     total_score += calc_hunting(f_map, e_map) * weights['hunting']
    if weights['control'] > 0:     total_score += calc_control(f_map, e_map) * weights['control']
    if weights.get('closeout', 0) > 0:
        total_score += calc_closeout(f_map, e_map) * weights['closeout']
    if weights.get('max_height', 0) > 0:
        total_score += calc_max_height_control(f_map, e_map) * weights['max_height']

    return total_score


def eval0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'phalanx_aggressive')


def eval1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_balanced')


def end_phase_eval(board: BoardState, root_color: int, is_red: bool) -> float:
    """Evaluation for the end phase, focused on final positioning and material."""
    return evaluate_board(board, root_color, is_red, 'endgame')


def eval_place_co0(board: BoardState, root_color: int, is_red: bool) -> float:
    """Defensive evaluation for placing phase, focused on building a strong fortress."""
    mat, cen, sup, agg = eval_place0(board, root_color)

    # Calculate weighted final score
    score = (mat * 0.40) + (cen * 0.40) + (sup * 0.20) + (agg * 0.00)
    return score


def eval_place_co1(board: BoardState, root_color: int, is_red: bool) -> float:
    """Aggressive evaluation for placing phase, focused on threatening the opponent."""
    mat, cen, sup, agg = eval_place0(board, root_color)

    # Calculate weighted final score
    score = (mat * 0.40) + (cen * 0.10) + (sup * 0.00) + (agg * 0.50)
    return score


def eval_place_co2(board: BoardState, root_color: int, is_red: bool) -> float:
    """Balanced evaluation for placing phase, with moderate emphasis on both defense and aggression."""
    mat, cen, sup, agg = eval_place0(board, root_color)

    # Calculate weighted final score
    score = (mat * 0.40) + (cen * 0.25) + (sup * 0.20) + (agg * 0.15)
    return score


PLACE_PROFILES = {
    'place_L': {'centrality': 0.30, 'synergy': 0.40, 'denial': 0.30},
}


def evaluate_place_board(board: BoardState, root_color: int, is_current_turn_red: bool, profile_name: str) -> float:
    weights = PLACE_PROFILES.get(profile_name, PLACE_PROFILES['place_L'])
    friendly, enemies = get_place_maps(board, root_color)

    total_score = 0.0
    if weights['centrality'] > 0:
        total_score += calc_place_centrality(friendly) * weights['centrality']
    if weights['synergy'] > 0:
        total_score += calc_place_synergy(friendly) * weights['synergy']
    if weights['denial'] > 0:
        total_score += calc_place_denial(board, friendly, enemies) * weights['denial']

    # Material difference (Universal base score)
    total_score += (len(friendly) - len(enemies)) * 15.0

    return total_score


def eval_L_place(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_L')
