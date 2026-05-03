from .board import BoardState
from .search import ADJACENT

RED = 1
BLUE = 0


#==============================================================================
# Center control weights
CENTER_WEIGHTS = [
    -2, -2, -2, -2, -2, -2, -2, -2,
    -2, 2, 2, 2, 2, 2, 2, -2,
    -2, 2, 3, 3, 3, 3, 2, -2,
    -2, 2, 3, 4, 4, 3, 2, -2,
    -2, 2, 3, 4, 4, 3, 2, -2,
    -2, 2, 3, 3, 3, 3, 2, -2,
    -2, 2, 2, 2, 2, 2, 2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2
]

# Distance for each cell to its closest edge
EDGE_DIST = [0] * 64
for _r in range(8):
    for _c in range(8):
        EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)
# ==============================================================================

# ==============================================================================
# Extract the friends and enemies into hash maps
def get_stack_maps(board: BoardState, root_color: int) -> tuple[dict, dict]:
    """Scans the board once and returns dictionaries of {index: height} for fast lookup."""
    friendly_stacks = {}
    enemy_stacks = {}
    
    for cell_index in range(64):
        if board.is_empty(cell_index): continue
        
        h = board.get_height(cell_index)
        is_friendly = (root_color == 1 and board.is_red(cell_index)) or \
                      (root_color == 0 and board.is_blue(cell_index))
                      
        if is_friendly:
            friendly_stacks[cell_index] = h
        else:
            enemy_stacks[cell_index] = h
            
    return friendly_stacks, enemy_stacks
# ==============================================================================

# ==============================================================================
# Calculate each aspect's score 
def calc_material(friendly_map: dict, enemy_map: dict) -> float:
    f_score = sum(h * 10.0 for h in friendly_map.values())
    e_score = sum(h * 10.0 for h in enemy_map.values())
    return f_score - e_score

def calc_positioning(friendly_map: dict, enemy_map: dict) -> float:
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

def calc_efficiency(friendly_map: dict) -> float:
    score = 0.0
    for h in friendly_map.values():
        if h == 1: score -= 10.0
        elif 2 <= h <= 3: score += 5.0
        elif 4 <= h <= 6: score += 15.0
        elif h == 7: score -= 15.0
        elif h > 7: score -= (30.0 + ((h - 7) * 10.0))
    return score

def calc_alignment(friendly_map: dict) -> float:
    score = 0.0
    indices = list(friendly_map.keys())
    for i in range(len(indices)):
        idx_1 = indices[i]
        r1, c1 = divmod(idx_1, 8)
        h1 = friendly_map[idx_1]
        
        for j in range(i + 1, len(indices)):
            idx_2 = indices[j]
            r2, c2 = divmod(idx_2, 8)
            h2 = friendly_map[idx_2]
            
            if r1 == r2 or c1 == c2:
                dist = abs(r1 - r2) + abs(c1 - c2)
                if (h1 - dist > EDGE_DIST[idx_2]) or (h2 - dist > EDGE_DIST[idx_1]): 
                    score += 5.0
    return score

def calc_safety(friendly_map: dict, enemy_map: dict, is_my_turn: bool) -> float:
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

def calc_hunting(friendly_map: dict, enemy_map: dict) -> float:
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
# ==============================================================================

# ==============================================================================
# Coefficient set up for each aspect for different profiles
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
# ==============================================================================

# ==============================================================================
# Evaluation functions that being tested
def eval0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_aggressive')

def eval1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_balanced')

# ===============================================================================

"""Placing phase evaluation functions"""

def eval_place0(board: BoardState, root_color: int) -> tuple[float, float, float, float]:
    """Returns (Material, Centrality, Support, Aggression) all scaled from 0 to 100."""
    friendly = []
    enemies = []
    center_raw = 0.0

    for idx in range(64):
        if board.is_empty(idx): continue
        if (root_color and board.is_red(idx)) or (not root_color and board.is_blue(idx)):
            friendly.append(idx)
            center_raw += CENTER_WEIGHTS[idx]
        else:
            enemies.append(idx)

    f_len = len(friendly)
    e_len = len(enemies)

    # Max difference in placement is 4. Scale from -4 to +4 into 0 to 100.
    material = ((f_len - e_len + 4.0) / 8.0) * 100.0

    # Average weight is between 1 (edges) and 4 (center).
    if f_len > 0:
        avg_center = center_raw / f_len
        centrality = ((avg_center - 1.0) / 3.0) * 100.0
    else:
        centrality = 0.0

    # Max theoretical pairs for 4 tokens is 6.
    support_pairs = 0
    for i in range(f_len):
        for j in range(i + 1, f_len):
            r1, c1 = divmod(friendly[i], 8)
            r2, c2 = divmod(friendly[j], 8)
            dist = abs(r1 - r2) + abs(c1 - c2)
            if 1 <= dist <= 3:
                support_pairs += 1
    support = (min(support_pairs, 6) / 6.0) * 100.0

    # How many of our tokens are exactly 2 steps from an enemy?
    threatening = 0
    for f_idx in friendly:
        fr, fc = divmod(f_idx, 8)
        for e_idx in enemies:
            er, ec = divmod(e_idx, 8)
            dist = abs(fr - er) + abs(fc - ec)
            if dist == 2:
                threatening += 1
                break # Count each friendly token max once
    aggression = (min(threatening, 4) / 4.0) * 100.0

    return material, centrality, support, aggression


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


# ==============================================================================
# L-SHAPE CENTRAL ANCHOR PLACEMENT STRATEGY
# ==============================================================================
# The 4 center cells where we want the L's elbow (merge target)
_CENTER_4 = frozenset({27, 28, 35, 36})  # (3,3), (3,4), (4,3), (4,4)

# Manhattan distance from each cell to its nearest center-4 cell
_DIST_TO_CENTER = [0] * 64
for _i in range(64):
    _r, _c = divmod(_i, 8)
    _DIST_TO_CENTER[_i] = min(
        abs(_r - 3) + abs(_c - 3),
        abs(_r - 3) + abs(_c - 4),
        abs(_r - 4) + abs(_c - 3),
        abs(_r - 4) + abs(_c - 4),
    )

# Pre-compute adjacency as sets for fast lookup
_ADJ_SET = [set() for _ in range(64)]
for _i in range(64):
    _r, _c = divmod(_i, 8)
    if _r > 0: _ADJ_SET[_i].add(_i - 8)
    if _r < 7: _ADJ_SET[_i].add(_i + 8)
    if _c > 0: _ADJ_SET[_i].add(_i - 1)
    if _c < 7: _ADJ_SET[_i].add(_i + 1)


def _detect_best_L(stacks):
    """
    Given a list of friendly stack indices, find the best L-shape pattern
    and return (score, elbow_idx).

    An L-shape has an elbow with two perpendicular arms:
      - Short arm: 1 stack adjacent to elbow in one cardinal direction
      - Long arm:  1-2 stacks extending from elbow in a perpendicular direction

    The elbow is where the short arm will MOVE onto, creating h6 (=h3+h3).
    """
    if len(stacks) < 3:
        return 0.0, -1

    stack_set = set(stacks)
    best_score = 0.0
    best_elbow = -1

    # Try each stack as the elbow
    for elbow in stacks:
        er, ec = divmod(elbow, 8)
        neighbors_in_set = _ADJ_SET[elbow] & stack_set

        if len(neighbors_in_set) < 2:
            continue  # Need at least 2 neighbors to form an L

        # Classify neighbors by direction from elbow
        arms = {}  # direction -> list of stack indices along that arm
        for direction in (-8, 8, -1, 1):  # UP, DOWN, LEFT, RIGHT
            arm = []
            pos = elbow
            while True:
                # Boundary check before stepping
                pr, pc = divmod(pos, 8)
                next_pos = pos + direction
                if next_pos < 0 or next_pos >= 64:
                    break
                nr, nc = divmod(next_pos, 8)
                # Prevent horizontal wrap across rows
                if direction in (-1, 1) and nr != pr:
                    break
                pos = next_pos
                if pos in stack_set:
                    arm.append(pos)
                else:
                    break
            if arm:
                arms[direction] = arm

        if len(arms) < 2:
            continue

        # Try all pairs of perpendicular arms
        horiz = [d for d in arms if d in (-1, 1)]
        vert  = [d for d in arms if d in (-8, 8)]

        for h_dir in horiz:
            for v_dir in vert:
                h_arm = arms[h_dir]
                v_arm = arms[v_dir]

                # Short arm = shorter, long arm = longer
                if len(h_arm) <= len(v_arm):
                    short_arm, long_arm = h_arm, v_arm
                else:
                    short_arm, long_arm = v_arm, h_arm

                short_len = len(short_arm)
                long_len = len(long_arm)

                # We want at least 1 in each arm
                if short_len < 1 or long_len < 1:
                    continue

                # Total stacks used in this L (including elbow)
                l_size = 1 + short_len + long_len

                # Score this L configuration
                score = 0.0

                # Base L-shape bonus scaled by how many stacks participate
                score += l_size * 8.0

                # Elbow at center-4 is ideal (merge target becomes central anchor)
                if elbow in _CENTER_4:
                    score += 30.0
                else:
                    score += max(0, 12.0 - _DIST_TO_CENTER[elbow] * 4.0)

                # Short arm length = 1 is ideal (one MOVE creates h6 at elbow)
                if short_len == 1:
                    score += 15.0
                elif short_len == 2:
                    score += 8.0  # still OK, two moves to consolidate

                # Long arm length 2 is ideal (good material depth)
                if long_len >= 2:
                    score += 10.0

                # Bonus: the entire L is close to center
                all_L = [elbow] + short_arm + long_arm
                avg_center_dist = sum(_DIST_TO_CENTER[s] for s in all_L) / len(all_L)
                score += max(0, 8.0 - avg_center_dist * 2.5)

                if score > best_score:
                    best_score = score
                    best_elbow = elbow

    return best_score, best_elbow


def eval_place_L(board: BoardState, root_color: int, is_red: bool) -> float:
    """
    Placement-phase evaluation that rewards L-shape formation near center.

    Strategy: build a connected L of friendly stacks with the elbow on a
    center-4 cell.  After placement ends, MOVE the short arm onto the
    elbow → h6 central anchor that dominates the board.

    Scoring components:
      1. Centrality   – each stack scored by proximity to center-4
      2. Connectivity – adjacent friendly pairs (needed for L)
      3. L-shape      – explicit L-pattern detection with elbow bonus
      4. Material     – token count parity
      5. Enemy buffer – mild penalty for being too close to enemies
    """
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

    score = 0.0

    # ── 1. MATERIAL (token parity) ──────────────────────
    score += (f_count - e_count) * 15.0

    # ── 2. CENTRALITY ───────────────────────────────────
    # Strong pull toward center.  Center-4 cells get extra bonus.
    for idx in friendly:
        d = _DIST_TO_CENTER[idx]
        score += (6 - d) * 5.0          # closer = better
        if idx in _CENTER_4:
            score += 12.0               # sitting on THE center
        if EDGE_DIST[idx] == 0:
            score -= 8.0                # edge is terrible for anchor

    # ── 3. CONNECTIVITY ─────────────────────────────────
    # Adjacent friendly stacks are needed to form the L and to merge.
    adj_count = 0
    for i in range(f_count):
        for j in range(i + 1, f_count):
            r1, c1 = divmod(friendly[i], 8)
            r2, c2 = divmod(friendly[j], 8)
            dist = abs(r1 - r2) + abs(c1 - c2)
            if dist == 1:
                adj_count += 1
                score += 10.0
            elif dist == 2 and (r1 == r2 or c1 == c2):
                # Collinear with 1 gap — still usable but weaker
                score += 3.0

    # ── 4. L-SHAPE DETECTION ────────────────────────────
    l_score, l_elbow = _detect_best_L(friendly)
    score += l_score

    # ── 5. ENEMY BUFFER ─────────────────────────────────
    # During placement we want some distance from enemies so we can
    # build our L undisturbed, but not too far (we want to threaten
    # soon after placement ends).
    for f_idx in friendly:
        fr, fc = divmod(f_idx, 8)
        for e_idx in enemy:
            er, ec = divmod(e_idx, 8)
            dist = abs(fr - er) + abs(fc - ec)
            if dist == 1:
                score -= 6.0   # right next to enemy — awkward
            elif dist == 2:
                score += 2.0   # threatening distance — good
            elif dist == 3:
                score += 1.0   # decent striking range

    return score
    