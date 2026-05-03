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
        if h == 1: 
            score -= 10.0
        elif 2 <= h <= 3: 
            score += 5.0
        elif 4 <= h <= 6: 
            score += 15.0
        elif h == 7: 
            score -= 15.0
        elif h > 7: 
            score -= (30.0 + ((h - 7) * 10.0))
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

    # Average weight is between 1 (edges) and 4 (centre).
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
# L-SHAPE CENTRAL ANCHOR PLACEMENT STRATEGY  (v2 — compact L + blocking)
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

# Precompute Manhattan distance
_MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        _MANHATTAN[_a][_b] = abs((_a >> 3) - (_b >> 3)) + abs((_a & 7) - (_b & 7))


def _detect_compact_L(stacks):
    """
    Find the best COMPACT L among the given stack positions.

    A compact L is exactly: elbow + 1 adjacent stack in one cardinal
    direction + 1 adjacent stack in a perpendicular direction.
    That's 3 stacks forming a right-angle, all mutually within distance 2.

    Returns (score, elbow_idx, [elbow, arm1, arm2]).
    """
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
                d1 = n1 - elbow   # direction offset to n1
                d2 = n2 - elbow   # direction offset to n2

                # Must be perpendicular: one horizontal, one vertical
                is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or \
                          (d1 in (-1, 1) and d2 in (-8, 8))
                if not is_perp:
                    continue

                score = 40.0    # base compact-L bonus

                # ── Reward ANY of the 3 stacks being on center-4 ──
                # The key insight: whether the elbow or an arm tip sits
                # on center-4, the L is strategically valuable because
                # we can merge onto that center cell.
                center_count = sum(1 for s in (elbow, n1, n2)
                                   if s in _CENTER_4)
                if center_count >= 2:
                    score += 40.0   # two stacks on center-4 = amazing
                elif center_count == 1:
                    score += 28.0   # one stack on center-4 = great
                else:
                    # No center-4, score by average proximity
                    score += max(0, 15.0 - _DIST_TO_CENTER[elbow] * 5.0)

                # ── All 3 stacks near center (average proximity) ──
                avg_cd = (_DIST_TO_CENTER[elbow] +
                          _DIST_TO_CENTER[n1] +
                          _DIST_TO_CENTER[n2]) / 3.0
                score += max(0, 12.0 - avg_cd * 3.0)

                # ── Penalize if any arm tip is on the edge ──
                if EDGE_DIST[n1] == 0:
                    score -= 12.0
                if EDGE_DIST[n2] == 0:
                    score -= 12.0

                if score > best_score:
                    best_score = score
                    best_elbow = elbow
                    best_L = [elbow, n1, n2]

    return best_score, best_elbow, best_L


def eval_place_L(board: BoardState, root_color: int, is_red: bool) -> float:
    """
    Placement-phase evaluation: compact center L + enemy blocking + merge
    readiness.

    Desired placement sequence (e.g. as Red):
      Move 1: Grab a center-4 cell.
      Move 2: Place adjacent to our center stack, in a direction that
              also blocks the opponent from expanding toward our side.
      Move 3: Complete the compact 3-stack L (elbow + 2 perpendicular
              adjacent stacks).
      Move 4: Place near the L for post-placement cascade/merge into h4+.

    Scoring:
      1. Material       — token count parity
      2. Centrality     — proximity to center-4 cells
      3. Adjacency      — connected friendly pairs (building the L)
      4. Compact L      — exact 3-stack right-angle detection
      5. Blocking       — restrict opponent's useful placement squares
      6. 4th-stack      — merge readiness (near L for cascade/move)
      7. Anti-enemy-L   — penalize if opponent is forming a good L
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
    f_set = set(friendly)

    score = 0.0

    # ── 1. MATERIAL ─────────────────────────────────────
    score += (f_count - e_count) * 15.0

    # ── 2. CENTRALITY ───────────────────────────────────
    for idx in friendly:
        d = _DIST_TO_CENTER[idx]
        score += (6 - d) * 6.0
        if idx in _CENTER_4:
            score += 18.0
        if EDGE_DIST[idx] == 0:
            score -= 12.0

    # ── 3. ADJACENCY (friendly connectivity) ────────────
    for i in range(f_count):
        for j in range(i + 1, f_count):
            dist = _MANHATTAN[friendly[i]][friendly[j]]
            if dist == 1:
                score += 10.0
            elif dist == 2:
                # Only reward collinear (same row or col) — useful for L
                if (friendly[i] >> 3) == (friendly[j] >> 3) or \
                   (friendly[i] & 7) == (friendly[j] & 7):
                    score += 3.0

    # ── 4. COMPACT L DETECTION ──────────────────────────
    l_score, l_elbow, l_stacks = _detect_compact_L(friendly)
    score += l_score

    # ── 5. BLOCKING ─────────────────────────────────────
    # Our stacks create a "threat zone": cells adjacent to us where
    # the opponent CANNOT place.  We want this zone to overlap with
    # cells the opponent would like to place at (adjacent to their
    # stacks).
    if friendly and enemy:
        # Cells adjacent to any of our stacks
        our_threat_zone = set()
        for f_idx in friendly:
            our_threat_zone.update(_ADJ_SET[f_idx])

        # Cells adjacent to enemy stacks that are empty — places the
        # enemy wants to expand into
        enemy_wants = set()
        for e_idx in enemy:
            for nb in _ADJ_SET[e_idx]:
                if board.is_empty(nb):
                    enemy_wants.add(nb)

        # Blocked = enemy wants these but our adjacency prevents it
        blocked = enemy_wants & our_threat_zone
        score += len(blocked) * 8.0

    # ── 6. 4TH-STACK PLACEMENT ───────────────────────────
    # Once 3 stacks form the L, the 4th must NOT just extend an arm
    # (useless line).  Instead it should be placed to block enemy
    # expansion or in a separate strategic zone for post-placement
    # threats.
    if l_stacks and f_count >= 4:
        l_set = set(l_stacks)
        elbow = l_stacks[0]
        arm1, arm2 = l_stacks[1], l_stacks[2]

        # L centroid for spread calculation
        l_avg_r = sum(s >> 3 for s in l_stacks) / 3.0
        l_avg_c = sum(s & 7 for s in l_stacks) / 3.0

        for idx in friendly:
            if idx in l_set:
                continue

            # ── Detect arm extension (BAD) ──
            # Two cases of useless line extension:
            # A) 4th adjacent to arm tip, same direction as elbow→arm
            #    (extends the arm outward)
            # B) 4th adjacent to elbow, OPPOSITE direction from an arm
            #    (extends the line through the elbow backward)
            is_extending = False
            for arm in (arm1, arm2):
                d_elbow_to_arm = arm - elbow
                # Case A: extends past arm tip
                if _MANHATTAN[idx][arm] == 1:
                    if (idx - arm) == d_elbow_to_arm:
                        is_extending = True
                        break
                # Case B: extends past elbow in opposite direction
                if _MANHATTAN[idx][elbow] == 1:
                    if (idx - elbow) == -d_elbow_to_arm:
                        is_extending = True
                        break

            if is_extending:
                score -= 35.0   # strong penalty: don't extend arms
            else:
                # ── Reward useful 4th positions ──
                # Near center for post-placement threats
                score += max(0, (4 - _DIST_TO_CENTER[idx]) * 5.0)

                # Spread bonus: 4th stack at distance 2-4 from L
                # centroid is ideal (strategic coverage, not clumped)
                ir, ic = idx >> 3, idx & 7
                spread = abs(ir - l_avg_r) + abs(ic - l_avg_c)
                if 2.0 <= spread <= 4.0:
                    score += 10.0
                elif spread > 4.0:
                    score += 3.0  # too far, less useful

                # Within cascade/move reach of L (dist 2-3 is ideal)
                min_d = min(_MANHATTAN[idx][l_idx] for l_idx in l_stacks)
                if min_d == 1:
                    # Adjacent to L but NOT extending — e.g. on the
                    # inside of the L or perpendicular to an arm
                    score += 8.0
                elif min_d <= 3:
                    score += 5.0

    # ── 7. ANTI-ENEMY-L ────────────────────────────────
    if e_count >= 3:
        enemy_l, _, _ = _detect_compact_L(enemy)
        score -= enemy_l * 0.5

    return score
    