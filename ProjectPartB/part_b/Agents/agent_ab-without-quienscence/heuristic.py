from Agents.common.board import BoardState
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

# The 4 center cells where we want the L's elbow (merge target)
CENTER_4 = frozenset({27, 28, 35, 36})  # (3,3), (3,4), (4,3), (4,4)

# Distance for each cell to its closest edge
EDGE_DIST = [0] * 64
for _r in range(8):
    for _c in range(8):
        EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)
        
# Precompute Manhattan distance
MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        MANHATTAN[_a][_b] = abs((_a >> 3) - (_b >> 3)) + abs((_a & 7) - (_b & 7))

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
    return evaluate_board(board, root_color, is_red, 'phalanx_aggressive')

# ===============================================================================
# Placing phase evaluation functions
def get_place_maps(board: BoardState, root_color: int) -> tuple[list, list, set, set]:
    """Scans the board once for placing phase features."""
    friendly = []
    enemies = []
    f_controlled = set()
    e_controlled = set()

    for idx in range(64):
        if board.is_empty(idx): continue
        
        is_friendly = (root_color == 1 and board.is_red(idx)) or \
                      (root_color == 0 and board.is_blue(idx))
        
        if is_friendly:
            friendly.append(idx)
            for adj in ADJACENT[idx].values():
                if board.is_empty(adj): f_controlled.add(adj)
        else:
            enemies.append(idx)
            for adj in ADJACENT[idx].values():
                if board.is_empty(adj): e_controlled.add(adj)
                
    return friendly, enemies, f_controlled, e_controlled


def calc_place_centrality(friendly: list) -> float:
    """A simple average of the center weights of our friendly stacks, rewarding central control."""
    if not friendly: 
        return 0.0
    center_raw = sum(CENTER_WEIGHTS[idx] for idx in friendly)
    return center_raw / len(friendly) # likely to between 3.3 - 4.0

def calc_place_zone_control(f_controlled: set, e_controlled: set) -> float:
    # Every empty square adjacent to our stack is a square the enemy CANNOT place on.
    return float(len(f_controlled) - len(e_controlled)) # -4 to +4

def calc_place_synergy(friendly: list) -> float:
    """
    Rewards L-shapes (diagonal distance 2) which allow safe cascade-to-merge setups.
    """
    score = 0.0
    f_len = len(friendly)
    for i in range(f_len):
        for j in range(i + 1, f_len):
            r1, c1 = divmod(friendly[i], 8)
            r2, c2 = divmod(friendly[j], 8)
            
            # The True L-Shape Check:
            # Must be exactly 1 row apart AND 1 column apart (a perfect diagonal)
            if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                score += 15.0
                
    return score


# ==============================================================================
# Coefficient set up for Placing phase profiles
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

# ==============================================================================
# Evaluation functions exposed to the Agent
def eval_place_co0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_territory')

def eval_place_co1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_trapper')

def eval_place_co2(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_place_board(board, root_color, is_red, 'place_phalanx')

# ==============================================================================
# PLACING PHASE 
# ==============================================================================
# def get_place_maps(board: BoardState, root_color: int) -> tuple[list, list]:
#     """Extracts positions of friendly and enemy tokens."""
#     friendly = []
#     enemies = []
#     for idx in range(64):
#         if board.is_empty(idx): continue
#         if (root_color == 1 and board.is_red(idx)) or (root_color == 0 and board.is_blue(idx)):
#             friendly.append(idx)
#         else:
#             enemies.append(idx)
#     return friendly, enemies


# def calc_place_centrality(friendly: list) -> float:
#     """Rewards placing tokens toward the absolute center-4 squares."""
#     score = 0.0
#     for idx in friendly:
#         score += CENTER_WEIGHTS[idx] * 6.0
#     return score


# def get_compact_l_score(stacks: list) -> float:
#     """Helper: Detects and scores the best Compact L-shape in a list of stacks."""
#     if len(stacks) < 3: 
#         return 0.0
        
#     best_score = 0.0
#     stack_set = set(stacks)

#     for elbow in stacks:
#         neighbors = [n for n in ADJACENT[elbow].values() if n in stack_set]
#         if len(neighbors) < 2: 
#             continue

#         for i in range(len(neighbors)):
#             for j in range(i + 1, len(neighbors)):
#                 n1, n2 = neighbors[i], neighbors[j]
#                 d1, d2 = n1 - elbow, n2 - elbow

#                 # Perpendicular check (one horizontal, one vertical offset)
#                 is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or (d1 in (-1, 1) and d2 in (-8, 8))
#                 if not is_perp: 
#                     continue

#                 score = 20.0
#                 center_score = sum(sc for sc in (CENTER_WEIGHTS[elbow] * 6.0, CENTER_WEIGHTS[n1] * 6.0, CENTER_WEIGHTS[n2] * 6.0))

#                 score += center_score

#                 if EDGE_DIST[n1] == 0: score -= CENTER_WEIGHTS[n1] * 6.0
#                 if EDGE_DIST[n2] == 0: score -= CENTER_WEIGHTS[n2] * 6.0

#                 if score > best_score:
#                     best_score = score
                    
#     return best_score


# def calc_place_synergy(friendly: list) -> float:
#     """Phase-aware synergy. Dictates the structure based on current turn count."""
#     f_count = len(friendly)
#     if f_count < 2: 
#         return 0.0

#     # Phase 2: Force the second piece to be a separated blocker
#     if f_count == 2:
#         dist = MANHATTAN[friendly[0]][friendly[1]]
#         if dist == 1: return -20.0  # Heavily penalize early adjacency
#         if dist == 2: return 8.0    # Perfect spacing
#         return 4.0

#     # Phase 3: Build the L-start (one pair adjacent, one piece separated)
#     if f_count == 3:
#         best = -999.0
#         for i in range(3):
#             for j in range(i + 1, 3):
#                 a, b = friendly[i], friendly[j]
#                 if MANHATTAN[a][b] != 1: continue
#                 blocker = friendly[3 - i - j]

#                 score = 0.0
#                 if a in CENTER_4 or b in CENTER_4: score += CENTER_WEIGHTS[27] * 6.0
#                 score += sum(CENTER_WEIGHTS[x] for x in (a, b))

#                 # Ensure blocker is separated from the pair
#                 min_d = min(MANHATTAN[blocker][a], MANHATTAN[blocker][b])
#                 if min_d >= 2: score += 12.0
#                 elif min_d == 1: score -= 5.0

#                 if score > best: best = score
#         return best if best > -999.0 else -20.0

#     # Phase 4: Full Compact L established
#     return get_compact_l_score(friendly)


# def calc_place_denial(board: BoardState, friendly: list, enemies: list) -> float:
#     """Evaluates how effectively the 'Blocker' piece disrupts the enemy."""
#     score = 0.0
#     if not friendly or not enemies: 
#         return score

#     # Blocking Enemy Expansion Zones
#     f_adj = set()
#     for f in friendly:
#         f_adj.update(ADJACENT[f].values())

#     e_adjacent = set()
#     for e in enemies:
#         for nb in ADJACENT[e].values():
#             if board.is_empty(nb):
#                 e_adjacent.add(nb)

#     blocked = len(e_adjacent & f_adj)
#     score += blocked * 6.0

#     # Pushing the Blocker Toward Enemy Centroid
#     cr, cc = 3.5, 3.5
#     e_r = sum(e >> 3 for e in enemies) / len(enemies)
#     e_c = sum(e & 7 for e in enemies) / len(enemies)
#     toward_r, toward_c = e_r - cr, e_c - cc

#     for f in friendly:
#         fr, fc = f >> 3, f & 7
#         dot = (fr - cr) * toward_r + (fc - cc) * toward_c
#         score += dot * 4.0

#     # Anti-Enemy L-Shape
#     if len(enemies) >= 3:
#         score -= get_compact_l_score(enemies) * 0.8

#     return score

# # Coefficient set up for each aspect for different profiles
# PLACE_PROFILES = {
#     'place_L': {'centrality': 0.30, 'synergy': 0.40, 'denial': 0.30},
# }

# def evaluate_place_board(board: BoardState, root_color: int, is_current_turn_red: bool, profile_name: str) -> float:
#     weights = PLACE_PROFILES.get(profile_name, PLACE_PROFILES['place_L'])
#     friendly, enemies = get_place_maps(board, root_color)
    
#     total_score = 0.0
#     if weights['centrality'] > 0:   
#         total_score += calc_place_centrality(friendly) * weights['centrality']
#     if weights['synergy'] > 0:      
#         total_score += calc_place_synergy(friendly) * weights['synergy']
#     if weights['denial'] > 0:   
#         total_score += calc_place_denial(board, friendly, enemies) * weights['denial']

#     # Material difference (Universal base score)
#     total_score += (len(friendly) - len(enemies)) * 15.0

#     return total_score


# def eval_L_place(board: BoardState, root_color: int, is_red: bool) -> float:
#     return evaluate_place_board(board, root_color, is_red, 'place_L')
    