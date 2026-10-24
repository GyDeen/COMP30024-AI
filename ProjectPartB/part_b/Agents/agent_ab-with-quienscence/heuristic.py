from Agents.common.board import BoardState
from .search import ADJACENT

RED = 1
BLUE = 0


#==============================================================================
# The 4 center cells where we want the L's elbow (merge target)
CENTER_4 = frozenset({27, 28, 35, 36})  # (3,3), (3,4), (4,3), (4,4)


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
    # Friendly positioning 
    for idx, h in friendly_map.items():
        weight = CENTER_WEIGHTS[idx]
        score += (weight * h / 2.0)
        if 4 <= h <= 5 and weight >= 3:
            score += (h * weight * 1.5)
            
    # Enemy positioning 
    for idx, h in enemy_map.items():
        weight = CENTER_WEIGHTS[idx]
        score -= (weight * h / 2.0)
        if 4 <= h <= 5 and weight >= 3:
            score -= (h * weight * 1.5)
    return score

def calc_efficiency(friendly_map: dict, enemy_map: dict) -> float:
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

def calc_threat(friendly_map: dict, enemy_map: dict, is_my_turn: bool) -> float:
    """Rewards the AI for actively projecting lethal threats (EAT or CASCADE) onto enemy stacks."""
    score = 0.0
    for f_idx, f_h in friendly_map.items():
        fr, fc = divmod(f_idx, 8)
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Threat A: Direct EAT
            adj_r, adj_c = fr + dr, fc + dc
            if 0 <= adj_r < 8 and 0 <= adj_c < 8:
                adj_idx = adj_r * 8 + adj_c
                if adj_idx in enemy_map and f_h >= enemy_map[adj_idx]:
                    target_h = enemy_map[adj_idx]
                    
                    # If it is our turn, this is an immediate kill
                    # If it is their turn, this forces them to waste a turn running away 
                    score += (target_h * 10.0) if is_my_turn else (target_h * 2.0)

            # Threat B: CASCADE Push
            if f_h >= 2:
                pieces_in_line = 0 
                for step in range(1, f_h + 1):
                    tr, tc = fr + (dr * step), fc + (dc * step)

                    # Calculate how many of OUR tokens we lose if we cascade
                    our_tokens_lost = 0
                    if not (0 <= tr < 8 and 0 <= tc < 8):
                        our_tokens_lost = (f_h + 1) - step
                        break # The rest of the cascade falls off the board
                        
                    t_idx = tr * 8 + tc
                    if t_idx in friendly_map or t_idx in enemy_map:
                        pieces_in_line += 1
                        push_r, push_c = fr + (dr * (f_h + pieces_in_line)), fc + (dc * (f_h + pieces_in_line))
                        
                        # Check if the enemy gets pushed off
                        if not (0 <= push_r < 8 and 0 <= push_c < 8):
                            if t_idx in enemy_map:
                                target_h = enemy_map[t_idx]
                                
                                # THE TRADE MATH
                                # We only consider this a valid threat if we win the material trade!
                                if target_h > our_tokens_lost:
                                    net_gain = target_h - our_tokens_lost
                                    score += (net_gain * 10.0) if is_my_turn else (net_gain * 2.0)
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
            if e_h < 2:
                break
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

def calc_hunting(friendly_map: dict, enemy_map: dict) -> float:
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
            
            # We can eat it, try to go adjacent with it
            if size_diff > 0:
                effective_diff = min(size_diff, 1)
                # Scaled down to prevent out-weighing the Material score
                hunt_val = ((MAX_DIST - dist) / 5.0) * effective_diff
                if hunt_val > best_hunt:
                    best_hunt = hunt_val
                    
            # If we are the same size, we can restrict it by taking the diagonal position, which locks it avaialbale move
            elif size_diff == 0:
                if dist == 2:
                    dx, dy = abs(f_row - e_row), abs(f_col - e_col)
                    if dx == 1 and dy == 1:
                        # Scaled down hard diagonal pin to 
                        hunt_val = ((MAX_DIST - dist) / 5.0) * 1.5
                        if hunt_val > best_hunt:
                            best_hunt = hunt_val
                            
            # We are smaller, try to get away from it
            elif size_diff < 0:
                # Height one friendly piece isn't carrying too many value, just pass 
                if f_h > 1: 
                    # The closer the big enemy is, the higher the fear value
                    # The higher the friendly piece, the more we care about it, but we cap the fear to prevent it from out-weighing the Material score
                    fear_val = ((MAX_DIST - dist) * f_h / 5.0) 
                    if fear_val > worst_fear:
                        worst_fear = fear_val
                        
        # Add the desire to chase, subtract the desire to flee
        score += best_hunt
        score -= worst_fear
        
    return score

def calc_control(friendly_map: dict, enemy_map: dict) -> float:
    best_anchor_score = 0.0
    
    for f_idx, f_h in friendly_map.items():
        if f_h < 2: continue 
        
        fr, fc = divmod(f_idx, 8)
        current_anchor_score = 0.0
        
        for e_idx, e_h in enemy_map.items():
            valid_moves = 0
            denied_moves = 0
            
            for adj_idx in ADJACENT[e_idx].values():
                if adj_idx in friendly_map: continue 
                    
                valid_moves += 1
                is_denied = False
                adj_r, adj_c = divmod(adj_idx, 8)
                
                # Simulating the push to see whether it can push off any other piece that on the same row or column
                if fr == adj_r:
                    if fc < adj_c <= fc + f_h:  # Right
                        N = 1  
                        for c in range(fc + 1, adj_c):
                            if fr * 8 + c in friendly_map or fr * 8 + c in enemy_map: N += 1
                        if fc + f_h + N >= 8: is_denied = True 
                            
                    elif fc > adj_c >= fc - f_h:  # Left
                        N = 1
                        for c in range(fc - 1, adj_c, -1):
                            if fr * 8 + c in friendly_map or fr * 8 + c in enemy_map: N += 1
                        if fc - f_h - N < 0: is_denied = True
                            
                elif fc == adj_c:
                    if fr < adj_r <= fr + f_h:  # Down
                        N = 1
                        for r in range(fr + 1, adj_r):
                            if r * 8 + fc in friendly_map or r * 8 + fc in enemy_map: N += 1
                        if fr + f_h + N >= 8: is_denied = True
                            
                    elif fr > adj_r >= fr - f_h:  # Up
                        N = 1
                        for r in range(fr - 1, adj_r, -1):
                            if r * 8 + fc in friendly_map or r * 8 + fc in enemy_map: N += 1
                        if fr - f_h - N < 0: is_denied = True
                
                # Adjacent EAT Threat Check
                if not is_denied:
                    for f_adj_idx in ADJACENT[adj_idx].values():
                        if f_adj_idx in friendly_map and friendly_map[f_adj_idx] >= e_h:
                            is_denied = True
                            break
                                
                if is_denied:
                    denied_moves += 1
                        
            # The Restriction Reward
            if valid_moves > 0:
                # Reward scales based on the height of the enemy being squeezed
                current_anchor_score += (denied_moves * e_h * 1.0)
                if denied_moves == valid_moves:
                    # Enemy has to move into a position that will be eliminated, reward it based on the enemy height will be eliminated
                    current_anchor_score += (e_h * 5.0)

        # Keep only the highest single anchor score
        if current_anchor_score > best_anchor_score:
            best_anchor_score = current_anchor_score

    return best_anchor_score
# ==============================================================================

# ==============================================================================
# Coefficient set up for each aspect for different profiles
PROFILES = {
    # The Phalanx Series (less emphasis on hunting, more on solid board presence)
    'phalanx_balanced':   {'material': 0.40, 'threaten_enemy': 0.10, 'safety': 0.20, 'positioning': 0.15, 'alignment': 0.00, 'efficiency': 0.30, 'hunting': 0.10, 'control': 0.10},
    'phalanx_aggressive': {'material': 0.40, 'threaten_enemy': 0.20, 'safety': 0.10, 'positioning': 0.15, 'alignment': 0.00, 'efficiency': 0.10, 'hunting': 0.15, 'control': 0.15},

    # The Hunt Series
    'hunt_aggressive':    {'material': 0.25, 'threaten_enemy': 0.20, 'safety': 0.15, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15, 'control': 0.10},
    'hunt_balanced':      {'material': 0.30, 'threaten_enemy': 0.10, 'safety': 0.20, 'positioning': 0.10, 'alignment': 0.20, 'efficiency': 0.15, 'hunting': 0.15, 'control': 0.10},
    'hunt_defensive':     {'material': 0.30, 'threaten_enemy': 0.05, 'safety': 0.25, 'positioning': 0.20, 'alignment': 0.15, 'efficiency': 0.10, 'hunting': 0.00, 'control': 0.10},


    # The endgame profile, focused on eliminating enemy pieces and securing the win
    'endgame':            {'material': 0.50, 'threaten_enemy': 0.00, 'safety': 0.15, 'positioning': 0.05, 'alignment': 0.00, 'efficiency': 0.10, 'hunting': 0.20, 'control': 0.00},
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
    if weights['efficiency'] > 0:  total_score += calc_efficiency(f_map, e_map) * weights['efficiency']
    if weights['alignment'] > 0:   total_score += calc_alignment(f_map) * weights['alignment']
    if weights['threaten_enemy'] > 0: total_score += calc_threat(f_map, e_map, is_my_turn) * weights['threaten_enemy']
    if weights['safety'] > 0:      total_score += calc_safety(f_map, e_map, is_my_turn) * weights['safety']
    if weights['hunting'] > 0:     total_score += calc_hunting(f_map, e_map) * weights['hunting']
    if weights['control'] > 0:   total_score += calc_control(f_map, e_map) * weights['control']

    return total_score
# ==============================================================================

# ==============================================================================
# Evaluation functions that being tested
def eval0(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'phalanx_aggressive')

def eval1(board: BoardState, root_color: int, is_red: bool) -> float:
    return evaluate_board(board, root_color, is_red, 'hunt_balanced')

def end_phase_eval(board: BoardState, root_color: int, is_red: bool) -> float:
    """Evaluation for the end phase, focused on final positioning and material."""
    return evaluate_board(board, root_color, is_red, 'endgame')

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
# PLACING PHASE 
# ==============================================================================
def get_place_maps(board: BoardState, root_color: int) -> tuple[list, list]:
    """Extracts positions of friendly and enemy tokens."""
    friendly = []
    enemies = []
    for idx in range(64):
        if board.is_empty(idx): continue
        if (root_color == 1 and board.is_red(idx)) or (root_color == 0 and board.is_blue(idx)):
            friendly.append(idx)
        else:
            enemies.append(idx)
    return friendly, enemies


def calc_place_centrality(friendly: list) -> float:
    """Rewards placing tokens toward the absolute center-4 squares."""
    score = 0.0
    for idx in friendly:
        score += CENTER_WEIGHTS[idx] * 6.0
    return score


def get_compact_l_score(stacks: list) -> float:
    """Helper: Detects and scores the best Compact L-shape in a list of stacks."""
    if len(stacks) < 3: 
        return 0.0
        
    best_score = 0.0
    stack_set = set(stacks)

    for elbow in stacks:
        neighbors = [n for n in ADJACENT[elbow].values() if n in stack_set]
        if len(neighbors) < 2: 
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                d1, d2 = n1 - elbow, n2 - elbow

                # Perpendicular check (one horizontal, one vertical offset)
                is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or (d1 in (-1, 1) and d2 in (-8, 8))
                if not is_perp: 
                    continue

                score = 20.0
                center_score = sum(sc for sc in (CENTER_WEIGHTS[elbow] * 6.0, CENTER_WEIGHTS[n1] * 6.0, CENTER_WEIGHTS[n2] * 6.0))

                score += center_score

                if EDGE_DIST[n1] == 0: score -= CENTER_WEIGHTS[n1] * 6.0
                if EDGE_DIST[n2] == 0: score -= CENTER_WEIGHTS[n2] * 6.0

                if score > best_score:
                    best_score = score
                    
    return best_score


def calc_place_synergy(friendly: list) -> float:
    """Phase-aware synergy. Dictates the structure based on current turn count."""
    f_count = len(friendly)
    if f_count < 2: 
        return 0.0

    # Phase 2: Force the second piece to be a separated blocker
    if f_count == 2:
        dist = MANHATTAN[friendly[0]][friendly[1]]
        if dist == 1: return -20.0  # Heavily penalize early adjacency
        if dist == 2: return 8.0    # Perfect spacing
        return 4.0

    # Phase 3: Build the L-start (one pair adjacent, one piece separated)
    if f_count == 3:
        best = -999.0
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = friendly[i], friendly[j]
                if MANHATTAN[a][b] != 1: continue
                blocker = friendly[3 - i - j]

                score = 0.0
                if a in CENTER_4 or b in CENTER_4: score += CENTER_WEIGHTS[27] * 6.0
                score += sum(CENTER_WEIGHTS[x] for x in (a, b))

                # Ensure blocker is separated from the pair
                min_d = min(MANHATTAN[blocker][a], MANHATTAN[blocker][b])
                if min_d >= 2: score += 12.0
                elif min_d == 1: score -= 5.0

                if score > best: best = score
        return best if best > -999.0 else -20.0

    # Phase 4: Full Compact L established
    return get_compact_l_score(friendly)


def calc_place_denial(board: BoardState, friendly: list, enemies: list) -> float:
    """Evaluates how effectively the 'Blocker' piece disrupts the enemy."""
    score = 0.0
    if not friendly or not enemies: 
        return score

    # Blocking Enemy Expansion Zones
    f_adj = set()
    for f in friendly:
        f_adj.update(ADJACENT[f].values())

    e_adjacent = set()
    for e in enemies:
        for nb in ADJACENT[e].values():
            if board.is_empty(nb):
                e_adjacent.add(nb)

    blocked = len(e_adjacent & f_adj)
    score += blocked * 6.0

    # Pushing the Blocker Toward Enemy Centroid
    cr, cc = 3.5, 3.5
    e_r = sum(e >> 3 for e in enemies) / len(enemies)
    e_c = sum(e & 7 for e in enemies) / len(enemies)
    toward_r, toward_c = e_r - cr, e_c - cc

    for f in friendly:
        fr, fc = f >> 3, f & 7
        dot = (fr - cr) * toward_r + (fc - cc) * toward_c
        score += dot * 4.0

    # Anti-Enemy L-Shape
    if len(enemies) >= 3:
        score -= get_compact_l_score(enemies) * 0.8

    return score

# Coefficient set up for each aspect for different profiles
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
    