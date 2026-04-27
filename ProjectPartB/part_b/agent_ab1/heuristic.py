from .board import BoardState
from .search import ADJACENT

RED = 1
BLUE = 0


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

""" Board evaluation function set 1"""
def get_eval_fun0(board: BoardState, root_color: bool) -> tuple[float, float, float, float, float]:
    f_tokens = 0
    e_tokens = 0
    f_stacks = 0
    e_stacks = 0
    center_score = 0.0

    threatened_enemies = set()
    threatened_friends = set()

    for idx in range(64):
        if board.is_empty(idx): continue

        is_friendly = (root_color and board.is_red(idx)) or (not root_color and board.is_blue(idx))
        h = board.get_height(idx)

        if is_friendly:
            f_tokens += h
            f_stacks += 1
            center_score += (CENTER_WEIGHTS[idx] * h)

            # Check all 4 orthogonal directions for threats
            for neighbor_idx in ADJACENT[idx].values():
                if not board.is_empty(neighbor_idx):
                    neighbor_friendly = (root_color and board.is_red(neighbor_idx)) or (not root_color and board.is_blue(neighbor_idx))
                    
                    if not neighbor_friendly:
                        enemy_h = board.get_height(neighbor_idx)
                        # We can eat them
                        if h >= enemy_h:
                            threatened_enemies.add(neighbor_idx)
                        # They can eat us
                        if enemy_h >= h:
                            threatened_friends.add(idx)
        else:
            e_tokens += h
            e_stacks += 1

    # Percentage of board tokens that are ours
    total_tokens = f_tokens + e_tokens
    material = (f_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0

    # 100% means all tokens are in one massive stack
    if f_tokens > f_stacks:
        stack_power = ((f_tokens - f_stacks) / (f_tokens - 1)) * 100.0
    else:
        stack_power = 0.0

    # Percentage of enemy stacks we are threatening
    threat_inflicted = (len(threatened_enemies) / e_stacks * 100.0) if e_stacks > 0 else 0.0

    # Inverse percentage of friendly stacks in danger
    safety = 100.0 - ((len(threatened_friends) / f_stacks * 100.0) if f_stacks > 0 else 0.0)

    # Max center weight is 4.0 per token
    centrality = (center_score / (f_tokens * 4.0) * 100.0) if f_tokens > 0 else 0.0

    return material, stack_power, threat_inflicted, safety, centrality


def eval_normalized_0(board: BoardState, root_color: bool, is_red: bool) -> float:
    """Aggressive evaluation for playing phase, focused on maximizing threats and minimizing our own vulnerabilities."""
    mat, stack, threat, safe, cen = get_eval_fun0(board, root_color)
    return (mat * 0.40) + (threat * 0.30) + (safe * 0.20) + (stack * 0.05) + (cen * 0.05)

def eval_normalized_1(board: BoardState, root_color: bool, is_red: bool) -> float:
    """Balanced evaluation for playing phase, with moderate emphasis on both offense and defense."""
    mat, stack, threat, safe, cen = get_eval_fun0(board, root_color)
    return (mat * 0.30) + (stack * 0.30) + (cen * 0.25) + (safe * 0.15) + (threat * 0.00)


def eval_normalized_2(board: BoardState, root_color: bool, is_red: bool) -> float:
    """Balanced evaluation for playing phase, with moderate emphasis on all aspects."""
    mat, stack, threat, safe, cen = get_eval_fun0(board, root_color)
    return (mat * 0.35) + (safe * 0.25) + (cen * 0.15) + (threat * 0.15) + (stack * 0.10)
    
"""End of board evaluation function set 1"""


""" Board evaluation function set 2"""
def get_eval_fun1(board: BoardState, root_color: int, is_current_turn_red: bool) -> dict:
    features = {
        'material': 0.0,
        'efficiency': 0.0,
        'alignment': 0.0,
        'safety': 0.0,
        'positioning': 0.0
    }
    
    is_my_turn = (root_color == 1 and is_current_turn_red) or \
                 (root_color == 0 and not is_current_turn_red)

    friendly_stacks_map = {}  
    enemy_stacks_map = {}
    
    # Material value and stack height
    for cell_index in range(64):
        if board.is_empty(cell_index): 
            continue
        
        stack_height = board.get_height(cell_index)
        is_friendly = (root_color == 1 and board.is_red(cell_index)) or \
                      (root_color == 0 and board.is_blue(cell_index))
        
        if is_friendly:
            friendly_stacks_map[cell_index] = stack_height
            features['material'] += (stack_height * 10.0)
            features['positioning'] += (CENTER_WEIGHTS[cell_index] * stack_height / 2)
            
            # Penalize for low stack heights (vulnerable to being eaten) and very high stacks (wasteful and vulnerable to cascade)
            if stack_height == 1:
                features['efficiency'] -= 10.0
            elif 2 <= stack_height <= 3:
                features['efficiency'] += 5.0   
            elif 4 <= stack_height <= 6:
                features['efficiency'] += 15.0  
            elif stack_height > 6:
                features['efficiency'] -= 15.0  
        else:
            enemy_stacks_map[cell_index] = stack_height
            features['material'] -= (stack_height * 10.0)
            features['positioning'] -= (CENTER_WEIGHTS[cell_index] * stack_height / 2)

    friendly_indices = list(friendly_stacks_map.keys())

    # Alignment check, make sure each other can help (Mutual Defense Range)
    for current_item_index in range(len(friendly_indices)):
        idx_1 = friendly_indices[current_item_index]
        row_1, col_1 = divmod(idx_1, 8)
        h1 = friendly_stacks_map[idx_1]
        
        for next_item_index in range(current_item_index + 1, len(friendly_indices)):
            idx_2 = friendly_indices[next_item_index]
            row_2, col_2 = divmod(idx_2, 8)
            h2 = friendly_stacks_map[idx_2]
            
            # Has alignment
            if row_1 == row_2 or col_1 == col_2:
                distance = abs(row_1 - row_2) + abs(col_1 - col_2)
                # If one of them gets attacked, can the other push the attacker off the board?
                can_push_off_if_attacked = (h1 - distance > EDGE_DIST[idx_2]) or (h2 - distance > EDGE_DIST[idx_1])
                # Can they reach each other to retaliate?
                if can_push_off_if_attacked: 
                    features['alignment'] += 5.0

    # Check the threats from enemy stacks. We consider two main types of threats:
    for enemy_index, enemy_height in enemy_stacks_map.items():
        enemy_row, enemy_col = divmod(enemy_index, 8)
        
        for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            
            # There is stack under direct attack
            adj_row, adj_col = enemy_row + delta_row, enemy_col + delta_col
            if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                adj_idx = adj_row * 8 + adj_col
                if adj_idx in friendly_stacks_map:
                    if enemy_height >= friendly_stacks_map[adj_idx]:
                        features['safety'] -= 5.0 if is_my_turn else 50.0
            
            # CASCADE Push-Off-Board
            for step_distance in range(1, enemy_height + 1):
                target_row = enemy_row + (delta_row * step_distance)
                target_col = enemy_col + (delta_col * step_distance)
                
                # If the line goes off board before hitting anything, stop checking
                if not (0 <= target_row < 8 and 0 <= target_col < 8): 
                    break 
                
                target_index = target_row * 8 + target_col
                friendly_in_line_count = 0
                if target_index in friendly_stacks_map:
                    friendly_in_line_count += 1
                    pushed_row = enemy_row + (delta_row * (enemy_height + friendly_in_line_count))
                    pushed_col = enemy_col + (delta_col * (enemy_height + friendly_in_line_count))

                    if not (0 <= pushed_row < 8 and 0 <= pushed_col < 8):
                        threatened_height = friendly_stacks_map[target_index]
                        if is_my_turn:
                            features['safety'] -= (threatened_height * 2.0)
                        else:
                            features['safety'] -= (threatened_height * 10.0)
                
    return features


def eval_phalanx_0(board: BoardState, root_color: int, is_current_turn_red: bool) -> float:
    """Balanced evaluation for playing phase, with moderate emphasis on all aspects."""
    features = get_eval_fun1(board, root_color, is_current_turn_red)
    return (features['material'] * 0.20) + (features['safety'] * 0.20) + (features['positioning'] * 0.20) + \
           (features['alignment'] * 0.20) + (features['efficiency'] * 0.20)

def eval_phalanx_1(board: BoardState, root_color: int, is_current_turn_red: bool) -> float:
    """Aggresive evaluation for playing phase. Focus more on eliminating opponent's token while maintain own material advantage."""
    features = get_eval_fun1(board, root_color, is_current_turn_red)
    return (features['material'] * 0.40) + (features['safety'] * 0.25) + (features['positioning'] * 0.15) + \
           (features['alignment'] * 0.15) + (features['efficiency'] * 0.10)

def eval_phalanx_2(board: BoardState, root_color: int, is_current_turn_red: bool) -> float:
    """Defensive evaluation for playing phase. Focus more on maintaining own material advantage and safety."""
    features = get_eval_fun1(board, root_color, is_current_turn_red)
    return (features['material'] * 0.30) + (features['safety'] * 0.30) + (features['positioning'] * 0.20) + \
           (features['alignment'] * 0.20) + (features['efficiency'] * 0.15)

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
    