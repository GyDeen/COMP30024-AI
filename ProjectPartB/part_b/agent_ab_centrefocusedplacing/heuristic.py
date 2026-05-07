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
# PLACEMENT EVALUATION — L-shape + blocker, all inside (2,2)-(5,5) zone
# ==============================================================================

# The 4x4 inner zone: rows 2-5, cols 2-5
_ZONE = frozenset({
    18, 19, 20, 21,   # (2,2) (2,3) (2,4) (2,5)
    26, 27, 28, 29,   # (3,2) (3,3) (3,4) (3,5)
    34, 35, 36, 37,   # (4,2) (4,3) (4,4) (4,5)
    42, 43, 44, 45,   # (5,2) (5,3) (5,4) (5,5)
})

# The 4 center cells
_CENTER_4 = frozenset({27, 28, 35, 36})  # (3,3) (3,4) (4,3) (4,4)

# Manhattan distance to nearest center-4 cell
_DIST_TO_CENTER = [0] * 64
for _i in range(64):
    _r, _c = divmod(_i, 8)
    _DIST_TO_CENTER[_i] = min(
        abs(_r - 3) + abs(_c - 3), abs(_r - 3) + abs(_c - 4),
        abs(_r - 4) + abs(_c - 3), abs(_r - 4) + abs(_c - 4),
    )

# Adjacency sets
_ADJ_SET = [set() for _ in range(64)]
for _i in range(64):
    _r, _c = divmod(_i, 8)
    if _r > 0: _ADJ_SET[_i].add(_i - 8)
    if _r < 7: _ADJ_SET[_i].add(_i + 8)
    if _c > 0: _ADJ_SET[_i].add(_i - 1)
    if _c < 7: _ADJ_SET[_i].add(_i + 1)

# Manhattan distance lookup
_MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        _MANHATTAN[_a][_b] = abs((_a >> 3) - (_b >> 3)) + abs((_a & 7) - (_b & 7))


# ── Helpers ──────────────────────────────────────────────────────────

def _detect_compact_L(stacks):
    """
    Find best compact right-angle L among stacks.
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
                d1, d2 = n1 - elbow, n2 - elbow
                is_perp = (d1 in (-8, 8) and d2 in (-1, 1)) or \
                          (d1 in (-1, 1) and d2 in (-8, 8))
                if not is_perp:
                    continue

                score = 40.0
                cc = sum(1 for s in (elbow, n1, n2) if s in _CENTER_4)
                if cc >= 2:   score += 40.0
                elif cc == 1: score += 28.0
                else:         score += max(0, 15.0 - _DIST_TO_CENTER[elbow] * 5.0)

                avg_cd = (_DIST_TO_CENTER[elbow] + _DIST_TO_CENTER[n1] + _DIST_TO_CENTER[n2]) / 3.0
                score += max(0, 12.0 - avg_cd * 3.0)

                if score > best_score:
                    best_score = score
                    best_elbow = elbow
                    best_L = [elbow, n1, n2]

    return best_score, best_elbow, best_L


def _is_L_completion(a, b, candidate):
    """Check if candidate completes a right-angle L with adjacent pair (a, b)."""
    d_ab = b - a
    for base in (a, b):
        for pd in (-8, 8, -1, 1):
            if pd == d_ab or pd == -d_ab:
                continue
            if base + pd == candidate:
                return True
    return False


def _count_blocked_enemy_cells(idx, enemy, f_set, e_set):
    """Count how many enemy expansion cells are blocked by placing at idx."""
    my_adj = _ADJ_SET[idx]
    enemy_wants = set()
    for e in enemy:
        for nb in _ADJ_SET[e]:
            if nb not in e_set and nb not in f_set:
                enemy_wants.add(nb)
    return len(enemy_wants & my_adj)


# ── Main evaluation ──────────────────────────────────────────────────

def eval_place_L(board: BoardState, root_color: int, is_red: bool) -> float:
    """
    Placement evaluation.  All stacks should stay inside the 4x4 zone
    (2,2)-(5,5).  Form a right-angle L (3 stacks) + 1 blocker.

    Phase 1 (f=1): grab center-4
    Phase 2 (f=2): blocker in zone, toward enemy, NOT adjacent
    Phase 3 (f=3): complete L if pair exists, else start pair
    Phase 4 (f=4): full L + well-placed blocker
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
    e_set = set(enemy)

    score = 0.0

    # ── ZONE ENFORCEMENT: heavy penalty for anything outside (2,2)-(5,5) ──
    for idx in friendly:
        if idx in _ZONE:
            score += 10.0                      # in zone: good
            if idx in _CENTER_4:
                score += 20.0                  # center-4: great
            score += (2 - _DIST_TO_CENTER[idx]) * 5.0  # closer to center = better
        else:
            score -= 50.0                      # outside zone: terrible

    # ── MATERIAL ──
    score += (f_count - e_count) * 10.0

    # ── Direction toward/away from enemy ──
    cr, cc = 3.5, 3.5
    if enemy:
        e_avg_r = sum(e >> 3 for e in enemy) / e_count
        e_avg_c = sum(e & 7 for e in enemy) / e_count
        toward_r, toward_c = e_avg_r - cr, e_avg_c - cc
        away_r, away_c = -toward_r, -toward_c
    else:
        e_avg_r = e_avg_c = 3.5
        toward_r = toward_c = away_r = away_c = 0.0

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Just grab center
    # ════════════════════════════════════════════════════════════════
    if f_count <= 1:
        pass

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: Center + Blocker
    # Blocker: in zone, NOT adjacent to center stack, toward enemy,
    # blocks enemy expansion.
    # ════════════════════════════════════════════════════════════════
    elif f_count == 2:
        a, b = friendly[0], friendly[1]
        dist_ab = _MANHATTAN[a][b]

        # Penalize adjacency — blocker should be separate
        if dist_ab == 1:
            score -= 20.0
        elif dist_ab == 2:
            score += 8.0

        # Identify blocker (the non-center one)
        if a in _CENTER_4 and b not in _CENTER_4:
            blocker = b
        elif b in _CENTER_4 and a not in _CENTER_4:
            blocker = a
        else:
            blocker = b if _DIST_TO_CENTER[a] < _DIST_TO_CENTER[b] else a

        # Blocker toward enemy (capped)
        if enemy:
            br, bc = blocker >> 3, blocker & 7
            dot = (br - cr) * toward_r + (bc - cc) * toward_c
            score += min(dot * 6.0, 10.0)

        # Blocker blocks enemy expansion
        if enemy:
            score += _count_blocked_enemy_cells(blocker, enemy, f_set, e_set) * 8.0

    # ════════════════════════════════════════════════════════════════
    # PHASE 3: Three stacks — complete L or place pair + blocker
    # If pair exists, 3rd stack should complete the right-angle.
    # If no pair, 3rd should be adjacent to center to START the pair.
    # ════════════════════════════════════════════════════════════════
    elif f_count == 3:
        # Find adjacent pairs
        pairs = []
        for i in range(3):
            for j in range(i + 1, 3):
                if _MANHATTAN[friendly[i]][friendly[j]] == 1:
                    third = friendly[3 - i - j]
                    pairs.append((friendly[i], friendly[j], third))

        if not pairs:
            score -= 25.0   # no pair = bad
        else:
            best = -9999.0
            for a, b, third in pairs:
                cfg = 0.0

                # Pair has center-4 cell
                if a in _CENTER_4 or b in _CENTER_4:
                    cfg += 25.0

                # Is 3rd stack an L-completion? (perpendicular neighbor)
                if _is_L_completion(a, b, third):
                    # ── L COMPLETED ── huge bonus
                    cfg += 50.0
                    l_all = [a, b, third]
                    cfg += sum(1 for s in l_all if s in _CENTER_4) * 10.0
                    # L away from enemy
                    if enemy:
                        lr = sum(s >> 3 for s in l_all) / 3.0
                        lc = sum(s & 7 for s in l_all) / 3.0
                        cfg += ((lr - cr) * away_r + (lc - cc) * away_c) * 5.0
                else:
                    # ── 3rd is blocker ── modest score
                    cfg += 5.0
                    # Toward enemy (small, capped)
                    if enemy:
                        tr, tc2 = third >> 3, third & 7
                        dot = (tr - cr) * toward_r + (tc2 - cc) * toward_c
                        cfg += min(dot * 3.0, 6.0)
                    # Blocks enemy
                    if enemy:
                        cfg += _count_blocked_enemy_cells(third, enemy, f_set, e_set) * 4.0
                    # Separated from pair
                    if min(_MANHATTAN[third][a], _MANHATTAN[third][b]) >= 2:
                        cfg += 6.0

                if cfg > best:
                    best = cfg
            score += best

    # ════════════════════════════════════════════════════════════════
    # PHASE 4: Complete L + blocker
    # ════════════════════════════════════════════════════════════════
    elif f_count >= 4:
        l_score, l_elbow, l_stacks = _detect_compact_L(friendly)
        score += l_score

        if l_stacks:
            l_set = set(l_stacks)
            elbow, arm1, arm2 = l_stacks[0], l_stacks[1], l_stacks[2]

            # L away from enemy
            if enemy:
                lr = sum(s >> 3 for s in l_stacks) / 3.0
                lc = sum(s & 7 for s in l_stacks) / 3.0
                score += ((lr - cr) * away_r + (lc - cc) * away_c) * 5.0

            # Evaluate blocker (non-L stacks)
            for idx in friendly:
                if idx in l_set:
                    continue

                # Arm extension = bad
                extending = False
                for arm in (arm1, arm2):
                    d = arm - elbow
                    if _MANHATTAN[idx][arm] == 1 and (idx - arm) == d:
                        extending = True; break
                    if _MANHATTAN[idx][elbow] == 1 and (idx - elbow) == -d:
                        extending = True; break

                if extending:
                    score -= 30.0
                else:
                    # Toward enemy (capped)
                    if enemy:
                        br, bc = idx >> 3, idx & 7
                        dot = (br - cr) * toward_r + (bc - cc) * toward_c
                        score += min(dot * 4.0, 8.0)
                    # Separated from L
                    if min(_MANHATTAN[idx][s] for s in l_stacks) >= 2:
                        score += 8.0
                    # Blocks enemy
                    if enemy:
                        score += _count_blocked_enemy_cells(idx, enemy, f_set, e_set) * 5.0

        # Block enemy expansion (universal)
        if friendly and enemy:
            our_adj = set()
            for f in friendly:
                our_adj.update(_ADJ_SET[f])
            ew = set()
            for e in enemy:
                for nb in _ADJ_SET[e]:
                    if board.is_empty(nb):
                        ew.add(nb)
            score += len(ew & our_adj) * 4.0

        # Penalize enemy L
        if e_count >= 3:
            el, _, _ = _detect_compact_L(enemy)
            score -= el * 0.5

    return score
    