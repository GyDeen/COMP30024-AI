"""The baseline's eight unweighted movement features.

Scalar functions retain their existing arithmetic and asymmetric behavior.
Agent evaluators can call only their nonzero-weight features; dataset preparation
can call feature_vector to calculate every feature in a stable order.
"""

from Agents.common.board import BoardState
from .geometry import ADJACENT, CENTER_WEIGHTS, EDGE_DIST

RED = 1
BLUE = 0
FEATURE_NAMES = (
    "material", "positioning", "efficiency", "alignment",
    "threaten_enemy", "safety", "hunting", "control",
)


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
    for f_idx, h in friendly_map.items():
        r, c = divmod(f_idx, 8)

        # Calculate the absolute furthest edge from this specific square
        max_useful_height = max(r, 7 - r, c, 7 - c)

        # Prefer height of 4 or 5
        if 2 <= h <= 3: f_score += 3.0
        elif 4 <= h < 5: f_score += 15.0
        elif h == 5: f_score += 5.0
        elif h == 6: f_score -= 10.0
        elif h > 6: f_score -= (30.0 + ((h - 6) * 10.0))

        # If the stack is taller than its furthest reach, penalize the wasted tokens
        if h > max_useful_height:
            redundant_tokens = h - max_useful_height
            f_score -= (redundant_tokens * 10.0)

    e_score = 0.0
    for e_idx, h in enemy_map.items():
        r, c = divmod(e_idx, 8)
        max_useful_height = max(r, 7 - r, c, 7 - c)

        if 2 <= h <= 3: e_score += 3.0
        elif 4 <= h < 5: e_score += 15.0
        elif h == 5: e_score += 5.0
        elif h == 6: e_score -= 10.0
        elif h > 6: e_score -= (30.0 + ((h - 6) * 10.0))

        if h > max_useful_height:
            redundant_tokens = h - max_useful_height
            e_score -= (redundant_tokens * 15.0)

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


def feature_vector(board: BoardState, root_color: int, is_red_turn: bool) -> tuple[float, ...]:
    """Return all eight raw values in FEATURE_NAMES order, without NumPy.

    root_color is 1 for Red or 0 for Blue. Side to move is independent of the
    chosen perspective. The caller chooses any storage dtype or model transform.
    """
    if root_color not in (RED, BLUE):
        raise ValueError("root_color must be 1 (RED) or 0 (BLUE)")
    friendly, enemy = get_stack_maps(board, root_color)
    is_my_turn = (root_color == RED) == is_red_turn
    return (
        calc_material(friendly, enemy),
        calc_positioning(friendly, enemy),
        calc_efficiency(friendly, enemy),
        calc_alignment(friendly),
        calc_threat(friendly, enemy, is_my_turn),
        calc_safety(friendly, enemy, is_my_turn),
        calc_hunting(friendly, enemy),
        calc_control(friendly, enemy),
    )
