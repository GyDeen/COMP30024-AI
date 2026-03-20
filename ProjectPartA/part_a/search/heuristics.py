from .core import PlayerColor
import math

DIST_TABLE = [[abs(i//8 - j//8) + abs(i%8 - j%8) for j in range(64)] for i in range(64)]

def heuristic(board_dict):
    """Admissible lower bound on actions remaining.

    We compute, for each blue stack, the Manhattan distance to the nearest red
    stack. In a single action a red stack can cover at most H cells in one
    direction, where H is the maximum possible red stack height (bounded above
    by the total number of red tokens). Therefore, reaching that blue stack
    must take at least ceil(dist / H) actions. Taking the maximum over all blue
    stacks yields a safe lower bound on the remaining cost while providing more
    guidance than the trivial 0 heuristic.
    """
    blue_coords = [coord for coord, cell in board_dict.items()
                  if cell.color == PlayerColor.BLUE]
    if not blue_coords:
        return 0

    red_coords = [coord for coord, cell in board_dict.items()
                 if cell.color == PlayerColor.RED]
    if not red_coords:
        return 0

    total_red_tokens = sum(cell.height for _, cell in board_dict.items()
                           if cell.color == PlayerColor.RED)
    max_step = max(1, total_red_tokens)

    def manhattan(a, b):
        return abs(a.r - b.r) + abs(a.c - b.c)

    lower_bounds = []
    for b in blue_coords:
        min_dist = min(manhattan(b, r) for r in red_coords)
        lower_bounds.append(math.ceil(min_dist / max_step))

    return max(lower_bounds)


def heuristic_bit(board_key):
    """Bitwise heuristic using integer board encoding."""
    reds = []   # Store as (index, height)
    blues = []  # Store as (index, height)
    
    # 1. Collect exact pieces and their heights
    for i, val in enumerate(board_key):
        if val > 0:
            reds.append((i, val))
        elif val < 0:
            blues.append((i, -val))

    if not blues: return 0
    if not reds: return 100

    max_min_cost = 0

    # 2. Calculate the specific cost for each Blue stack
    for b_idx, b_height in blues:
        min_cost_for_this_blue = float('inf')

        # Fast edge distance calculation
        br, bc = divmod(b_idx, 8)
        # Distance to the closest edge (0 = on edge, 1 = 1 step away, etc.)
        min_edge_dist = min(br, 7 - br, bc, 7 - bc)

        for r_idx, r_height in reds:
            dist = DIST_TABLE[b_idx][r_idx]

            # The "Align and Fire" base cost
            if r_height == 1:
                cost = dist
            else:
                cost = max(1, dist - r_height + 1)

            # YOUR NEW "PUSH POWER" PENALTY:
            if r_height < b_height:
                # Condition 1: Height 1 cannot cascade at all, so it can NEVER push.
                # Condition 2: Red's height is too small to push Blue off the closest edge.
                if r_height == 1 or r_height <= min_edge_dist:
                    cost += 1 

            if cost < min_cost_for_this_blue:
                min_cost_for_this_blue = cost

        if min_cost_for_this_blue > max_min_cost:
            max_min_cost = min_cost_for_this_blue

    return max_min_cost
    # red_positions = []
    # blue_positions = []
    # highest_red = 0
    
    # # Generate the position lists and track the highest red stack height
    # for i, val in enumerate(board_key):
    #     if val == 0: continue
    #     if val > 0:
    #         red_positions.append(i)
    #         if val > highest_red: highest_red = val
    #     else:
    #         blue_positions.append(i)

    # if not blue_positions: return 0
    # if not red_positions: return 100

    # # Find the minimum Manhattan distance from each blue to the nearest red, then take the maximum of those
    # max_min_distance_between_red_blue = 0
    # for blue_idx in blue_positions:
    #     min_distance_to_red = min(DIST_TABLE[blue_idx][red_idx] for red_idx in red_positions)
    #     max_min_distance_between_red_blue = max(max_min_distance_between_red_blue, min_distance_to_red)

    # return max(math.ceil(max_min_distance_between_red_blue / highest_red), len(blue_positions) / 7)
