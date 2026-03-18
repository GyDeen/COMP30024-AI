from .core import PlayerColor
import math


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
    red_positions = []
    blue_positions = []
    total_red = 0
    highest_red = 0
    
    for i, val in enumerate(board_key):
        if val == 0: continue
        r, c = i // 8, i % 8
        if val > 0:
            red_positions.append((r, c))
            if val > highest_red: highest_red = val
        else:
            blue_positions.append((r, c))

    if not blue_positions: return 0
    if not red_positions: return 100

    max_min_distance_between_red_blue = 0
    for br, bc in blue_positions:
        min_distance_to_red = min(abs(br - rr) + abs(bc - rc) for rr, rc in red_positions)
        max_min_distance_between_red_blue = max(max_min_distance_between_red_blue, min_distance_to_red)

    return math.ceil(max_min_distance_between_red_blue / highest_red)
