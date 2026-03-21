DIST_TABLE = [[abs(i//8 - j//8) + abs(i%8 - j%8) for j in range(64)] for i in range(64)]

def heuristic_bit(board_key):
    """Bitwise heuristic using integer board encoding."""
    reds = []   # Store as (index, height)
    blues = []  # Store as (index, height)
    
    # Collect exact pieces and their heights
    for i, val in enumerate(board_key):
        if val > 0:
            reds.append((i, val))
        elif val < 0:
            blues.append((i, -val))

    if not blues: return 0
    if not reds: return 100

    max_min_cost = 0

    # Calculate the specific cost for each Blue stack
    for b_idx, b_height in blues:
        min_cost_for_this_blue = float('inf')

        # Fast edge distance calculation
        br, bc = divmod(b_idx, 8)
        # Distance to the closest edge
        min_edge_dist = min(br, 7 - br, bc, 7 - bc)

        for r_idx, r_height in reds:
            dist = DIST_TABLE[b_idx][r_idx]

            # The minimum cost to reach the blue stack 
            if r_height == 1:
                cost = dist
            else:
                cost = max(1, dist - r_height + 1)

            # If red height is not sufficient to push blue off the edge or eat it, add 1 since it need to "merge" with another red
            # to be useful
            if r_height < b_height:
                if r_height == 1 or r_height <= min_edge_dist:
                    cost += 1 

            # There is a lower cost of eliminating this blue stack, so we take the minimum over all red stacks
            if cost < min_cost_for_this_blue:
                min_cost_for_this_blue = cost

        # We want the maximum of these minimum costs across all blue stacks to get a safe lower bound on the remaining cost
        if min_cost_for_this_blue > max_min_cost:
            max_min_cost = min_cost_for_this_blue

    return max_min_cost

