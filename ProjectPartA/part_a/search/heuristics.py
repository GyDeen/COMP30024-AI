MAX_ROW = 8
MAX_COL = 8


def heuristic_bit(board_key: tuple[int, ...]) -> int:
    # identify red and blue stacks and calculate cumulative height of red tokens
    red_indices = []
    blue_indices = []
    total_red_height = 0
    
    for i, val in enumerate(board_key):
        if val > 0:
            red_indices.append(i)
            total_red_height += val
        elif val < 0:
            blue_indices.append(i)

    # check if all blue pieces are eliminated
    if not blue_indices:
        return 0
    
    # check if there are no red pieces - impossible to 
    if not red_indices:
        return 999 

    # for each blue stack, find the closest red stack and determine how many actions it would take to eliminate it
    max_min_actions = 0
    # max reach possible would be the total height of the red stacks
    max_reach = max(1, total_red_height)


    for b_idx in blue_indices:
        br, bc = divmod(b_idx, MAX_COL) # get row and column of the blue stack (no. row = no. col)
        
        best_dist = MAX_COL * MAX_ROW  # initialise to a large number
        for r_idx in red_indices:
            rr, rc = divmod(r_idx, 8)
            dist = abs(br - rr) + abs(bc - rc)
            if dist < best_dist:
                best_dist = dist
        
        # cost = distance / maximum reach per action
        actions_needed = (best_dist + max_reach - 1) // max_reach
        
        # the maximum of the minimum actions needed guarantees admissibility
        if actions_needed > max_min_actions:
            max_min_actions = actions_needed

    return max_min_actions
