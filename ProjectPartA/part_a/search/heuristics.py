DIST_TABLE = [[abs(i//8 - j//8) + abs(i%8 - j%8) for j in range(64)] for i in range(64)]

def heuristic_bit(board_key):
    reds = []   
    blues = []  
    
    for i, val in enumerate(board_key):
        if val > 0:
            reds.append((i, val))
        elif val < 0:
            blues.append((i, -val))

    if not blues: return 0
    if not reds: return 100

    max_min_cost = 0

    for b_idx, b_height in blues:
        best_cost_for_blue = float('inf')

        # We treat every Red piece as a potential "Anchor" where the final merge happens
        for anchor_idx, anchor_h in reds:
            d_target = DIST_TABLE[b_idx][anchor_idx]
            current_h = anchor_h
            
            # Cost if this anchor attacks alone without any help
            best_cost_for_anchor = max(1, d_target - current_h + 1)
            
            # Gather all other Red pieces to potentially pull them in
            other_reds = []
            for r_idx, r_height in reds:
                if r_idx != anchor_idx:
                    dist_to_anchor = DIST_TABLE[anchor_idx][r_idx]
                    other_reds.append((dist_to_anchor, r_height))
                    
            # Sort by distance so we naturally absorb the closest pieces first
            other_reds.sort(key=lambda x: x[0])
            
            max_radius = 0
            # Simulate pulling in pieces to build a super-stack
            for dist_to_anchor, h in other_reds:
                current_h += h
                
                # The minimum travel cost to bring this piece to the anchor
                max_radius = dist_to_anchor 
                
                # Total cost = (Travel distance of the furthest piece) + (Actions to cascade)
                cost = max_radius + max(1, d_target - current_h + 1)
                
                # If pulling this piece in actually saved us moves, update the best cost
                if cost < best_cost_for_anchor:
                    best_cost_for_anchor = cost
                    
        # The lowest cost found among all anchors is the true lower bound for this Blue stack
        if best_cost_for_anchor < best_cost_for_blue:
            best_cost_for_blue = best_cost_for_anchor

        # We take the bottleneck across all Blue stacks
        if best_cost_for_blue > max_min_cost:
            max_min_cost = best_cost_for_blue

    return max_min_cost
