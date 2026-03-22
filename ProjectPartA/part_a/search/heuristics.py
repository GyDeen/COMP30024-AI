# DIST_TABLE = [[abs(i//8 - j//8) + abs(i%8 - j%8) for j in range(64)] for i in range(64)]

# def heuristic_bit(board_key):
#     #all g except test 8
#     reds = []   
#     blues = []  
    
#     for i, val in enumerate(board_key):
#         if val > 0:
#             reds.append((i, val))
#         elif val < 0:
#             blues.append((i, -val))

#     if not blues: return 0
#     if not reds: return 100

#     max_min_cost = 0

#     for b_idx, b_height in blues:
#         best_cost_for_blue = float('inf')

#         # We treat every Red piece as a potential "Anchor" where the final merge happens
#         for anchor_idx, anchor_h in reds:
#             d_target = DIST_TABLE[b_idx][anchor_idx]
#             current_h = anchor_h
            
#             # Cost if this anchor attacks alone without any help
#             best_cost_for_anchor = max(1, d_target - current_h + 1)
            
#             # Gather all other Red pieces to potentially pull them in
#             other_reds = []
#             for r_idx, r_height in reds:
#                 if r_idx != anchor_idx:
#                     dist_to_anchor = DIST_TABLE[anchor_idx][r_idx]
#                     other_reds.append((dist_to_anchor, r_height))
                    
#             # Sort by distance so we naturally absorb the closest pieces first
#             other_reds.sort(key=lambda x: x[0])
            
#             max_radius = 0
#             # Simulate pulling in pieces to build a super-stack
#             for dist_to_anchor, h in other_reds:
#                 current_h += h
                
#                 # The minimum travel cost to bring this piece to the anchor
#                 max_radius = dist_to_anchor 
                
#                 # Total cost = (Travel distance of the furthest piece) + (Actions to cascade)
#                 cost = max_radius + max(1, d_target - current_h + 1)
                
#                 # If pulling this piece in actually saved us moves, update the best cost
#                 if cost < best_cost_for_anchor:
#                     best_cost_for_anchor = cost
                    
#         # The lowest cost found among all anchors is the true lower bound for this Blue stack
#         if best_cost_for_anchor < best_cost_for_blue:
#             best_cost_for_blue = best_cost_for_anchor

#         # We take the bottleneck across all Blue stacks
#         if best_cost_for_blue > max_min_cost:
#             max_min_cost = best_cost_for_blue

#     return max_min_cost

# def heuristic_bit(board_key):
#     #all g except test 6
#     reds = []
#     blues = []
#     for i, val in enumerate(board_key):
#         if val > 0:
#             reds.append((i, val))
#         elif val < 0:
#             blues.append((i, -val))

#     if not blues:
#         return 0
#     if not reds:
#         return 100

#     num_blues = len(blues)
#     num_reds = len(reds)

#     pair_cost = [[0] * num_blues for _ in range(num_reds)]

#     for ri, (r_idx, r_h) in enumerate(reds):
#         rr, rc = divmod(r_idx, 8)
#         for bi, (b_idx, _) in enumerate(blues):
#             br, bc = divmod(b_idx, 8)
#             best = 100

#             # RIGHT: red at (br, bc-k), push blue right off col 7
#             push = 8 - bc
#             km = min(bc, r_h - push + 1)
#             if km >= 1:
#                 k = max(1, min(km, bc - rc))
#                 c = abs(rr - br) + abs(rc - (bc - k)) + 1
#                 if c < best: best = c

#             # LEFT: red at (br, bc+k), push blue left off col -1
#             push = bc + 1
#             km = min(7 - bc, r_h - push + 1)
#             if km >= 1:
#                 k = max(1, min(km, rc - bc))
#                 c = abs(rr - br) + abs(rc - (bc + k)) + 1
#                 if c < best: best = c

#             # DOWN: red at (br-k, bc), push blue down off row 7
#             push = 8 - br
#             km = min(br, r_h - push + 1)
#             if km >= 1:
#                 k = max(1, min(km, br - rr))
#                 c = abs(rr - (br - k)) + abs(rc - bc) + 1
#                 if c < best: best = c

#             # UP: red at (br+k, bc), push blue up off row -1
#             push = br + 1
#             km = min(7 - br, r_h - push + 1)
#             if km >= 1:
#                 k = max(1, min(km, rr - br))
#                 c = abs(rr - (br + k)) + abs(rc - bc) + 1
#                 if c < best: best = c

#             pair_cost[ri][bi] = best

#     # Partition lower bound
#     if num_reds == 1:
#         return max(pair_cost[0]) + num_blues - 1

#     best_h = 100
#     if num_reds == 2:
#         for mask in range(1 << num_blues):
#             max_c0 = max_c1 = 0
#             size0 = size1 = 0
#             for bi in range(num_blues):
#                 if (mask >> bi) & 1:
#                     c = pair_cost[0][bi]
#                     if c > max_c0: max_c0 = c
#                     size0 += 1
#                 else:
#                     c = pair_cost[1][bi]
#                     if c > max_c1: max_c1 = c
#                     size1 += 1
#             total = 0
#             if size0: total += max_c0 + size0 - 1
#             if size1: total += max_c1 + size1 - 1
#             if total < best_h: best_h = total
#     else:
#         from itertools import product as iprod
#         for assignment in iprod(range(num_reds), repeat=num_blues):
#             max_c = [0] * num_reds
#             size = [0] * num_reds
#             for bi, ri in enumerate(assignment):
#                 c = pair_cost[ri][bi]
#                 if c > max_c[ri]: max_c[ri] = c
#                 size[ri] += 1
#             total = sum(max_c[ri] + size[ri] - 1
#                         for ri in range(num_reds) if size[ri] > 0)
#             if total < best_h: best_h = total

#     return best_h


def heuristic_bit(board_key):
    # 1-11 ok, 8 needs 2 or more minutes
    reds = []
    blues = []
    for i, val in enumerate(board_key):
        if val > 0:
            reds.append((i, val))
        elif val < 0:
            blues.append((i, -val))

    if not blues: return 0
    if not reds: return 100

    num_blues = len(blues)
    num_reds = len(reds)
    pair_cost = [[0] * num_blues for _ in range(num_reds)]

    for ri, (r_idx, r_h) in enumerate(reds):
        rr, rc = divmod(r_idx, 8)
        for bi, (b_idx, b_h) in enumerate(blues):
            br, bc = divmod(b_idx, 8)
            # Walk/EAT baseline
            best = abs(rr - br) + abs(rc - bc)
            # Fire RIGHT: from (br, bc-k)
            km = min(r_h, bc)
            if km >= 1:
                k = max(1, min(km, bc - rc))
                c = abs(rr - br) + abs(rc - bc + k) + 1
                if c < best: best = c
            # Fire LEFT: from (br, bc+k)
            km = min(r_h, 7 - bc)
            if km >= 1:
                k = max(1, min(km, rc - bc))
                c = abs(rr - br) + abs(rc - bc - k) + 1
                if c < best: best = c
            # Fire DOWN: from (br-k, bc)
            km = min(r_h, br)
            if km >= 1:
                k = max(1, min(km, br - rr))
                c = abs(rr - br + k) + abs(rc - bc) + 1
                if c < best: best = c
            # Fire UP: from (br+k, bc)
            km = min(r_h, 7 - br)
            if km >= 1:
                k = max(1, min(km, rr - br))
                c = abs(rr - br - k) + abs(rc - bc) + 1
                if c < best: best = c
            pair_cost[ri][bi] = best

    if num_reds == 1:
        return max(pair_cost[0]) + num_blues - 1

    best_h = 100
    if num_reds == 2:
        for mask in range(1 << num_blues):
            max_c0 = max_c1 = 0
            size0 = size1 = 0
            for bi in range(num_blues):
                if (mask >> bi) & 1:
                    c = pair_cost[0][bi]
                    if c > max_c0: max_c0 = c
                    size0 += 1
                else:
                    c = pair_cost[1][bi]
                    if c > max_c1: max_c1 = c
                    size1 += 1
            total = 0
            if size0: total += max_c0 + size0 - 1
            if size1: total += max_c1 + size1 - 1
            if total < best_h: best_h = total
    else:
        from itertools import product as iprod
        for assignment in iprod(range(num_reds), repeat=num_blues):
            max_c = [0] * num_reds
            size = [0] * num_reds
            for bi, ri in enumerate(assignment):
                c = pair_cost[ri][bi]
                if c > max_c[ri]: max_c[ri] = c
                size[ri] += 1
            total = sum(max_c[ri] + size[ri] - 1
                        for ri in range(num_reds) if size[ri] > 0)
            if total < best_h: best_h = total
    return best_h