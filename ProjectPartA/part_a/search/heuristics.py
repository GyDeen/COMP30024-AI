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

    num_blues = len(blues)
    num_reds = len(reds)
    # Precompute the cost for each red-blue pair, considering both direct distance and potential cascade, and store it in a matrix for quick lookup during assignment
    pair_cost = [[0] * num_blues for _ in range(num_reds)]


    # Find the best cost for each red-blue pair, considering both direct distance and potential cascade
    for ri, (r_idx, r_h) in enumerate(reds):
        rr, rc = divmod(r_idx, 8)
        for bi, (b_idx, b_h) in enumerate(blues):
            br, bc = divmod(b_idx, 8)
            # Walk/EAT baseline
            best = abs(rr - br) + abs(rc - bc)
            # Cascade RIGHT: from (br, bc-k)
            km = min(r_h, bc)
            if km >= 1:
                k = max(1, min(km, bc - rc))
                c = abs(rr - br) + abs(rc - bc + k) + 1
                if c < best: best = c
            # Cascade LEFT: from (br, bc+k)
            km = min(r_h, 7 - bc)
            if km >= 1:
                k = max(1, min(km, rc - bc))
                c = abs(rr - br) + abs(rc - bc - k) + 1
                if c < best: best = c
            # Cascade DOWN: from (br-k, bc)
            km = min(r_h, br)
            if km >= 1:
                k = max(1, min(km, br - rr))
                c = abs(rr - br + k) + abs(rc - bc) + 1
                if c < best: best = c
            # Cascade UP: from (br+k, bc)
            km = min(r_h, 7 - br)
            if km >= 1:
                k = max(1, min(km, rr - br))
                c = abs(rr - br - k) + abs(rc - bc) + 1
                if c < best: best = c
            pair_cost[ri][bi] = best

    # If only one red, the absolute lowest cost is to eat the furtherest blue plus the cost of directly eating the rest
    if num_reds == 1:
        return max(pair_cost[0]) + num_blues - 1

    best_h = 100
    if num_reds == 2:
        # Exam all the possible assignment of blues to the two reds, and find the one with the lowest cost
        for mask in range(1 << num_blues):
            max_c0 = max_c1 = 0
            size0 = size1 = 0
            # For each blue, assign it to red 0 or red 1 based on the mask, and calculate the max cost and size for each red
            for bi in range(num_blues):
                # Assigned to red 0
                if (mask >> bi) & 1:
                    c = pair_cost[0][bi]
                    # If current assignment is the highest cost for red 0
                    if c > max_c0: max_c0 = c
                    size0 += 1
                # Assigned to red 1
                else:
                    c = pair_cost[1][bi]
                    # If current assignment is the highest cost for red 1
                    if c > max_c1: max_c1 = c
                    size1 += 1
            total = 0
            if size0: total += max_c0 + size0 - 1
            if size1: total += max_c1 + size1 - 1
            if total < best_h: best_h = total
    else:
        from itertools import product as iprod
        # Try all the possible assignment of blues to reds, and find the one with the lowest cost
        for assignment in iprod(range(num_reds), repeat=num_blues):
            max_c = [0] * num_reds
            size = [0] * num_reds
            for bi, ri in enumerate(assignment):
                c = pair_cost[ri][bi]
                # If current assignment is the highest cost for red ri
                if c > max_c[ri]: max_c[ri] = c
                size[ri] += 1
            total = sum(max_c[ri] + size[ri] - 1
                        for ri in range(num_reds) if size[ri] > 0)
            if total < best_h: best_h = total
    return best_h