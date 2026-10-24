"""
Evaluation function for Cascade — v5: simple aggressive + tactical override.


DESIGN: Keep the eval simple and aggressive. The tactical override in MCTS
handles captures (always take safe captures immediately). The eval just needs
to steer stacks TOWARD enemies and value material advantage.


The previous versions failed because they tried to make the eval do
everything — capture detection, survivability, trade analysis. This
created perverse incentives (preferring to keep options open over acting).


This version returns to basics:
 - Material diff + ratio (moderate weight, not dominant)
 - Strong proximity bonus (steer toward enemies)
 - Eat threat bonus (reward being adjacent to eatable enemies)
 - Cascade vulnerability (unique to Cascade)
 - Positional safety (tiebreaker)


The MCTS tactical override handles: instant wins, safe captures.
The eval handles: approach, positioning, material evaluation.
"""
from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import is_valid_coord, UP, DOWN, LEFT, RIGHT


INF = 999999
OFFSETS = (UP, DOWN, LEFT, RIGHT)


EDGE_DIST = [0] * 64
for _r in range(8):
   for _c in range(8):
       EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)


FAR_EDGE_DIST = [0] * 256
for _i in range(64):
   _r, _c = _i >> 3, _i & 7
   FAR_EDGE_DIST[_i * 4 + 0] = _r
   FAR_EDGE_DIST[_i * 4 + 1] = 7 - _r
   FAR_EDGE_DIST[_i * 4 + 2] = _c
   FAR_EDGE_DIST[_i * 4 + 3] = 7 - _c


SCAN_OFFSETS = (DOWN, UP, RIGHT, LEFT)


ADJACENTS = [[] for _ in range(64)]
for _i in range(64):
   for _off in OFFSETS:
       if is_valid_coord(_i, _off):
           ADJACENTS[_i].append(_i + _off)
ADJACENTS = [tuple(a) for a in ADJACENTS]


MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
   for _b in range(64):
       MANHATTAN[_a][_b] = abs((_a>>3)-(_b>>3)) + abs((_a&7)-(_b&7))




def evaluate(board: BoardState, is_red_maximizing: bool) -> float:
   red_tokens = board.red_tokens
   blue_tokens = board.blue_tokens


   if red_tokens == 0: return -INF if is_red_maximizing else INF
   if blue_tokens == 0: return INF if is_red_maximizing else -INF


   my_flag  = RED_FLAG if is_red_maximizing else BLUE_FLAG
   opp_flag = BLUE_FLAG if is_red_maximizing else RED_FLAG
   state = board.state


   my_tokens  = red_tokens if is_red_maximizing else blue_tokens
   opp_tokens = blue_tokens if is_red_maximizing else red_tokens


   my_stacks = []
   opp_stacks = []
   for idx in range(64):
       val = state[idx]
       if val == 0: continue
       h = val & HEIGHT_MASK
       if val & my_flag: my_stacks.append((idx, h))
       else: opp_stacks.append((idx, h))


   # 1. Material
   material_diff = my_tokens - opp_tokens
   total = my_tokens + opp_tokens
   material_ratio = material_diff / total if total > 0 else 0.0


   # 2. Proximity: our eatable-height stacks approaching weak enemies
   proximity = 0.0
   for mi, mh in my_stacks:
       for oi, oh in opp_stacks:
           if mh >= oh:
               d = MANHATTAN[mi][oi]
               proximity += max(0, 8 - d) * 0.6


   # 3. Eat threats (adjacent enemies we can eat)
   eat_bonus = 0.0
   for mi, mh in my_stacks:
       for nb in ADJACENTS[mi]:
           nb_val = state[nb]
           if nb_val and (nb_val & opp_flag):
               nb_h = nb_val & HEIGHT_MASK
               if mh >= nb_h:
                   eat_bonus += nb_h * 2.0


   # 4. Cascade vulnerability
   my_vuln = 0.0
   opp_vuln = 0.0
   for idx, h in my_stacks:
       ed = EDGE_DIST[idx]
       if ed < 4:
           for d_idx in range(4):
               me = FAR_EDGE_DIST[idx*4+d_idx]
               if me >= 4: continue
               scan = SCAN_OFFSETS[d_idx]; pos = idx
               while is_valid_coord(pos, scan):
                   pos += scan; sv = state[pos]
                   if sv == 0: continue
                   if sv & opp_flag:
                       if (sv & HEIGHT_MASK) >= FAR_EDGE_DIST[pos*4+d_idx]:
                           my_vuln += h * (1.0 - me * 0.2)
                   break
   for idx, h in opp_stacks:
       ed = EDGE_DIST[idx]
       if ed < 4:
           for d_idx in range(4):
               oe = FAR_EDGE_DIST[idx*4+d_idx]
               if oe >= 4: continue
               scan = SCAN_OFFSETS[d_idx]; pos = idx
               while is_valid_coord(pos, scan):
                   pos += scan; sv = state[pos]
                   if sv == 0: continue
                   if sv & my_flag:
                       if (sv & HEIGHT_MASK) >= FAR_EDGE_DIST[pos*4+d_idx]:
                           opp_vuln += h * (1.0 - oe * 0.2)
                   break


   # 5. Positional safety
   my_safety = sum(EDGE_DIST[i]*h*0.1 for i,h in my_stacks)
   opp_safety = sum(EDGE_DIST[i]*h*0.1 for i,h in opp_stacks)


   # 6. Structure
   my_cas = sum(1 for _,h in my_stacks if h>=2)
   opp_cas = sum(1 for _,h in opp_stacks if h>=2)
   mc, oc = len(my_stacks), len(opp_stacks)


   elim = 0.0
   if oc <= 1: elim += 5.0
   elif oc == 2: elim += 2.0
   if mc <= 1: elim -= 5.0
   elif mc == 2: elim -= 2.0


   score = (
       material_diff * 6.0
       + material_ratio * 12.0
       + proximity
       + eat_bonus
       + elim
       + (my_safety - opp_safety)
       - my_vuln * 1.5
       + opp_vuln * 1.5
       + (my_cas - opp_cas) * 0.5
   )
   return score

