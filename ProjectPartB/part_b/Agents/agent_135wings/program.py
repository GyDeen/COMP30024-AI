# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: MCTS + Alpha-Beta Hybrid Agent
#
# WHY HYBRID: Pure MCTS fails in Cascade because the game is deeply
# tactical — a 5-move eat sequence is invisible to a 1-ply leaf eval.
# Pure alpha-beta is strong but explores narrowly.
#
# This hybrid uses:
#   - MCTS for strategic breadth (explore diverse game plans)
#   - Alpha-beta at each leaf node for tactical depth (see forced
#     captures, avoid traps, find multi-step winning sequences)
#   - Tactical override for obvious moves (safe captures, instant wins)


import math
import time
from referee.game import PlayerColor, Coord, Direction, \
   Action, PlaceAction, MoveAction, EatAction, CascadeAction


from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import (
   place, move, eat, cascade,
   is_valid_coord,
   BOARD_N, UP, DOWN, LEFT, RIGHT, MOVE, EAT, CASCADE
)
from .evaluation import evaluate as raw_evaluate


OFFSETS = (UP, DOWN, LEFT, RIGHT)
DIR_MAP = {UP: Direction.Up, DOWN: Direction.Down,
          LEFT: Direction.Left, RIGHT: Direction.Right}
INV_DIR_MAP = {v: k for k, v in DIR_MAP.items()}
PLACEMENT_TURNS = 8
INF = 999999
EVAL_TEMPERATURE = 8.0




# ──────────────────────────────────────────────
# Move generation
# ──────────────────────────────────────────────
def generate_actions_ordered(board, is_red):
   """EATs at end (popped first by expand). Within EATs, biggest last."""
   my_flag = RED_FLAG if is_red else BLUE_FLAG
   opp_flag = BLUE_FLAG if is_red else RED_FLAG
   moves = []; cascades = []; eats = []
   for idx in range(64):
       val = board.state[idx]
       if not (val & my_flag): continue
       h = val & HEIGHT_MASK
       for offset in OFFSETS:
           if not is_valid_coord(idx, offset): continue
           tv = board.state[idx + offset]
           if tv == 0: moves.append((MOVE, idx, offset))
           elif tv & my_flag: moves.append((MOVE, idx, offset))
           elif tv & opp_flag:
               th = tv & HEIGHT_MASK
               if h >= th: eats.append((EAT, idx, offset, th))
       if h >= 2:
           for offset in OFFSETS:
               cascades.append((CASCADE, idx, offset))
   eats.sort(key=lambda x: x[3])
   return moves + cascades + [(a[0], a[1], a[2]) for a in eats]




def generate_actions_flat(board, is_red):
   """Unordered, for alpha-beta and rollouts."""
   my_flag = RED_FLAG if is_red else BLUE_FLAG
   opp_flag = BLUE_FLAG if is_red else RED_FLAG
   actions = []
   for idx in range(64):
       val = board.state[idx]
       if not (val & my_flag): continue
       h = val & HEIGHT_MASK
       for offset in OFFSETS:
           if not is_valid_coord(idx, offset): continue
           tv = board.state[idx + offset]
           if tv == 0: actions.append((MOVE, idx, offset))
           elif tv & my_flag: actions.append((MOVE, idx, offset))
           elif tv & opp_flag:
               if h >= (tv & HEIGHT_MASK): actions.append((EAT, idx, offset))
       if h >= 2:
           for offset in OFFSETS:
               actions.append((CASCADE, idx, offset))
   return actions




def generate_placement_actions(board, is_red, pc):
   opp_flag = BLUE_FLAG if is_red else RED_FLAG
   actions = []
   for idx in range(64):
       if board.state[idx] != 0: continue
       if pc > 0:
           adj = False
           for offset in OFFSETS:
               if is_valid_coord(idx, offset) and (board.state[idx + offset] & opp_flag):
                   adj = True; break
           if adj: continue
       actions.append(idx)
   return actions




def apply_action(board, action_type, start_idx, offset):
   if action_type == MOVE: return move(board, start_idx, offset)
   elif action_type == EAT: return eat(board, start_idx, offset)
   elif action_type == CASCADE: return cascade(board, start_idx, offset)
   return board




# ──────────────────────────────────────────────
# Tactical decision layer
# ──────────────────────────────────────────────
def tactical_decision(board, is_red):
   """
   Check for moves that should be played immediately:
   1. Instant win (eliminates all opponent tokens)
   2. Safe capture (eat with no risk of being eaten back)
   Returns action tuple or None.
   """
   my_flag = RED_FLAG if is_red else BLUE_FLAG
   opp_flag = BLUE_FLAG if is_red else RED_FLAG
   state = board.state


   best_capture = None
   best_capture_gain = 0


   for idx in range(64):
       val = state[idx]
       if not (val & my_flag): continue
       h = val & HEIGHT_MASK
       for offset in OFFSETS:
           if not is_valid_coord(idx, offset): continue
           tv = state[idx + offset]
           if not (tv & opp_flag): continue
           th = tv & HEIGHT_MASK
           if h < th: continue


           dest = idx + offset
           nb = apply_action(board, EAT, idx, offset)


           # Instant win?
           opp_left = nb.blue_tokens if is_red else nb.red_tokens
           if opp_left == 0:
               return (EAT, idx, offset)


           # Safe capture? (no enemy adjacent to destination can eat us)
           safe = True
           for nb_off in OFFSETS:
               if not is_valid_coord(dest, nb_off): continue
               nv = nb.state[dest + nb_off]
               if nv and (nv & opp_flag) and (nv & HEIGHT_MASK) >= h:
                   safe = False; break


           if safe and th > best_capture_gain:
               best_capture_gain = th
               best_capture = (EAT, idx, offset)


   return best_capture




# ──────────────────────────────────────────────
# Alpha-beta leaf evaluator (for MCTS leaf nodes)
# ──────────────────────────────────────────────
def _ab_eval_raw(board, depth, alpha, beta, is_red_turn, is_red_max):
   """
   Alpha-beta search returning raw eval score for is_red_max.
   Used at MCTS leaf nodes to add tactical depth.
   """
   if board.red_tokens == 0:
       return -INF if is_red_max else INF
   if board.blue_tokens == 0:
       return INF if is_red_max else -INF
   if depth <= 0:
       return raw_evaluate(board, is_red_max)


   actions = generate_actions_flat(board, is_red_turn)
   if not actions:
       return 0.0  # stalemate


   is_max = (is_red_turn == is_red_max)


   if is_max:
       val = -INF
       for a in actions:
           v = _ab_eval_raw(apply_action(board, a[0], a[1], a[2]),
                            depth - 1, alpha, beta, not is_red_turn, is_red_max)
           if v > val: val = v
           if val > alpha: alpha = val
           if alpha >= beta: break
       return val
   else:
       val = INF
       for a in actions:
           v = _ab_eval_raw(apply_action(board, a[0], a[1], a[2]),
                            depth - 1, alpha, beta, not is_red_turn, is_red_max)
           if v < val: val = v
           if val < beta: beta = val
           if alpha >= beta: break
       return val




def ab_leaf_eval(board, is_red_turn, root_is_red, depth=3):
   """
   Evaluate a leaf node using alpha-beta search.
   Returns win probability in [0, 1] for root_is_red.
  
   depth=3 gives good tactical awareness (~20ms per eval on a
   mid-game position with ~30 actions per side). With MCTS doing
   ~500-1500 sims in a 2s budget, total time stays under budget.
   """
   raw = _ab_eval_raw(board, depth, -INF, INF, is_red_turn, root_is_red)


   # Map to [0, 1]
   if raw >= INF - 100: return 1.0
   if raw <= -INF + 100: return 0.0
   x = max(-20.0, min(20.0, raw / EVAL_TEMPERATURE))
   return 1.0 / (1.0 + math.exp(-x))




# ──────────────────────────────────────────────
# MCTS Node — negamax style
# ──────────────────────────────────────────────
class MCTSNode:
   __slots__ = ['board', 'is_red_turn', 'parent', 'action',
                'children', 'untried_actions', 'visits', 'value']


   def __init__(self, board, is_red_turn, parent=None, action=None):
       self.board = board
       self.is_red_turn = is_red_turn
       self.parent = parent
       self.action = action
       self.children = []
       self.untried_actions = None
       self.visits = 0
       self.value = 0.0


   def is_terminal(self):
       return self.board.red_tokens == 0 or self.board.blue_tokens == 0


   def is_fully_expanded(self):
       if self.untried_actions is None:
           self.untried_actions = generate_actions_ordered(
               self.board, self.is_red_turn)
       return len(self.untried_actions) == 0


   def best_child(self, c_param):
       log_parent = math.log(self.visits)
       best_score = -1.0; best_node = None
       for child in self.children:
           if child.visits == 0: return child
           exploit = 1.0 - (child.value / child.visits)
           explore = c_param * math.sqrt(log_parent / child.visits)
           uct = exploit + explore
           if uct > best_score: best_score = uct; best_node = child
       return best_node


   def most_visited_child(self):
       best = None; bv = -1
       for ch in self.children:
           if ch.visits > bv: bv = ch.visits; best = ch
       return best


   def expand(self):
       if self.untried_actions is None:
           self.untried_actions = generate_actions_ordered(
               self.board, self.is_red_turn)
       action = self.untried_actions.pop()
       new_board = apply_action(self.board, action[0], action[1], action[2])
       child = MCTSNode(new_board, not self.is_red_turn,
                        parent=self, action=action)
       self.children.append(child)
       return child


   def backpropagate(self, result):
       node = self; value = result
       while node is not None:
           node.visits += 1
           node.value += value
           value = 1.0 - value
           node = node.parent




# ──────────────────────────────────────────────
# MCTS Searcher (hybrid with AB leaf)
# ──────────────────────────────────────────────
class MCTSSearcher:
   def __init__(self, c_param=0.7, leaf_depth=3):
       """
       c_param: UCT exploration constant.
       leaf_depth: alpha-beta depth at each leaf node.
           - 2: ~100ms budget, ~800 sims/2s — lightweight
           - 3: ~20ms budget, ~2000 sims/2s — good balance
           - 4: ~100ms budget, ~400 sims/2s — heavy tactical
       """
       self.c_param = c_param
       self.leaf_depth = leaf_depth


   def search(self, board, is_red_turn, time_budget):
       # 1. Tactical override
       forced = tactical_decision(board, is_red_turn)
       if forced is not None:
           return forced, 1, 1


       all_actions = generate_actions_ordered(board, is_red_turn)
       if not all_actions: return None, 0, 0
       if len(all_actions) == 1: return all_actions[0], 1, 1


       # 2. MCTS with AB leaf evaluation
       root = MCTSNode(board, is_red_turn)
       start_time = time.process_time()
       simulations = 0


       while True:
           if simulations & 31 == 0 and simulations > 0:
               if time.process_time() - start_time > time_budget:
                   break


           node = root
           while not node.is_terminal() and node.is_fully_expanded():
               node = node.best_child(self.c_param)
           if not node.is_terminal() and not node.is_fully_expanded():
               node = node.expand()


           # Alpha-beta at leaf for tactical depth
           result = ab_leaf_eval(node.board, node.is_red_turn,
                                 node.is_red_turn, self.leaf_depth)
           node.backpropagate(result)
           simulations += 1


       if not root.children:
           return (all_actions[0], 0, simulations)
       best = root.most_visited_child()
       return best.action, best.visits, simulations




# ──────────────────────────────────────────────
# Placement / helpers
# ──────────────────────────────────────────────
PREFERRED_PLACEMENTS = sorted(range(64),
   key=lambda idx: abs(idx // 8 - 3.5) + abs(idx % 8 - 3.5))


def choose_placement(board, is_red, pc):
   legal = set(generate_placement_actions(board, is_red, pc))
   if not legal: raise RuntimeError("No legal placements")
   for idx in PREFERRED_PLACEMENTS:
       if idx in legal: return idx
   return next(iter(legal))


def idx_to_coord(idx): return Coord(idx // 8, idx % 8)
def coord_to_idx(coord): return coord.r * 8 + coord.c


def action_tuple_to_action(action_type, start_idx, offset):
   coord = idx_to_coord(start_idx)
   direction = DIR_MAP[offset]
   if action_type == MOVE: return MoveAction(coord, direction)
   elif action_type == EAT: return EatAction(coord, direction)
   elif action_type == CASCADE: return CascadeAction(coord, direction)
   raise ValueError(f"Unknown action type: {action_type}")




# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────
class Agent:
   def __init__(self, color: PlayerColor, **referee: dict):
       self._color = color
       self._is_red = (color == PlayerColor.RED)
       self._board = BoardState()
       self._placement_count = 0
       self._turn_count = 0
       # leaf_depth=3: each MCTS sim does a 3-ply alpha-beta search
       # ~2000 sims per 2s budget = MCTS breadth + AB depth
       self._searcher = MCTSSearcher(c_param=0.7, leaf_depth=3)


   def action(self, **referee: dict) -> Action:
       time_remaining = referee.get("time_remaining", None)
       if self._placement_count < PLACEMENT_TURNS:
           idx = choose_placement(self._board, self._is_red, self._placement_count)
           return PlaceAction(idx_to_coord(idx))


       if time_remaining is not None: remaining = time_remaining
       else: remaining = 170.0


       if remaining < 10: budget = remaining * 0.10
       elif remaining < 30: budget = remaining * 0.06
       elif self._turn_count < 5: budget = min(1.5, remaining * 0.02)
       else: budget = min(2.5, remaining * 0.04)
       budget = max(0.05, budget)


       best, visits, sims = self._searcher.search(
           self._board, self._is_red, budget)
       if best is None:
           actions = generate_actions_flat(self._board, self._is_red)
           if actions: best = actions[0]
           else: raise RuntimeError("No legal actions")
       return action_tuple_to_action(best[0], best[1], best[2])


   def update(self, color: PlayerColor, action: Action, **referee: dict):
       is_red_actor = (color == PlayerColor.RED)
       match action:
           case PlaceAction(coord):
               self._board = place(self._board, coord_to_idx(coord), is_red_actor)
               self._placement_count += 1
           case MoveAction(coord, direction):
               self._board = apply_action(self._board, MOVE, coord_to_idx(coord), INV_DIR_MAP[direction])
           case EatAction(coord, direction):
               self._board = apply_action(self._board, EAT, coord_to_idx(coord), INV_DIR_MAP[direction])
           case CascadeAction(coord, direction):
               self._board = apply_action(self._board, CASCADE, coord_to_idx(coord), INV_DIR_MAP[direction])
       if color == self._color:
           self._turn_count += 1

