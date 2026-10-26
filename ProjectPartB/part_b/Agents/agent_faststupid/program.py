# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: MCTS Agent for Cascade — v3
#
# KEY CHANGES:
#   1. QUIESCENCE IN EVAL: eval checks for immediate safe eats and
#      evaluates the post-eat position. This is THE critical fix —
#      it lets MCTS "see" that being adjacent to a killable enemy
#      is worth points because the eat is guaranteed next turn.
#
#   2. FORCED APPROACH: When we're the tallest stack on the board,
#      the tactical layer FORCES approach moves toward killable
#      enemies. No more retreating with B9 when R3 is right there.
#
#   3. ROOT ANTI-RETREAT: When tallest, moves that increase distance
#      to the nearest killable enemy are filtered from root actions.
#      MCTS literally cannot choose to retreat.
#
#   4. REPEAT PENALTY: Positions seen before get penalized in eval,
#      breaking the loop/draw pattern.
#
#   5. CORNER DEATH: eval heavily penalizes being near corners with
#      enemies approaching.

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

EDGE_DIST = [0] * 64
for _r in range(8):
    for _c in range(8):
        EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)

MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        MANHATTAN[_a][_b] = abs((_a>>3)-(_b>>3)) + abs((_a&7)-(_b&7))


# ──────────────────────────────────────────────
# Move generation
# ──────────────────────────────────────────────
def generate_actions_ordered(board, is_red):
    """EATs at end (popped first by expand)."""
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
# Board analysis helpers
# ──────────────────────────────────────────────
def _get_stacks(board, is_red):
    """Return list of (idx, height) for the given side."""
    flag = RED_FLAG if is_red else BLUE_FLAG
    stacks = []
    for idx in range(64):
        val = board.state[idx]
        if val and (val & flag):
            stacks.append((idx, val & HEIGHT_MASK))
    return stacks


def _min_kill_distance(board, is_red):
    """
    Min manhattan distance from any of our stacks to any enemy
    that we're strictly taller than.
    Returns (min_dist, our_stack_idx, enemy_idx) or (99, -1, -1).
    """
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)
    best_d = 99; best_mi = -1; best_oi = -1
    for mi, mh in my_stacks:
        for oi, oh in opp_stacks:
            if mh > oh:
                d = MANHATTAN[mi][oi]
                if d < best_d:
                    best_d = d; best_mi = mi; best_oi = oi
    return best_d, best_mi, best_oi


def _we_are_tallest(board, is_red):
    """True if our tallest stack is strictly taller than all enemies."""
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)
    if not my_stacks or not opp_stacks: return False
    my_max = max(h for _, h in my_stacks)
    opp_max = max(h for _, h in opp_stacks)
    return my_max > opp_max


# ──────────────────────────────────────────────
# Tactical decision layer
# ──────────────────────────────────────────────
def tactical_decision(board, is_red, position_history):
    """
    1. Instant win (eat or cascade that kills all enemies)
    2. Safe eat (capture where enemy can't recapture)
    3. FORCED APPROACH: when we're the tallest stack, move toward
       the nearest killable enemy.
    4. CASCADE-MERGE: when no height advantage, cascade a stack toward
       a friendly to gain height (h3 + cascade → h4 beats enemy h3).
    5. Profitable cascade push-off (gain >= 3)
    6. Forced flee toward center when shorter.
    """
    my_flag = RED_FLAG if is_red else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red else RED_FLAG
    state = board.state
    my_before = board.red_tokens if is_red else board.blue_tokens
    opp_before = board.blue_tokens if is_red else board.red_tokens

    best_eat = None
    best_eat_gain = 0
    best_cascade = None
    best_cascade_gain = 0

    # ── Phase 1: Check all EATs ──
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

            nb = apply_action(board, EAT, idx, offset)
            opp_left = nb.blue_tokens if is_red else nb.red_tokens
            if opp_left == 0:
                return (EAT, idx, offset)  # instant win

            dest = idx + offset
            safe = True
            for nb_off in OFFSETS:
                if not is_valid_coord(dest, nb_off): continue
                nv = nb.state[dest + nb_off]
                if nv and (nv & opp_flag) and (nv & HEIGHT_MASK) >= h:
                    safe = False; break

            if safe and th > best_eat_gain:
                best_eat_gain = th
                best_eat = (EAT, idx, offset)

    if best_eat is not None:
        return best_eat

    # ── Phase 2: Forced approach when tallest ──
    # When we're strictly taller than all enemies, find the move
    # that minimizes distance to the nearest killable enemy.
    # This prevents the "B9 runs from R3" disaster.
    if _we_are_tallest(board, is_red):
        cur_dist, our_idx, target_idx = _min_kill_distance(board, is_red)

        if cur_dist >= 2 and our_idx >= 0:
            # Find the move of our tallest stack that gets closest
            # to the target while staying safe (no equal-or-taller
            # enemy adjacent at destination).
            my_stacks = _get_stacks(board, is_red)
            opp_stacks = _get_stacks(board, not is_red)
            our_h = state[our_idx] & HEIGHT_MASK

            best_approach = None
            best_new_dist = cur_dist

            for offset in OFFSETS:
                if not is_valid_coord(our_idx, offset): continue
                dest = our_idx + offset
                dest_val = state[dest]

                # Can we move here? (empty or friendly)
                if dest_val != 0 and (dest_val & opp_flag):
                    continue  # that's an eat, handled in phase 1
                if dest_val != 0 and not (dest_val & my_flag):
                    continue

                # New height after move (merge if friendly)
                new_h = our_h
                if dest_val and (dest_val & my_flag):
                    new_h = our_h + (dest_val & HEIGHT_MASK)

                # Distance to target after move
                new_dist = MANHATTAN[dest][target_idx]
                if new_dist >= best_new_dist:
                    continue  # not closer

                # Safety check: no enemy at dest can eat us
                safe = True
                for check_off in OFFSETS:
                    if not is_valid_coord(dest, check_off): continue
                    cv = state[dest + check_off]
                    if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                        safe = False; break

                # Edge safety: don't walk to edge unless chasing
                dest_ed = EDGE_DIST[dest]
                if dest_ed <= 1 and new_dist > 2:
                    continue  # don't go to edge unless target is close

                if safe and new_dist < best_new_dist:
                    # Check for repetition
                    nb = apply_action(board, MOVE, our_idx, offset)
                    nb_hash = bytes(nb.state)
                    if position_history.count(nb_hash) >= 2:
                        continue  # would cause threefold
                    best_new_dist = new_dist
                    best_approach = (MOVE, our_idx, offset)

            if best_approach is not None:
                return best_approach

    # ── Phase 3: Cascade-merge for height advantage ──
    # If all stacks are h3 (no height advantage), cascade one friendly
    # toward another to create h4+. This is the test agent's key tactic:
    # R3 cascade toward R3 → R1, R4, R1. Now R4 can safely eat any B3.
    #
    # Requirements:
    #   - We have NO height advantage (our max_h <= enemy max_h)
    #   - The cascade creates a taller max stack
    #   - We don't lose tokens off the edge (or loss is minimal)
    #   - The debris (h=1 stacks) are NOT near enemies
    #   - The new tall stack is reasonably centered (not on edge)
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)
    
    if my_stacks and opp_stacks:
        my_max_h_before = max(h for _, h in my_stacks)
        opp_max_h = max(h for _, h in opp_stacks)
        
        if my_max_h_before <= opp_max_h:
            # No height advantage — cascade-merge could help
            best_merge = None
            best_merge_score = 0
            
            for idx in range(64):
                val = state[idx]
                if not (val & my_flag): continue
                h = val & HEIGHT_MASK
                if h < 2: continue  # need at least h2 to cascade
                
                for offset in OFFSETS:
                    nb = apply_action(board, CASCADE, idx, offset)
                    nb_my = nb.red_tokens if is_red else nb.blue_tokens
                    nb_opp = nb.blue_tokens if is_red else nb.red_tokens
                    
                    # Don't lose too many tokens
                    my_lost = my_before - nb_my
                    if my_lost > 1: continue  # at most lose 1 off edge
                    
                    # Must not destroy any opponent (that's cascade push-off, Phase 4)
                    # This phase is purely about merging our own stacks
                    opp_lost = opp_before - nb_opp
                    
                    # Check if we gained height
                    nb_my_stacks = []
                    for ci in range(64):
                        cv = nb.state[ci]
                        if cv and (cv & my_flag):
                            nb_my_stacks.append((ci, cv & HEIGHT_MASK))
                    if not nb_my_stacks: continue
                    
                    nb_max_h = max(sh for _, sh in nb_my_stacks)
                    if nb_max_h <= my_max_h_before: continue  # no height gain
                    
                    # Check debris safety: no h=1 stack near any enemy
                    debris_safe = True
                    for ci, ch in nb_my_stacks:
                        if ch > 1: continue  # only check debris
                        for oi, oh in opp_stacks:
                            if MANHATTAN[ci][oi] <= 2:
                                debris_safe = False
                                break
                        if not debris_safe: break
                    if not debris_safe: continue
                    
                    # Check the new tall stack is centered (not on edge)
                    tall_ed = 0
                    for ci, ch in nb_my_stacks:
                        if ch == nb_max_h:
                            tall_ed = EDGE_DIST[ci]
                            break
                    if tall_ed <= 1: continue  # tall stack on edge = bad
                    
                    # Score the cascade-merge
                    height_gain = nb_max_h - my_max_h_before
                    merge_score = (
                        height_gain * 10.0  # primary: how much taller
                        + tall_ed * 2.0     # prefer centered tall stack
                        + opp_lost * 5.0    # bonus if also destroyed enemy
                        - my_lost * 8.0     # penalty for lost tokens
                    )
                    
                    if merge_score > best_merge_score:
                        best_merge_score = merge_score
                        best_merge = (CASCADE, idx, offset)
            
            if best_merge is not None:
                return best_merge

    # ── Phase 4: Profitable cascade push-off ──
    opp_near_edge = any(
        state[idx] and (state[idx] & opp_flag) and EDGE_DIST[idx] <= 2
        for idx in range(64)
    )
    if opp_near_edge:
        for idx in range(64):
            val = state[idx]
            if not (val & my_flag): continue
            h = val & HEIGHT_MASK
            if h < 2: continue
            for offset in OFFSETS:
                nb = apply_action(board, CASCADE, idx, offset)
                opp_left = nb.blue_tokens if is_red else nb.red_tokens
                my_left = nb.red_tokens if is_red else nb.blue_tokens
                if opp_left == 0 and my_left > 0:
                    return (CASCADE, idx, offset)
                opp_destroyed = opp_before - opp_left
                my_lost = my_before - my_left
                if opp_destroyed == 0: continue
                my_at_risk = 0
                for ci in range(64):
                    cv = nb.state[ci]
                    if not cv or not (cv & my_flag): continue
                    ch = cv & HEIGHT_MASK
                    for oi in range(64):
                        ov = nb.state[oi]
                        if not ov or not (ov & opp_flag): continue
                        oh = ov & HEIGHT_MASK
                        if oh < ch: continue
                        d = MANHATTAN[ci][oi]
                        if d <= 1:   my_at_risk += ch;       break
                        elif d <= 2: my_at_risk += ch * 0.7; break
                        elif d <= 3: my_at_risk += ch * 0.3; break
                adjusted = opp_destroyed - my_lost - my_at_risk
                if adjusted >= 3 and adjusted > best_cascade_gain:
                    best_cascade_gain = adjusted
                    best_cascade = (CASCADE, idx, offset)

    if best_cascade is not None:
        return best_cascade

    # ── Phase 4: Forced flee toward center when shorter ──
    # When we're the SHORTER stack (enemy is taller), we MUST flee.
    # But not to just any square — TOWARD THE CENTER.
    # 
    # Corner = death by cascade push-off.
    # Edge = limited escape routes.
    # Center = maximum mobility, hardest to corner.
    #
    # This fires when we have 1-2 stacks and any enemy is taller.
    # It overrides MCTS because MCTS doesn't understand that
    # "distance from enemy" and "centrality" BOTH matter.
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)
    
    if my_stacks and opp_stacks and len(my_stacks) <= 2:
        my_max_h = max(h for _, h in my_stacks)
        opp_max_h = max(h for _, h in opp_stacks)
        
        if opp_max_h > my_max_h:
            # We're shorter — need to flee intelligently.
            # Move the most-threatened stack toward center.
            best_flee = None
            best_flee_score = -INF
            
            for mi, mh in my_stacks:
                cur_ed = EDGE_DIST[mi]
                # Find nearest taller enemy
                min_threat_dist = 99
                for oi, oh in opp_stacks:
                    if oh > mh:
                        d = MANHATTAN[mi][oi]
                        if d < min_threat_dist:
                            min_threat_dist = d
                
                if min_threat_dist > 5 and cur_ed >= 2:
                    continue  # safe enough, don't force
                
                for offset in OFFSETS:
                    if not is_valid_coord(mi, offset): continue
                    dest = mi + offset
                    dest_val = state[dest]
                    
                    # Can we move here?
                    if dest_val != 0 and (dest_val & opp_flag):
                        continue  # enemy square
                    if dest_val != 0 and not (dest_val & my_flag):
                        continue
                    
                    new_h = mh
                    if dest_val and (dest_val & my_flag):
                        new_h = mh + (dest_val & HEIGHT_MASK)
                    
                    # Is destination safe? (no taller enemy adjacent)
                    safe = True
                    for check_off in OFFSETS:
                        if not is_valid_coord(dest, check_off): continue
                        cv = state[dest + check_off]
                        if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                            safe = False; break
                    if not safe:
                        continue
                    
                    dest_ed = EDGE_DIST[dest]
                    
                    # Distance from nearest taller enemy after move
                    new_threat_dist = 99
                    for oi, oh in opp_stacks:
                        if oh > new_h:
                            d = MANHATTAN[dest][oi]
                            if d < new_threat_dist:
                                new_threat_dist = d
                    
                    # Score: distance from enemy + centrality
                    # Centrality is weighted 3x because corners kill.
                    score = new_threat_dist * 1.5 + dest_ed * 3.0
                    
                    # Bonus for merging (makes us taller = safer)
                    if dest_val and (dest_val & my_flag):
                        score += (dest_val & HEIGHT_MASK) * 2.0
                    
                    # Check for repetition
                    nb = apply_action(board, MOVE, mi, offset)
                    nb_hash = bytes(nb.state)
                    if position_history.count(nb_hash) >= 2:
                        score -= 100  # avoid threefold
                    
                    if score > best_flee_score:
                        best_flee_score = score
                        best_flee = (MOVE, mi, offset)
            
            if best_flee is not None:
                return best_flee

    return None


# ──────────────────────────────────────────────
# Eval → [0, 1] with repeat penalty
# ──────────────────────────────────────────────
def eval_to_win_prob(board, player_is_red, position_history=None):
    r, b = board.red_tokens, board.blue_tokens
    if r == 0: return 0.0 if player_is_red else 1.0
    if b == 0: return 1.0 if player_is_red else 0.0
    raw = raw_evaluate(board, player_is_red)

    # Penalize repeated positions
    if position_history is not None:
        board_hash = bytes(board.state)
        repeats = position_history.count(board_hash)
        if repeats >= 1:
            raw -= repeats * 3.0

    x = max(-20.0, min(20.0, raw / EVAL_TEMPERATURE))
    return 1.0 / (1.0 + math.exp(-x))


# ──────────────────────────────────────────────
# MCTS Node
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
# MCTS Searcher v3
# ──────────────────────────────────────────────
class MCTSSearcher:
    def __init__(self, c_param=0.7):
        self.c_param = c_param

    def _filter_root_actions(self, board, is_red, actions, position_history):
        """
        Root-level action filtering:
        1. Remove pointless cascades
        2. When tallest: REMOVE moves that increase distance to killable enemies
        3. Penalize repeats
        """
        my_flag = RED_FLAG if is_red else BLUE_FLAG
        opp_flag = BLUE_FLAG if is_red else RED_FLAG
        my_before = board.red_tokens if is_red else board.blue_tokens
        opp_before = board.blue_tokens if is_red else board.red_tokens
        tallest = _we_are_tallest(board, is_red)
        cur_kill_dist = _min_kill_distance(board, is_red)[0] if tallest else 99

        scored = []
        for act in actions:
            priority = 0

            # Filter pointless cascades
            if act[0] == CASCADE:
                nb = apply_action(board, act[0], act[1], act[2])
                my_after = nb.red_tokens if is_red else nb.blue_tokens
                opp_after = nb.blue_tokens if is_red else nb.red_tokens
                net = (opp_before - opp_after) - (my_before - my_after)
                if net <= 0:
                    continue
                if net < 3:
                    debris_near = False
                    for ci in range(64):
                        cv = nb.state[ci]
                        if not cv or not (cv & my_flag): continue
                        if (cv & HEIGHT_MASK) > 1: continue
                        for oi in range(64):
                            ov = nb.state[oi]
                            if ov and (ov & opp_flag) and MANHATTAN[ci][oi] <= 3:
                                debris_near = True; break
                        if debris_near: break
                    if debris_near: continue

            nb = apply_action(board, act[0], act[1], act[2])
            nb_hash = bytes(nb.state)
            repeats = position_history.count(nb_hash)

            # Score by action type
            if act[0] == EAT:
                captured = opp_before - (nb.blue_tokens if is_red else nb.red_tokens)
                priority = 200 + captured * 10
            elif act[0] == CASCADE:
                priority = 100
            elif act[0] == MOVE:
                priority = 10

            # When tallest: reward moves that close distance, penalize retreat
            if tallest and act[0] == MOVE:
                new_kill_dist = _min_kill_distance(nb, is_red)[0]
                if new_kill_dist < cur_kill_dist:
                    priority += 50  # approaching = good
                elif new_kill_dist > cur_kill_dist:
                    priority -= 80  # retreating = very bad (but not filtered)

            # Heavy penalty for repeats
            priority -= repeats * 60

            scored.append((act, priority))

        if not scored:
            return actions

        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in scored]

    def search(self, board, is_red_turn, time_budget, position_history=None):
        if position_history is None:
            position_history = []

        forced = tactical_decision(board, is_red_turn, position_history)
        if forced is not None:
            return forced, 1, 1

        all_actions = generate_actions_ordered(board, is_red_turn)
        if not all_actions: return None, 0, 0
        if len(all_actions) == 1: return all_actions[0], 1, 1

        filtered = self._filter_root_actions(
            board, is_red_turn, all_actions, position_history)

        root = MCTSNode(board, is_red_turn)
        root.untried_actions = list(filtered)

        start_time = time.process_time()
        simulations = 0

        while True:
            if simulations & 127 == 0 and simulations > 0:
                if time.process_time() - start_time > time_budget:
                    break

            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_param)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            result = eval_to_win_prob(
                node.board, node.is_red_turn, position_history)
            node.backpropagate(result)
            simulations += 1

        if not root.children:
            return (filtered[0] if filtered else all_actions[0], 0, simulations)
        best = root.most_visited_child()
        return best.action, best.visits, simulations


# ──────────────────────────────────────────────
# Placement strategy
# ──────────────────────────────────────────────
def choose_placement(board, is_red, placement_count):
    """
    Strategic placement:
    
    Turn 1 (placement_count=0, first player): Center cell. (3,3)/(4,4) etc.
    
    Turn 2 (placement_count=1, second player): Diagonal to opponent's first
    piece, toward center. This blocks 2 of opponent's adjacent cells for
    their next placement (adjacency restriction kicks in at placement_count>0).
    E.g. opponent at (3,3) → place at (4,4) → blocks (3,4) and (4,3).
    
    Subsequent turns: Create cascade-merge pairs. Place at distance 2 from
    a friendly stack in a line (same row or column), so later we can cascade
    one into the other to gain height advantage.
    E.g. R3 at (3,3) → place at (3,5) → later cascade R3(3,3) right →
    token lands on R3(3,5) → R4 at (3,5). Now we're taller than any B3.
    
    Always: avoid edges, prefer center, don't cluster too tight.
    """
    legal = generate_placement_actions(board, is_red, placement_count)
    if not legal:
        raise RuntimeError("No legal placements")

    my_flag = RED_FLAG if is_red else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red else RED_FLAG
    state = board.state
    legal_set = set(legal)

    # ── First placement: center ──
    if placement_count == 0:
        # Pure center cells, prefer (3,3) or (4,4)
        for idx in [27, 36, 28, 35]:  # (3,3), (4,4), (3,4), (4,3)
            if idx in legal_set:
                return idx

    # ── Second player's first turn: diagonal block ──
    if placement_count == 1:
        # Find opponent's piece
        opp_idx = -1
        for i in range(64):
            if state[i] and (state[i] & opp_flag):
                opp_idx = i
                break

        if opp_idx >= 0:
            opp_r, opp_c = opp_idx >> 3, opp_idx & 7
            # Diagonal positions, sorted by distance to center
            diags = []
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = opp_r + dr, opp_c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    idx = nr * 8 + nc
                    if idx in legal_set:
                        center_dist = abs(nr - 3.5) + abs(nc - 3.5)
                        diags.append((center_dist, idx))
            diags.sort()
            if diags:
                return diags[0][1]

    # ── Subsequent placements: cascade-merge pairs ──
    my_positions = []
    for i in range(64):
        if state[i] and (state[i] & my_flag):
            my_positions.append(i)

    opp_positions = []
    for i in range(64):
        if state[i] and (state[i] & opp_flag):
            opp_positions.append(i)

    best_idx = legal[0]
    best_score = -999

    for idx in legal:
        score = 0.0
        r, c = idx >> 3, idx & 7
        ed = EDGE_DIST[idx]

        # 1. Avoid edges (crucial — edge stacks die to cascades)
        score += ed * 4.0

        # 2. Center bias
        score += max(0, 4 - abs(r - 3.5) - abs(c - 3.5)) * 1.0

        # 3. CASCADE-MERGE PAIR BONUS
        # Reward placing exactly 2 cells from a friendly in a line.
        # This creates a pair that can cascade-merge later.
        # Distance 2 in a line means one empty cell between them.
        for mi in my_positions:
            mr, mc = mi >> 3, mi & 7
            # Same row, 2 apart
            if mr == r and abs(mc - c) == 2:
                # Check the cell between is empty (merge path clear)
                between = mr * 8 + (mc + c) // 2
                if state[between] == 0:
                    score += 6.0  # strong pair bonus
                else:
                    score += 2.0  # blocked pair, still OK
            # Same column, 2 apart
            elif mc == c and abs(mr - r) == 2:
                between = ((mr + r) // 2) * 8 + mc
                if state[between] == 0:
                    score += 6.0
                else:
                    score += 2.0

        # 4. General friendly proximity (support)
        for mi in my_positions:
            d = MANHATTAN[idx][mi]
            if d == 2 or d == 3:
                score += 2.0
            elif d == 1:
                score += 0.5  # too close, limits moves
            elif d == 4:
                score += 1.0

        # 5. Block opponent expansion
        # Bonus for being near opponent positions (limiting their space)
        # but not too close (we can't place adjacent anyway after turn 1)
        for oi in opp_positions:
            d = MANHATTAN[idx][oi]
            if d == 2:
                score += 1.5  # constrains their placement options
            elif d == 3:
                score += 0.5

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
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
    """
    MCTS agent v3:
    - Quiescence in eval (sees immediate eats)
    - Forced approach when tallest (no retreating with B9)
    - Root anti-retreat filtering
    - Position repeat penalty
    - Corner death penalty
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._is_red = (color == PlayerColor.RED)
        self._board = BoardState()
        self._placement_count = 0
        self._turn_count = 0
        self._searcher = MCTSSearcher(c_param=0.7)
        self._position_history = []
        self._total_time_used = 0.0

    def action(self, **referee: dict) -> Action:
        turn_start = time.process_time()
        time_remaining = referee.get("time_remaining", None)

        if self._placement_count < PLACEMENT_TURNS:
            idx = choose_placement(self._board, self._is_red, self._placement_count)
            return PlaceAction(idx_to_coord(idx))

        if time_remaining is not None:
            remaining = time_remaining
        else:
            remaining = 180.0 - self._total_time_used

        # Adaptive budget
        my_tokens = self._board.red_tokens if self._is_red else self._board.blue_tokens
        opp_tokens = self._board.blue_tokens if self._is_red else self._board.red_tokens
        total = my_tokens + opp_tokens

        if remaining < 5:    budget = 0.15
        elif remaining < 15: budget = 0.3
        elif remaining < 40: budget = 0.5
        elif remaining < 80: budget = 0.8
        else:                budget = 1.0

        # Think harder when losing
        if total > 0 and my_tokens < opp_tokens:
            budget *= 1.3
        # Think harder late game
        if total <= 12:
            budget *= 1.2

        budget = max(0.05, min(budget, remaining * 0.15))

        best, visits, sims = self._searcher.search(
            self._board, self._is_red, budget, self._position_history)

        if best is None:
            actions = generate_actions_flat(self._board, self._is_red)
            if actions: best = actions[0]
            else: raise RuntimeError("No legal actions")

        # Final anti-repetition check
        test_board = apply_action(self._board, best[0], best[1], best[2])
        test_hash = bytes(test_board.state)
        repeats = self._position_history.count(test_hash)

        if repeats >= 1:
            all_actions = generate_actions_ordered(self._board, self._is_red)
            best_alt = None
            best_alt_score = -INF

            for alt in all_actions:
                if alt[0] == best[0] and alt[1] == best[1] and alt[2] == best[2]:
                    continue
                alt_board = apply_action(self._board, alt[0], alt[1], alt[2])
                alt_hash = bytes(alt_board.state)
                alt_repeats = self._position_history.count(alt_hash)
                if alt_repeats < repeats:
                    score = raw_evaluate(alt_board, self._is_red)
                    score += (repeats - alt_repeats) * 5.0
                    if score > best_alt_score:
                        best_alt_score = score
                        best_alt = alt

            if best_alt is not None:
                original_score = raw_evaluate(test_board, self._is_red)
                if best_alt_score > original_score - 10.0:
                    best = best_alt

        elapsed = time.process_time() - turn_start
        self._total_time_used += elapsed

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
        if self._placement_count >= PLACEMENT_TURNS:
            self._position_history.append(bytes(self._board.state))