# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: MCTS Agent for Cascade
#
# Architecture:
#   1. Tactical override: instant wins, safe captures, profitable cascades
#   2. MCTS with raw eval at leaves (~25K sims/s) for strategy
#   3. Anti-repetition to avoid threefold draw
#   4. Time-aware budgeting with elapsed time tracking

import math
import time
from referee.game import PlayerColor, Coord, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from .board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from .action import (
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
RECENT_POSITION_WINDOW = 12


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


def board_signature(board, is_red_turn):
    return (bytes(board.state), is_red_turn)


def opposite_offset(offset):
    if offset == UP: return DOWN
    if offset == DOWN: return UP
    if offset == LEFT: return RIGHT
    return LEFT


def reverse_of_last_move(action, last_action):
    if last_action is None:
        return False
    if action[0] != MOVE or last_action[0] != MOVE:
        return False
    last_dest = last_action[1] + last_action[2]
    return action[1] == last_dest and action[2] == opposite_offset(last_action[2])


# ──────────────────────────────────────────────
# Tactical decision layer
# ──────────────────────────────────────────────
EDGE_DIST = [0] * 64
for _r in range(8):
    for _c in range(8):
        EDGE_DIST[_r * 8 + _c] = min(_r, 7 - _r, _c, 7 - _c)

MANHATTAN = [[0]*64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        MANHATTAN[_a][_b] = abs((_a>>3)-(_b>>3)) + abs((_a&7)-(_b&7))


def tactical_decision(board, is_red):
    """
    Only recommends actions that are CLEARLY and SAFELY profitable.
    
    1. Instant win: always take (EAT or CASCADE that kills all enemies)
    2. Safe EAT: capture with no enemy that can eat us back at destination
    3. CASCADE: ONLY when gain is overwhelming after counting ALL risks:
       - Tokens lost off the edge
       - Our new h=1 stacks adjacent to enemies (eaten next turn)
       - Our new h=1 stacks within 2 cells of enemies (eaten in 2 turns)
       Minimum threshold: adjusted gain >= 3 (must be worth it)
       Exception: instant win cascades always taken.
    
    Safe EATs are ALWAYS preferred over cascades (same gain → take the eat).
    Returns None when no clearly profitable action exists → MCTS searches.
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

    # ── Phase 1: Check all EATs (always preferred) ──
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

    # ── Phase 2: Check CASCADEs (strict criteria) ──
    opp_near_edge = False
    for idx in range(64):
        if state[idx] and (state[idx] & opp_flag) and EDGE_DIST[idx] <= 2:
            opp_near_edge = True; break

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
                    return (CASCADE, idx, offset)  # instant win

                opp_destroyed = opp_before - opp_left
                my_lost_edge = my_before - my_left
                if opp_destroyed == 0:
                    continue

                # Count our tokens at risk: adjacent OR within 2 cells
                # Adjacent = eaten next turn (full risk)
                # Distance 2 = eaten in 2 turns (partial risk)
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
                        if d <= 1:
                            my_at_risk += ch      # dead next turn
                            break
                        elif d <= 2:
                            my_at_risk += ch * 0.7  # very likely dead
                            break
                        elif d <= 3:
                            my_at_risk += ch * 0.3  # possibly dead
                            break

                adjusted = opp_destroyed - my_lost_edge - my_at_risk

                # Require gain >= 3 for non-instant-win cascades.
                # This prevents marginal cascades like "destroy 3, lose 2,
                # risk 1 → adjusted=0.3 → technically positive but terrible."
                if adjusted >= 3 and adjusted > best_cascade_gain:
                    best_cascade_gain = adjusted
                    best_cascade = (CASCADE, idx, offset)

    # Prefer EAT over CASCADE (even if cascade has higher raw gain)
    if best_eat is not None:
        return best_eat
    if best_cascade is not None:
        return best_cascade
    return None


def root_progress_score(board, is_red, action):
    """
    Reward moves that help convert a winning taller-stack endgame and
    penalize sideways repetition.
    """
    my_flag = RED_FLAG if is_red else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red else RED_FLAG
    my_before = board.red_tokens if is_red else board.blue_tokens
    opp_before = board.blue_tokens if is_red else board.red_tokens
    nb = apply_action(board, action[0], action[1], action[2])
    my_after = nb.red_tokens if is_red else nb.blue_tokens
    opp_after = nb.blue_tokens if is_red else nb.red_tokens

    captured = opp_before - opp_after
    self_loss = my_before - my_after

    my_stacks = []
    opp_stacks = []
    my_stacks_after = []
    opp_stacks_after = []
    for idx in range(64):
        v = board.state[idx]
        if v:
            h = v & HEIGHT_MASK
            if v & my_flag: my_stacks.append((idx, h))
            elif v & opp_flag: opp_stacks.append((idx, h))
        nv = nb.state[idx]
        if nv:
            nh = nv & HEIGHT_MASK
            if nv & my_flag: my_stacks_after.append((idx, nh))
            elif nv & opp_flag: opp_stacks_after.append((idx, nh))

    my_max_before = max((h for _, h in my_stacks), default=0)
    opp_max_before = max((h for _, h in opp_stacks), default=0)
    my_max_after = max((h for _, h in my_stacks_after), default=0)
    opp_max_after = max((h for _, h in opp_stacks_after), default=0)

    pressure_gain = 0.0
    if my_max_before > opp_max_before and my_stacks and opp_stacks:
        big_before = max(my_stacks, key=lambda x: x[1])[0]
        nearest_before = min(
            (MANHATTAN[big_before][oi] for oi, oh in opp_stacks if my_max_before > oh),
            default=8
        )
        if my_stacks_after and opp_stacks_after and my_max_after > opp_max_after:
            big_after = max(my_stacks_after, key=lambda x: x[1])[0]
            nearest_after = min(
                (MANHATTAN[big_after][oi] for oi, oh in opp_stacks_after if my_max_after > oh),
                default=8
            )
            pressure_gain += (nearest_before - nearest_after) * 3.5

    return captured * 12.0 - self_loss * 9.0 + pressure_gain


# ──────────────────────────────────────────────
# Eval → [0, 1]
# ──────────────────────────────────────────────
def eval_to_win_prob(board, player_is_red):
    r, b = board.red_tokens, board.blue_tokens
    if r == 0: return 0.0 if player_is_red else 1.0
    if b == 0: return 1.0 if player_is_red else 0.0
    raw = raw_evaluate(board, player_is_red)
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
# MCTS Searcher — raw eval only (fast: ~25K sims/s)
# ──────────────────────────────────────────────
class MCTSSearcher:
    def __init__(self, c_param=0.7):
        self.c_param = c_param

    def choose_final_child(self, root, is_red_turn, recent_positions=None, last_action=None):
        best = None
        best_score = -INF
        total_visits = max(1, root.visits)
        for child in root.children:
            if child.visits == 0:
                continue
            win_rate = 1.0 - (child.value / child.visits)
            visit_share = child.visits / total_visits
            progress = root_progress_score(root.board, is_red_turn, child.action)
            penalty = 0.0
            if recent_positions is not None and board_signature(child.board, not is_red_turn) in recent_positions:
                penalty += 10.0
            if reverse_of_last_move(child.action, last_action):
                penalty += 6.0
            score = win_rate * 100.0 + visit_share * 6.0 + progress - penalty
            if score > best_score:
                best_score = score
                best = child
        return best if best is not None else root.most_visited_child()

    def _filter_root_actions(self, board, is_red, actions):
        """
        Filter out clearly bad actions at the root:
        
        1. CASCADE: only if net gain > 0 and debris is safe (existing logic)
        2. MOVE suicide: if we have ≤2 stacks, don't move a stack adjacent
           to an enemy of equal-or-greater height (they eat us next turn).
           EXCEPTION: if we're strictly taller, we eat THEM — that's safe.
        
        This prevents MCTS from exploring suicidal lines that waste sims.
        """
        my_flag = RED_FLAG if is_red else BLUE_FLAG
        opp_flag = BLUE_FLAG if is_red else RED_FLAG
        my_before = board.red_tokens if is_red else board.blue_tokens
        opp_before = board.blue_tokens if is_red else board.red_tokens
        state = board.state

        # Count our stacks
        my_stack_count = sum(1 for i in range(64) if state[i] and (state[i] & my_flag))

        filtered = []
        for act in actions:
            # ── CASCADE filter (existing) ──
            if act[0] == CASCADE:
                nb = apply_action(board, act[0], act[1], act[2])
                my_after = nb.red_tokens if is_red else nb.blue_tokens
                opp_after = nb.blue_tokens if is_red else nb.red_tokens
                net = (opp_before - opp_after) - (my_before - my_after)
                if net <= 0: continue
                if net < 3:
                    debris_near_enemy = False
                    for ci in range(64):
                        cv = nb.state[ci]
                        if cv and (cv & my_flag) and (cv & HEIGHT_MASK) == 1:
                            for oi in range(64):
                                ov = nb.state[oi]
                                if ov and (ov & opp_flag) and MANHATTAN[ci][oi] <= 3:
                                    debris_near_enemy = True; break
                            if debris_near_enemy: break
                    if debris_near_enemy: continue
                filtered.append(act)
                continue

            # ── MOVE suicide filter (only when few stacks) ──
            if act[0] == MOVE and my_stack_count <= 2:
                src = act[1]
                src_h = state[src] & HEIGHT_MASK
                dest = src + act[2]

                # Check if destination is adjacent to an enemy that can eat us
                landing_threatened = False
                for off in OFFSETS:
                    if not is_valid_coord(dest, off): continue
                    nv = state[dest + off]
                    if nv and (nv & opp_flag):
                        nh = nv & HEIGHT_MASK
                        if nh >= src_h:
                            # This includes merges: check if we'd be taller after
                            # moving onto a friendly stack
                            dest_val = state[dest]
                            if dest_val and (dest_val & my_flag):
                                # Merging: combined height
                                merged_h = src_h + (dest_val & HEIGHT_MASK)
                                if merged_h > nh:
                                    continue  # merge makes us taller — safe
                            landing_threatened = True
                            break

                if landing_threatened:
                    continue  # skip this suicidal move

            filtered.append(act)

        return filtered if filtered else actions

    def search(self, board, is_red_turn, time_budget, recent_positions=None, last_action=None):
        forced = tactical_decision(board, is_red_turn)
        if forced is not None:
            return forced, 1, 1

        all_actions = generate_actions_ordered(board, is_red_turn)
        if not all_actions: return None, 0, 0
        if len(all_actions) == 1: return all_actions[0], 1, 1

        # Filter bad cascades at root level
        filtered = self._filter_root_actions(board, is_red_turn, all_actions)

        root = MCTSNode(board, is_red_turn)
        # Override root's untried actions with filtered list
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

            result = eval_to_win_prob(node.board, node.is_red_turn)
            node.backpropagate(result)
            simulations += 1

        if not root.children:
            return (all_actions[0], 0, simulations)
        best = self.choose_final_child(root, is_red_turn, recent_positions, last_action)
        return best.action, best.visits, simulations


# ──────────────────────────────────────────────
# Placement strategy
# ──────────────────────────────────────────────
def choose_placement(board, is_red, placement_count):
    legal = generate_placement_actions(board, is_red, placement_count)
    if not legal:
        raise RuntimeError("No legal placements")

    my_flag = RED_FLAG if is_red else BLUE_FLAG
    state = board.state

    my_positions = [i for i in range(64) if state[i] and (state[i] & my_flag)]

    best_idx = legal[0]
    best_score = -999

    for idx in legal:
        score = 0.0
        ed = EDGE_DIST[idx]

        # Avoid edges — cascade death zone
        score += ed * 3.0

        # Near friendly stacks at distance 2-3 for mutual support
        for mi in my_positions:
            d = abs((idx >> 3) - (mi >> 3)) + abs((idx & 7) - (mi & 7))
            if d == 2 or d == 3:
                score += 2.0
            elif d == 1:
                score += 0.5
            elif d == 4:
                score += 1.0

        # Center bias tiebreaker
        r, c = idx >> 3, idx & 7
        score += max(0, 4 - abs(r - 3.5) - abs(c - 3.5)) * 0.5

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
    MCTS agent with:
    - Tactical override (instant wins, safe captures, profitable cascades)
    - Raw eval leaves (~25K sims/s for maximum breadth)
    - Anti-repetition (avoids threefold draw)
    - Time tracking with budget scaling
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
        self._recent_positions = []
        self._last_my_action = None

    def action(self, **referee: dict) -> Action:
        turn_start = time.process_time()
        time_remaining = referee.get("time_remaining", None)

        # ── Placement phase (instant, no search) ──
        if self._placement_count < PLACEMENT_TURNS:
            idx = choose_placement(self._board, self._is_red, self._placement_count)
            return PlaceAction(idx_to_coord(idx))

        # ── Time budget ──
        # Use referee's time_remaining if available, else estimate.
        if time_remaining is not None:
            remaining = time_remaining
        else:
            remaining = 180.0 - self._total_time_used

        # Scale budget: spend proportionally to remaining time.
        # Reserve at least 30s for endgame.
        if remaining < 5:
            budget = 0.1
        elif remaining < 15:
            budget = 0.2
        elif remaining < 40:
            budget = 0.3
        elif remaining < 80:
            budget = 0.5
        else:
            budget = 0.8
        budget = max(0.05, budget)

        # ── MCTS search ──
        best, visits, sims = self._searcher.search(
            self._board, self._is_red, budget,
            recent_positions=set(self._recent_positions),
            last_action=self._last_my_action,
        )

        if best is None:
            actions = generate_actions_flat(self._board, self._is_red)
            if actions: best = actions[0]
            else: raise RuntimeError("No legal actions")

        # ── Anti-repetition ──
        # Only avoid positions we've seen TWICE (one more = threefold draw).
        # This is much less aggressive than the old "seen once" check.
        test_board = apply_action(self._board, best[0], best[1], best[2])
        test_hash = bytes(test_board.state)
        if self._position_history.count(test_hash) >= 2:
            # This would cause threefold draw. Find alternative.
            all_actions = generate_actions_ordered(self._board, self._is_red)
            for alt in reversed(all_actions):
                if alt[0] == best[0] and alt[1] == best[1] and alt[2] == best[2]:
                    continue
                alt_board = apply_action(self._board, alt[0], alt[1], alt[2])
                alt_hash = bytes(alt_board.state)
                if self._position_history.count(alt_hash) < 2:
                    best = alt
                    break

        # ── Track time ──
        elapsed = time.process_time() - turn_start
        self._total_time_used += elapsed
        self._last_my_action = best

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
        # Record positions during play phase for repetition detection
        if self._placement_count >= PLACEMENT_TURNS:
            self._position_history.append(bytes(self._board.state))
            if color != self._color:
                self._recent_positions.append(board_signature(self._board, self._is_red))
                if len(self._recent_positions) > RECENT_POSITION_WINDOW:
                    self._recent_positions.pop(0)
