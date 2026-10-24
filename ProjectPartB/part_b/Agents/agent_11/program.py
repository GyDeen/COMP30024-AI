# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: MCTS Agent for Cascade — v3.2
#
# CHANGES IN v3.2 (central anchor strategy):
#   * CENTER_4 = {(3,3),(3,4),(4,3),(4,4)} — the 4 cells where we
#     want to build a height-4+ "anchor" stack.
#   * Move-merge prefers destinations on CENTER_4. A merge that lands
#     at center beats the same-height merge elsewhere.
#   * Cascade-merge biases the resulting tall stack to land at center.
#   * Forced approach (when tallest), equal-height approach, pincer,
#     and corner-force phases all SKIP the anchor stack — only
#     non-anchor "hunters" move. The anchor stays as a static central
#     threat.
#   * Root filter: penalize cascading the anchor and moving the anchor
#     when we have one; reward MOVE actions that land at center.
#   * Cascade push-off requires destroying >=2 enemy tokens (user's
#     constraint: "do not cascade unless pushing >2 stack off").
#   * Placement: 3-of-4 blocking + 1 merge-support when going second
#     (from v3.1), with center bonus.

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

# 4 center cells: (3,3)=27, (3,4)=28, (4,3)=35, (4,4)=36
CENTER_4 = frozenset({27, 28, 35, 36})
ANCHOR_MIN_H = 4  # height threshold for "anchor"

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
    flag = RED_FLAG if is_red else BLUE_FLAG
    stacks = []
    for idx in range(64):
        val = board.state[idx]
        if val and (val & flag):
            stacks.append((idx, val & HEIGHT_MASK))
    return stacks


def _is_anchor(idx, h):
    """A stack qualifies as 'anchor' iff it sits on a center cell with h>=4."""
    return idx in CENTER_4 and h >= ANCHOR_MIN_H


def _has_anchor(board, is_red):
    flag = RED_FLAG if is_red else BLUE_FLAG
    for idx in CENTER_4:
        v = board.state[idx]
        if v and (v & flag) and (v & HEIGHT_MASK) >= ANCHOR_MIN_H:
            return True
    return False


def _min_kill_distance(board, is_red):
    """Closest (our-stack, killable-enemy) pair, regardless of anchor."""
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


def _min_kill_distance_hunter(board, is_red):
    """Closest non-anchor (our-stack, killable-enemy) pair.
    Used to pick a HUNTER for forced approach — anchor stays in place."""
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)
    best_d = 99; best_mi = -1; best_oi = -1
    for mi, mh in my_stacks:
        if _is_anchor(mi, mh):
            continue
        for oi, oh in opp_stacks:
            if mh > oh:
                d = MANHATTAN[mi][oi]
                if d < best_d:
                    best_d = d; best_mi = mi; best_oi = oi
    return best_d, best_mi, best_oi


def _we_are_tallest(board, is_red):
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
    my_flag = RED_FLAG if is_red else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red else RED_FLAG
    state = board.state
    my_before = board.red_tokens if is_red else board.blue_tokens
    opp_before = board.blue_tokens if is_red else board.red_tokens

    best_eat = None
    best_eat_gain = 0
    best_cascade = None
    best_cascade_gain = 0

    # ── Phase 1: Safe eat ─────────────────────────────────
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
                return (EAT, idx, offset)

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

    # ── Phase 2: Move-merge — PREFERS CENTER destinations ──
    # Score = new_h*10 + (25 if dest in CENTER_4) + edge_dist*2
    # This is THE move that builds the anchor.
    my_stacks_2 = _get_stacks(board, is_red)
    opp_stacks_2 = _get_stacks(board, not is_red)

    if my_stacks_2 and opp_stacks_2:
        my_max_h_2 = max(h for _, h in my_stacks_2)
        opp_max_h_2 = max(h for _, h in opp_stacks_2)

        if my_max_h_2 <= opp_max_h_2:
            best_merge_move = None
            best_merge_score = 0

            for mi, mh in my_stacks_2:
                for offset in OFFSETS:
                    if not is_valid_coord(mi, offset): continue
                    dest = mi + offset
                    dest_val = state[dest]
                    if not (dest_val and (dest_val & my_flag)):
                        continue

                    dest_h = dest_val & HEIGHT_MASK
                    new_h = mh + dest_h

                    safe = True
                    for check_off in OFFSETS:
                        if not is_valid_coord(dest, check_off): continue
                        cv = state[dest + check_off]
                        if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                            safe = False; break
                    if not safe:
                        continue
                    if EDGE_DIST[dest] <= 1:
                        continue

                    merge_score = new_h * 10
                    if dest in CENTER_4:
                        merge_score += 25  # ★ ANCHOR BONUS ★
                    merge_score += EDGE_DIST[dest] * 2

                    if merge_score > best_merge_score:
                        best_merge_score = merge_score
                        best_merge_move = (MOVE, mi, offset)

            if best_merge_move is not None:
                return best_merge_move

    # ── Phase 3: Forced approach when tallest (HUNTERS only) ──
    if _we_are_tallest(board, is_red):
        cur_dist, our_idx, target_idx = _min_kill_distance_hunter(board, is_red)
        if our_idx < 0:
            cur_dist, our_idx, target_idx = _min_kill_distance(board, is_red)

        if cur_dist >= 2 and our_idx >= 0:
            our_h = state[our_idx] & HEIGHT_MASK

            best_approach = None
            best_new_dist = cur_dist

            for offset in OFFSETS:
                if not is_valid_coord(our_idx, offset): continue
                dest = our_idx + offset
                dest_val = state[dest]

                if dest_val != 0 and (dest_val & opp_flag):
                    continue
                if dest_val != 0 and not (dest_val & my_flag):
                    continue

                new_h = our_h
                if dest_val and (dest_val & my_flag):
                    new_h = our_h + (dest_val & HEIGHT_MASK)

                new_dist = MANHATTAN[dest][target_idx]
                if new_dist >= best_new_dist:
                    continue

                safe = True
                for check_off in OFFSETS:
                    if not is_valid_coord(dest, check_off): continue
                    cv = state[dest + check_off]
                    if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                        safe = False; break

                dest_ed = EDGE_DIST[dest]
                if dest_ed <= 1 and new_dist > 2:
                    continue

                if safe and new_dist < best_new_dist:
                    nb = apply_action(board, MOVE, our_idx, offset)
                    nb_hash = bytes(nb.state)
                    if position_history.count(nb_hash) >= 2:
                        continue
                    best_new_dist = new_dist
                    best_approach = (MOVE, our_idx, offset)

            if best_approach is not None:
                return best_approach

    # ── Phase 2b: Equal-height approach (HUNTERS only) ────
    my_stacks_eq = _get_stacks(board, is_red)
    opp_stacks_eq = _get_stacks(board, not is_red)
    if my_stacks_eq and opp_stacks_eq:
        my_max_eq = max(h for _, h in my_stacks_eq)
        opp_max_eq = max(h for _, h in opp_stacks_eq)

        if my_max_eq == opp_max_eq:
            best_eq_dist = 99
            eq_target = -1
            eq_source = -1
            eq_h = 0

            for mi, mh in my_stacks_eq:
                if _is_anchor(mi, mh):
                    continue  # anchor stays put
                for oi, oh in opp_stacks_eq:
                    if mh > oh:
                        d = MANHATTAN[mi][oi]
                        if d < best_eq_dist:
                            best_eq_dist = d
                            eq_target = oi
                            eq_source = mi
                            eq_h = mh

            if best_eq_dist >= 2 and eq_target >= 0:
                best_move = None
                best_new_d = best_eq_dist

                for offset in OFFSETS:
                    if not is_valid_coord(eq_source, offset): continue
                    dest = eq_source + offset
                    dest_val = state[dest]
                    if dest_val != 0 and (dest_val & opp_flag): continue
                    if dest_val != 0 and not (dest_val & my_flag): continue

                    new_h = eq_h
                    if dest_val and (dest_val & my_flag):
                        new_h = eq_h + (dest_val & HEIGHT_MASK)

                    new_d = MANHATTAN[dest][eq_target]
                    if new_d >= best_new_d: continue

                    safe = True
                    for co in OFFSETS:
                        if not is_valid_coord(dest, co): continue
                        cv = state[dest + co]
                        if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                            safe = False; break
                    if not safe: continue

                    if EDGE_DIST[dest] <= 1 and new_d > 2: continue

                    best_new_d = new_d
                    best_move = (MOVE, eq_source, offset)

                if best_move is not None:
                    return best_move

    # ── Phase 3 (cascade-merge) — PREFERS CENTER landing ──
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)

    if my_stacks and opp_stacks:
        my_max_h_before = max(h for _, h in my_stacks)
        opp_max_h = max(h for _, h in opp_stacks)

        if my_max_h_before <= opp_max_h:
            best_merge = None
            best_merge_score = 0

            for idx in range(64):
                val = state[idx]
                if not (val & my_flag): continue
                h = val & HEIGHT_MASK
                if h < 2: continue

                for offset in OFFSETS:
                    nb = apply_action(board, CASCADE, idx, offset)
                    nb_my = nb.red_tokens if is_red else nb.blue_tokens
                    nb_opp = nb.blue_tokens if is_red else nb.red_tokens

                    my_lost = my_before - nb_my
                    if my_lost > 1: continue

                    opp_lost = opp_before - nb_opp

                    nb_my_stacks = []
                    for ci in range(64):
                        cv = nb.state[ci]
                        if cv and (cv & my_flag):
                            nb_my_stacks.append((ci, cv & HEIGHT_MASK))
                    if not nb_my_stacks: continue

                    nb_max_h = max(sh for _, sh in nb_my_stacks)
                    if nb_max_h <= my_max_h_before: continue

                    debris_safe = True
                    for ci, ch in nb_my_stacks:
                        if ch > 1: continue
                        for oi, oh in opp_stacks:
                            if MANHATTAN[ci][oi] <= 2:
                                debris_safe = False
                                break
                        if not debris_safe: break
                    if not debris_safe: continue

                    tall_idx = -1
                    tall_ed = 0
                    for ci, ch in nb_my_stacks:
                        if ch == nb_max_h:
                            tall_idx = ci
                            tall_ed = EDGE_DIST[ci]
                            break
                    if tall_ed <= 1: continue

                    height_gain = nb_max_h - my_max_h_before
                    # ★ CENTER BONUS: huge if new tall stack lands at anchor cell
                    center_bonus = 30.0 if tall_idx in CENTER_4 else 0.0

                    merge_score = (
                        height_gain * 10.0
                        + tall_ed * 2.0
                        + center_bonus
                        + opp_lost * 5.0
                        - my_lost * 8.0
                    )

                    if merge_score > best_merge_score:
                        best_merge_score = merge_score
                        best_merge = (CASCADE, idx, offset)

            if best_merge is not None:
                return best_merge

    # ── Phase 4: Profitable cascade push-off ──────────────
    # User constraint: only cascade if we destroy a meaningful (h>=2)
    # opponent stack. We approximate by requiring opp_destroyed >= 2
    # AND adjusted gain >= 3.
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
            # Don't sacrifice the anchor for a cascade
            if _is_anchor(idx, h): continue
            for offset in OFFSETS:
                nb = apply_action(board, CASCADE, idx, offset)
                opp_left = nb.blue_tokens if is_red else nb.red_tokens
                my_left = nb.red_tokens if is_red else nb.blue_tokens
                if opp_left == 0 and my_left > 0:
                    return (CASCADE, idx, offset)
                opp_destroyed = opp_before - opp_left
                my_lost = my_before - my_left
                if opp_destroyed < 2: continue  # user: only push h>2 stacks off
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

    # ── Phase 6: Pincer setup (HUNTERS only) ──────────────
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)

    if len(my_stacks) >= 2 and opp_stacks:
        best_pincer = None
        best_pincer_score = 0

        for mi_anchor, mh_anchor in my_stacks:
            ar, ac = mi_anchor >> 3, mi_anchor & 7

            for oi, oh in opp_stacks:
                or_, oc = oi >> 3, oi & 7

                if ar == or_ and abs(ac - oc) <= 3:
                    if ac < oc:
                        for target_c in range(oc + 1, min(oc + 4, 8)):
                            target_idx = or_ * 8 + target_c
                            for mi_mover, mh_mover in my_stacks:
                                if mi_mover == mi_anchor: continue
                                if _is_anchor(mi_mover, mh_mover): continue
                                if MANHATTAN[mi_mover][target_idx] != 1: continue
                                tv = state[target_idx]
                                if tv != 0 and not (tv & my_flag): continue
                                can_eat = (mh_anchor >= oh) and (mh_mover >= oh)
                                perp_escape = min(or_, 7 - or_)
                                gap = target_c - ac - 1

                                score_p = oh * 2.0
                                if can_eat:
                                    score_p += oh * 4.0
                                if gap <= 2:
                                    score_p += oh * 2.0
                                if perp_escape <= 1:
                                    score_p += oh * 3.0

                                if score_p > best_pincer_score:
                                    diff = target_idx - mi_mover
                                    for off in OFFSETS:
                                        if off == diff:
                                            best_pincer_score = score_p
                                            best_pincer = (MOVE, mi_mover, off)
                                            break
                    else:
                        for target_c in range(oc - 1, max(oc - 4, -1), -1):
                            target_idx = or_ * 8 + target_c
                            for mi_mover, mh_mover in my_stacks:
                                if mi_mover == mi_anchor: continue
                                if _is_anchor(mi_mover, mh_mover): continue
                                if MANHATTAN[mi_mover][target_idx] != 1: continue
                                tv = state[target_idx]
                                if tv != 0 and not (tv & my_flag): continue
                                can_eat = (mh_anchor >= oh) and (mh_mover >= oh)
                                perp_escape = min(or_, 7 - or_)
                                gap = ac - target_c - 1

                                score_p = oh * 2.0
                                if can_eat: score_p += oh * 4.0
                                if gap <= 2: score_p += oh * 2.0
                                if perp_escape <= 1: score_p += oh * 3.0

                                if score_p > best_pincer_score:
                                    diff = target_idx - mi_mover
                                    for off in OFFSETS:
                                        if off == diff:
                                            best_pincer_score = score_p
                                            best_pincer = (MOVE, mi_mover, off)
                                            break

                if ac == oc and abs(ar - or_) <= 3:
                    if ar < or_:
                        for target_r in range(or_ + 1, min(or_ + 4, 8)):
                            target_idx = target_r * 8 + oc
                            for mi_mover, mh_mover in my_stacks:
                                if mi_mover == mi_anchor: continue
                                if _is_anchor(mi_mover, mh_mover): continue
                                if MANHATTAN[mi_mover][target_idx] != 1: continue
                                tv = state[target_idx]
                                if tv != 0 and not (tv & my_flag): continue
                                can_eat = (mh_anchor >= oh) and (mh_mover >= oh)
                                perp_escape = min(oc, 7 - oc)
                                gap = target_r - ar - 1

                                score_p = oh * 2.0
                                if can_eat: score_p += oh * 4.0
                                if gap <= 2: score_p += oh * 2.0
                                if perp_escape <= 1: score_p += oh * 3.0

                                if score_p > best_pincer_score:
                                    diff = target_idx - mi_mover
                                    for off in OFFSETS:
                                        if off == diff:
                                            best_pincer_score = score_p
                                            best_pincer = (MOVE, mi_mover, off)
                                            break
                    else:
                        for target_r in range(or_ - 1, max(or_ - 4, -1), -1):
                            target_idx = target_r * 8 + oc
                            for mi_mover, mh_mover in my_stacks:
                                if mi_mover == mi_anchor: continue
                                if _is_anchor(mi_mover, mh_mover): continue
                                if MANHATTAN[mi_mover][target_idx] != 1: continue
                                tv = state[target_idx]
                                if tv != 0 and not (tv & my_flag): continue
                                can_eat = (mh_anchor >= oh) and (mh_mover >= oh)
                                perp_escape = min(oc, 7 - oc)
                                gap = ar - target_r - 1

                                score_p = oh * 2.0
                                if can_eat: score_p += oh * 4.0
                                if gap <= 2: score_p += oh * 2.0
                                if perp_escape <= 1: score_p += oh * 3.0

                                if score_p > best_pincer_score:
                                    diff = target_idx - mi_mover
                                    for off in OFFSETS:
                                        if off == diff:
                                            best_pincer_score = score_p
                                            best_pincer = (MOVE, mi_mover, off)
                                            break

        if best_pincer is not None and best_pincer_score >= 8.0:
            return best_pincer

    # ── Phase 7: Corner-forcing move (HUNTERS only) ──────
    if my_stacks and opp_stacks:
        best_corner_force = None
        best_cf_score = 0

        for oi, oh in opp_stacks:
            or_, oc = oi >> 3, oi & 7
            opp_ed = EDGE_DIST[oi]
            if opp_ed > 2: continue

            for mi, mh in my_stacks:
                if _is_anchor(mi, mh): continue  # anchor doesn't chase
                if mh < oh and mh < opp_ed + 1: continue

                cur_dist = MANHATTAN[mi][oi]
                if cur_dist > 4: continue

                for offset in OFFSETS:
                    if not is_valid_coord(mi, offset): continue
                    dest = mi + offset
                    dest_val = state[dest]

                    if dest_val != 0 and not (dest_val & my_flag): continue
                    if dest_val != 0 and (dest_val & opp_flag): continue

                    dest_r, dest_c = dest >> 3, dest & 7
                    new_dist = MANHATTAN[dest][oi]
                    dest_ed = EDGE_DIST[dest]

                    blocking = False
                    if dest_r == or_ and dest_ed > opp_ed:
                        blocking = True
                    if dest_c == oc and dest_ed > opp_ed:
                        blocking = True
                    if new_dist == 1 and (dest_r == or_ or dest_c == oc):
                        blocking = True
                    if new_dist == 1 and dest_ed > opp_ed:
                        blocking = True

                    if not blocking: continue
                    if new_dist > cur_dist: continue

                    safe = True
                    for check_off in OFFSETS:
                        if not is_valid_coord(dest, check_off): continue
                        cv = state[dest + check_off]
                        new_h = mh + ((dest_val & HEIGHT_MASK) if dest_val and (dest_val & my_flag) else 0)
                        if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                            safe = False; break
                    if not safe: continue

                    cf_score = oh * (3 - opp_ed) * 2.0
                    if new_dist == 1:
                        cf_score += oh * 3.0
                    if blocking:
                        cf_score += oh * 2.0

                    if cf_score > best_cf_score:
                        best_cf_score = cf_score
                        best_corner_force = (MOVE, mi, offset)

        if best_corner_force is not None and best_cf_score >= 6.0:
            return best_corner_force

    # ── Phase 8: Forced flee toward center when shorter ──
    my_stacks = _get_stacks(board, is_red)
    opp_stacks = _get_stacks(board, not is_red)

    if my_stacks and opp_stacks and len(my_stacks) <= 2:
        my_max_h = max(h for _, h in my_stacks)
        opp_max_h = max(h for _, h in opp_stacks)

        if opp_max_h > my_max_h:
            best_flee = None
            best_flee_score = -INF

            for mi, mh in my_stacks:
                cur_ed = EDGE_DIST[mi]
                min_threat_dist = 99
                for oi, oh in opp_stacks:
                    if oh > mh:
                        d = MANHATTAN[mi][oi]
                        if d < min_threat_dist:
                            min_threat_dist = d

                if min_threat_dist > 5 and cur_ed >= 2:
                    continue

                for offset in OFFSETS:
                    if not is_valid_coord(mi, offset): continue
                    dest = mi + offset
                    dest_val = state[dest]

                    if dest_val != 0 and (dest_val & opp_flag):
                        continue
                    if dest_val != 0 and not (dest_val & my_flag):
                        continue

                    new_h = mh
                    if dest_val and (dest_val & my_flag):
                        new_h = mh + (dest_val & HEIGHT_MASK)

                    safe = True
                    for check_off in OFFSETS:
                        if not is_valid_coord(dest, check_off): continue
                        cv = state[dest + check_off]
                        if cv and (cv & opp_flag) and (cv & HEIGHT_MASK) >= new_h:
                            safe = False; break
                    if not safe:
                        continue

                    dest_ed = EDGE_DIST[dest]

                    new_threat_dist = 99
                    for oi, oh in opp_stacks:
                        if oh > new_h:
                            d = MANHATTAN[dest][oi]
                            if d < new_threat_dist:
                                new_threat_dist = d

                    score = new_threat_dist * 1.5 + dest_ed * 3.0

                    if dest_val and (dest_val & my_flag):
                        score += (dest_val & HEIGHT_MASK) * 2.0

                    nb = apply_action(board, MOVE, mi, offset)
                    nb_hash = bytes(nb.state)
                    if position_history.count(nb_hash) >= 2:
                        score -= 100

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
# MCTS Searcher
# ──────────────────────────────────────────────
class MCTSSearcher:
    def __init__(self, c_param=0.7):
        self.c_param = c_param

    def _filter_root_actions(self, board, is_red, actions, position_history):
        my_flag = RED_FLAG if is_red else BLUE_FLAG
        opp_flag = BLUE_FLAG if is_red else RED_FLAG
        my_before = board.red_tokens if is_red else board.blue_tokens
        opp_before = board.blue_tokens if is_red else board.red_tokens
        tallest = _we_are_tallest(board, is_red)
        cur_kill_dist = _min_kill_distance(board, is_red)[0] if tallest else 99
        we_have_anchor = _has_anchor(board, is_red)

        scored = []
        for act in actions:
            priority = 0

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

            if act[0] == EAT:
                captured = opp_before - (nb.blue_tokens if is_red else nb.red_tokens)
                priority = 200 + captured * 10
            elif act[0] == CASCADE:
                priority = 100
                # Penalize cascading the anchor (breaks strategic core)
                src_val = board.state[act[1]]
                if (src_val & my_flag) and _is_anchor(act[1], src_val & HEIGHT_MASK):
                    priority -= 60
            elif act[0] == MOVE:
                priority = 10
                # Reward moves that LAND on a center cell (build/maintain anchor)
                dest = act[1] + act[2]
                if dest in CENTER_4:
                    priority += 30

            if tallest and act[0] == MOVE:
                # Don't move the anchor when we already have one
                if we_have_anchor:
                    src_val = board.state[act[1]]
                    if (src_val & my_flag) and _is_anchor(act[1], src_val & HEIGHT_MASK):
                        priority -= 50

                new_kill_dist = _min_kill_distance(nb, is_red)[0]
                if new_kill_dist < cur_kill_dist:
                    priority += 50
                elif new_kill_dist > cur_kill_dist:
                    priority -= 80

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
    First player (placements 0,2,4,6): center then merge cluster.
    Second player (1,3,5,7): diag-block, block, block, merge support.
    All placements: avoid edges, prefer center-2x2 area.
    """
    legal = generate_placement_actions(board, is_red, placement_count)
    if not legal:
        raise RuntimeError("No legal placements")

    my_flag = RED_FLAG if is_red else BLUE_FLAG
    opp_flag = BLUE_FLAG if is_red else RED_FLAG
    state = board.state
    legal_set = set(legal)
    going_first = (placement_count % 2 == 0)

    if placement_count == 0:
        for idx in [27, 36, 28, 35]:
            if idx in legal_set:
                return idx

    my_positions = []
    opp_positions = []
    for i in range(64):
        v = state[i]
        if v == 0: continue
        if v & my_flag: my_positions.append(i)
        elif v & opp_flag: opp_positions.append(i)

    if placement_count == 1 and opp_positions:
        opp_idx = opp_positions[0]
        opp_r, opp_c = opp_idx >> 3, opp_idx & 7
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

    enemy_adjacent = set()
    for oi in opp_positions:
        or_, oc = oi >> 3, oi & 7
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = or_ + dr, oc + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                enemy_adjacent.add(nr * 8 + nc)

    if going_first:
        block_weight = 0.5
        merge_weight = 1.6
    elif placement_count == 7:
        block_weight = 0.4
        merge_weight = 1.8
    else:
        block_weight = 1.7
        merge_weight = 0.6

    best_idx = legal[0]
    best_score = -1e9

    for idx in legal:
        score = 0.0
        r, c = idx >> 3, idx & 7
        ed = EDGE_DIST[idx]

        score += ed * 4.0
        score += max(0, 4 - abs(r - 3.5) - abs(c - 3.5)) * 1.0

        if idx in CENTER_4:
            score += 3.0  # seed for the future anchor

        block_score = 0.0
        for oi in opp_positions:
            or_, oc = oi >> 3, oi & 7
            d = MANHATTAN[idx][oi]
            if d == 2:
                if abs(r - or_) == 1 and abs(c - oc) == 1:
                    block_score += 6.0
                else:
                    block_score += 3.5
            elif d == 3:
                block_score += 1.2
        score += block_score * block_weight

        merge_score = 0.0
        for mi in my_positions:
            mr, mc = mi >> 3, mi & 7
            mi_ed = EDGE_DIST[mi]
            d = MANHATTAN[idx][mi]

            if d == 1:
                if ed >= 2 and mi_ed >= 2:
                    mi_safe = mi not in enemy_adjacent
                    idx_safe = idx not in enemy_adjacent
                    if mi_safe:
                        merge_score += 8.0
                    elif idx_safe:
                        merge_score += 6.0
                    else:
                        merge_score += 2.0
                    if mi in CENTER_4 or idx in CENTER_4:
                        merge_score += 4.0
                else:
                    merge_score += 1.5
            elif d == 2:
                between = -1
                if mr == r and abs(mc - c) == 2:
                    between = mr * 8 + (mc + c) // 2
                elif mc == c and abs(mr - r) == 2:
                    between = ((mr + r) // 2) * 8 + mc
                if between >= 0:
                    if state[between] == 0 and between not in enemy_adjacent:
                        merge_score += 4.0
                    else:
                        merge_score += 1.0
                else:
                    merge_score += 1.5
        score += merge_score * merge_weight

        for mi in my_positions:
            d = MANHATTAN[idx][mi]
            if d == 3:
                score += 0.8
            elif d == 4:
                score += 0.4

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

        my_tokens = self._board.red_tokens if self._is_red else self._board.blue_tokens
        opp_tokens = self._board.blue_tokens if self._is_red else self._board.red_tokens
        total = my_tokens + opp_tokens

        if remaining < 5:    budget = 0.15
        elif remaining < 15: budget = 0.3
        elif remaining < 40: budget = 0.5
        elif remaining < 80: budget = 0.8
        else:                budget = 1.0

        if total > 0 and my_tokens < opp_tokens:
            budget *= 1.3
        if total <= 12:
            budget *= 1.2

        budget = max(0.05, min(budget, remaining * 0.15))

        best, visits, sims = self._searcher.search(
            self._board, self._is_red, budget, self._position_history)

        if best is None:
            actions = generate_actions_flat(self._board, self._is_red)
            if actions: best = actions[0]
            else: raise RuntimeError("No legal actions")

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