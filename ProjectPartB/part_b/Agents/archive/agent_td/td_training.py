"""
TD-Leaf(λ) Training for Cascade

Run this OFFLINE to learn evaluation weights through self-play:
    cd part_b
    python -m Agents.archive.agent_td.td_training

It plays ~2000 self-play games, updating weights after each game.
Takes 1-4 hours depending on your machine. Saves weights to
agent/learned_weights.npy which the Agent loads at game time.

HOW TD-LEAF(λ) WORKS:
────────────────────
1. Play a game using alpha-beta search with current weights.

2. At each turn t, the alpha-beta search produces a minimax value
   V(t) = eval(best_leaf). This is our PREDICTION of the position.

3. After the game, we know what ACTUALLY happened at each turn.
   The temporal difference error is:
     δ(t) = V(t+1) - V(t)
   If V jumped from 0.3 to 0.7, our turn-t prediction was too low.

4. We update weights to reduce these prediction errors:
     w += α × Σ_t [features(t) × Σ_{k≥t} λ^(k-t) × δ(k)]
   
   This is the TD(λ) update rule. λ controls the bias-variance
   tradeoff:
     λ=0: Only learn from the next turn's prediction (low variance,
          high bias — doesn't see far ahead)
     λ=1: Learn from the full game outcome (high variance, low bias
          — Monte Carlo, noisy but unbiased)
     λ=0.7: Good compromise — learns from ~3-4 turns ahead

5. The "Leaf" part: V(t) comes from the LEAF node of the minimax
   tree, not the root. This means the eval is trained on the
   positions that alpha-beta actually evaluates, not the positions
   it plays from. This makes training more stable.

REFERENCE: Baxter, Tridgell, Weaver (2000) "Learning to Play Chess
Using Temporal Differences" — the KnightCap paper.
"""

import sys
from pathlib import Path
import os
import time
import random
import numpy as np

# Support both module execution and direct execution from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import (
    place, move, eat, cascade, is_valid_coord,
    UP, DOWN, LEFT, RIGHT, MOVE, EAT, CASCADE
)
from Agents.archive.agent_td.features import extract_features, evaluate_linear, NUM_FEATURES, DEFAULT_WEIGHTS

OFFSETS = (UP, DOWN, LEFT, RIGHT)
INF = 999999


# ──────────────────────────────────────────────
# Move generation (same as program.py)
# ──────────────────────────────────────────────
def generate_actions(board, is_red):
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
            if tv == 0: actions.append((MOVE, idx, offset, 0))
            elif tv & my_flag: actions.append((MOVE, idx, offset, 1))
            elif tv & opp_flag:
                th = tv & HEIGHT_MASK
                if h >= th: actions.append((EAT, idx, offset, 10 + th))
        if h >= 2:
            for offset in OFFSETS:
                actions.append((CASCADE, idx, offset, 5))
    actions.sort(key=lambda x: x[3], reverse=True)
    return actions


def apply_action(board, action_type, start_idx, offset):
    if action_type == MOVE: return move(board, start_idx, offset)
    elif action_type == EAT: return eat(board, start_idx, offset)
    elif action_type == CASCADE: return cascade(board, start_idx, offset)
    return board


def generate_placement_actions(board, is_red, pc):
    opp_flag = BLUE_FLAG if is_red else RED_FLAG
    actions = []
    for idx in range(64):
        if board.state[idx] != 0: continue
        if pc > 0:
            adj = False
            for off in OFFSETS:
                if is_valid_coord(idx, off) and (board.state[idx + off] & opp_flag):
                    adj = True; break
            if adj: continue
        actions.append(idx)
    return actions


# ──────────────────────────────────────────────
# Alpha-beta search (simplified for training speed)
# ──────────────────────────────────────────────
def alphabeta(board, depth, alpha, beta, is_red_turn, is_red_max, weights):
    """
    Returns (value, leaf_features).
    leaf_features is the feature vector of the leaf node that
    determined the minimax value — this is what TD-Leaf trains on.
    """
    if board.red_tokens == 0:
        v = -INF if is_red_max else INF
        return v, None
    if board.blue_tokens == 0:
        v = INF if is_red_max else -INF
        return v, None
    if depth <= 0:
        features = extract_features(board, is_red_max)
        v = float(np.dot(weights, features))
        return v, features

    actions = generate_actions(board, is_red_turn)
    if not actions:
        return 0.0, np.zeros(NUM_FEATURES)

    is_max = (is_red_turn == is_red_max)
    best_features = None

    if is_max:
        val = -INF
        for act in actions:
            nb = apply_action(board, act[0], act[1], act[2])
            v, feats = alphabeta(nb, depth - 1, alpha, beta,
                                 not is_red_turn, is_red_max, weights)
            if v > val:
                val = v
                best_features = feats
            alpha = max(alpha, val)
            if alpha >= beta: break
        return val, best_features
    else:
        val = INF
        for act in actions:
            nb = apply_action(board, act[0], act[1], act[2])
            v, feats = alphabeta(nb, depth - 1, alpha, beta,
                                 not is_red_turn, is_red_max, weights)
            if v < val:
                val = v
                best_features = feats
            beta = min(beta, val)
            if alpha >= beta: break
        return val, best_features


def choose_best_action(board, is_red, weights, depth=3):
    """Pick the best action using alpha-beta with given weights."""
    actions = generate_actions(board, is_red)
    if not actions:
        return None, 0.0, None

    best_act = actions[0]
    best_val = -INF
    best_feats = None

    for act in actions:
        nb = apply_action(board, act[0], act[1], act[2])
        v, feats = alphabeta(nb, depth - 1, best_val, INF,
                             not is_red, is_red, weights)
        if v > best_val:
            best_val = v
            best_act = act
            best_feats = feats

    return best_act, best_val, best_feats


# ──────────────────────────────────────────────
# Play one self-play game, collect training data
# ──────────────────────────────────────────────
def play_training_game(weights, search_depth=3, max_turns=60):
    """
    Play a complete game, recording at each play-phase turn:
      - leaf_features: the feature vector of the minimax leaf
      - leaf_value: the minimax value (our prediction)
    
    Returns: (values, features_list, outcome)
      outcome: 1.0 = Red wins, 0.0 = Blue wins, 0.5 = draw
    """
    board = BoardState()
    placement_count = 0
    is_red_turn = True

    # Placement phase: random center-ish placements
    EDGE_DIST = [min(r, 7-r, c, 7-c) for r in range(8) for c in range(8)]
    for _ in range(8):
        legal = generate_placement_actions(board, is_red_turn, placement_count)
        # Weighted random: prefer center
        weights_p = [EDGE_DIST[idx] + 1 for idx in legal]
        total_w = sum(weights_p)
        weights_p = [w / total_w for w in weights_p]
        idx = random.choices(legal, weights=weights_p, k=1)[0]
        board = place(board, idx, is_red_turn)
        placement_count += 1
        is_red_turn = not is_red_turn

    # Play phase: collect data
    values = []          # V(t) for each turn
    features_list = []   # leaf features for each turn
    play_turns = 0

    while play_turns < max_turns:
        # Both players use the SAME weights (self-play)
        act, val, feats = choose_best_action(
            board, is_red_turn, weights, search_depth
        )

        if act is None:
            break  # stalemate

        # Record from Red's perspective (consistent across turns)
        if is_red_turn:
            values.append(val)
        else:
            values.append(-val)  # flip: Red's perspective

        if feats is not None:
            if not is_red_turn:
                # Features were from Blue's perspective, flip sign
                features_list.append(-feats)
            else:
                features_list.append(feats)
        else:
            features_list.append(np.zeros(NUM_FEATURES))

        # Apply action
        board = apply_action(board, act[0], act[1], act[2])
        play_turns += 1

        # Check terminal
        if board.red_tokens == 0:
            return values, features_list, 0.0  # Blue wins
        if board.blue_tokens == 0:
            return values, features_list, 1.0  # Red wins

        is_red_turn = not is_red_turn

    # Draw (turn limit or stalemate)
    if board.red_tokens > board.blue_tokens:
        return values, features_list, 1.0
    elif board.blue_tokens > board.red_tokens:
        return values, features_list, 0.0
    return values, features_list, 0.5


# ──────────────────────────────────────────────
# TD-Leaf(λ) weight update
# ──────────────────────────────────────────────
def td_leaf_update(weights, values, features_list, outcome,
                   alpha=0.01, lam=0.7):
    """
    Update weights using the TD-Leaf(λ) algorithm.
    
    Parameters:
      weights: current weight vector (modified in place)
      values: list of V(t) values from Red's perspective
      features_list: list of leaf feature vectors
      outcome: game result (1.0=Red wins, 0.0=Blue wins, 0.5=draw)
      alpha: learning rate (how much to adjust per game)
      lam: TD lambda (0=one-step, 1=Monte Carlo, 0.7=recommended)
    
    The update rule:
      For each turn t:
        δ(t) = V(t+1) - V(t)    (temporal difference error)
        
        eligibility = Σ_{k≥t} λ^(k-t) × δ(k)
        
        w += α × features(t) × eligibility
    
    The final V(T) is set to the actual game outcome (mapped to
    the eval's scale) so the last prediction is compared against
    ground truth.
    """
    T = len(values)
    if T < 2:
        return

    # Map outcome to eval scale for the final target
    # Use a moderate value rather than ±INF
    if outcome == 1.0:
        final_val = 50.0   # Red wins
    elif outcome == 0.0:
        final_val = -50.0  # Blue wins
    else:
        final_val = 0.0    # Draw

    # Append final value and compute TD errors
    extended_values = list(values) + [final_val]

    # Compute δ(t) = V(t+1) - V(t), clamped to avoid huge updates
    deltas = []
    for t in range(T):
        delta = extended_values[t + 1] - extended_values[t]
        delta = max(-20.0, min(20.0, delta))  # clamp
        deltas.append(delta)

    # Compute weight update
    weight_update = np.zeros(NUM_FEATURES, dtype=np.float64)

    for t in range(T):
        feats = features_list[t]
        if feats is None:
            continue

        # Eligibility: weighted sum of future TD errors
        eligibility = 0.0
        for k in range(t, T):
            eligibility += (lam ** (k - t)) * deltas[k]

        weight_update += feats * eligibility

    # Apply update with learning rate
    # Normalize by number of turns to keep updates stable
    weights += alpha * weight_update / T

    # Clamp weights to prevent divergence
    weights[:] = np.clip(weights, -50.0, 50.0)


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────
def train(num_games=2000, search_depth=2, alpha=0.01, lam=0.7,
          save_every=100, output_path=None):
    """
    Train evaluation weights through TD-Leaf(λ) self-play.
    
    Parameters:
      num_games: number of self-play games (2000 is good)
      search_depth: alpha-beta depth during training (2-3)
      alpha: learning rate (0.01 is conservative, 0.02 faster)
      lam: TD lambda (0.7 is standard)
      save_every: save weights every N games
      output_path: where to save learned_weights.npy
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'learned_weights.npy')

    weights = DEFAULT_WEIGHTS.copy()

    # Check if we can resume from existing weights
    if os.path.exists(output_path):
        weights = np.load(output_path)
        print(f"Resuming from {output_path}")
    else:
        print(f"Starting from default weights")

    print(f"Training: {num_games} games, depth={search_depth}, "
          f"α={alpha}, λ={lam}")
    print(f"Initial weights: {weights}")
    print()

    wins = draws = losses = 0
    start_time = time.time()

    for game in range(1, num_games + 1):
        # Play one game
        values, features_list, outcome = play_training_game(
            weights, search_depth, max_turns=60
        )

        if outcome == 1.0: wins += 1
        elif outcome == 0.0: losses += 1
        else: draws += 1

        # TD-Leaf update
        td_leaf_update(weights, values, features_list, outcome, alpha, lam)

        # Progress reporting
        if game % save_every == 0 or game == num_games:
            elapsed = time.time() - start_time
            rate = game / elapsed
            eta = (num_games - game) / rate if rate > 0 else 0

            print(f"Game {game:5d}/{num_games}: "
                  f"W={wins} D={draws} L={losses} "
                  f"({elapsed:.0f}s, {rate:.1f} games/s, "
                  f"ETA {eta:.0f}s)")
            print(f"  Weights: {np.array2string(weights, precision=2, suppress_small=True)}")

            # Save weights
            np.save(output_path, weights)
            print(f"  Saved to {output_path}")
            print()

            # Reset counters
            wins = draws = losses = 0

    print("Training complete!")
    print(f"Final weights: {weights}")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    # Default: 2000 games at depth 2 (fast, ~30 min)
    # For better results: 5000 games at depth 3 (~3 hours)
    import argparse
    parser = argparse.ArgumentParser(description='TD-Leaf(λ) training for Cascade')
    parser.add_argument('--games', type=int, default=2000, help='Number of training games')
    parser.add_argument('--depth', type=int, default=2, help='Alpha-beta search depth')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--lam', type=float, default=0.7, help='TD lambda')
    parser.add_argument('--save-every', type=int, default=100, help='Save interval')
    args = parser.parse_args()

    train(args.games, args.depth, args.alpha, args.lam, args.save_every)

