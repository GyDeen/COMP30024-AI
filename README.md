# COMP30024 Artificial Intelligence — Cascade

A two-part artificial intelligence project completed for **COMP30024 Artificial Intelligence at the University of Melbourne (Semester 1, 2026)**.

The project applies classical search techniques to the game of Cascade:

- **Part A:** a single-player solver based on A* search.
- **Part B:** a competitive game-playing agent using time-bounded adversarial search.

## Tech stack

- Python 3
- A* search and priority queues
- Iterative deepening and alpha-beta pruning
- Quiescence search
- Transposition tables
- Hand-designed evaluation functions
- Python `unittest`
- Course-provided Cascade referee API

The project does not use a machine-learning framework. Its focus is efficient state-space search, heuristic design and decision-making under time and memory constraints.

## Part A — Single-player Cascade

Part A finds a sequence of `MOVE`, `EAT` and `CASCADE` actions that removes all blue stacks from an (8 \times 8) board.

The solver uses A* search:

$$
f(n) = g(n) + h(n),
$$

where (g(n)) is the number of actions taken and (h(n)) estimates the remaining work.

### Implementation highlights

- Encodes each board as an immutable 64-element tuple for efficient hashing.
- Uses a priority queue to expand the state with the lowest estimated cost.
- Records the best known cost for each state and discards dominated paths.
- Applies deterministic action priorities when candidates have equal scores.
- Reconstructs the final solution from a compact internal action path.
- Uses a distance-based heuristic to guide red stacks toward blue targets.

Run Part A by piping a CSV board into the search module:

```bash
cd ProjectPartA
python -m part_a.search < input.csv
```

## Part B — Adversarial game-playing agent

The main Part B implementation is **`agent_ab-q1`**. It plays Cascade against another agent while respecting the referee's CPU-time and memory limits.

### Search architecture

The agent combines:

- **Iterative deepening:** repeatedly searches to greater depth while preserving the best move from the last completed iteration.
- **Alpha-beta pruning:** avoids branches that cannot affect the final decision.
- **Transposition tables:** caches exact values and upper/lower bounds for previously explored positions.
- **Cached move ordering:** examines the previously best move first to improve pruning.
- **Tactical move ordering:** prioritises `EAT`, then `CASCADE`, then normal `MOVE` actions.
- **Adaptive time management:** derives a per-turn search budget from the remaining match time.
- **Safe fallback behaviour:** always returns a legal move if a deeper search cannot finish.

At a normal leaf, the agent evaluates the board using a weighted combination of material, safety, threats, positioning, stack efficiency, hunting pressure and movement control.

### Selective quiescence search

The agent extends leaf nodes through immediately forcing `EAT` actions before evaluating the position. This reduces the horizon effect when a material exchange is already in progress.

`CASCADE` actions are intentionally excluded from quiescence search. A cascade is available whenever a stack has height (h \ge 2), so cascades exist in a large proportion of positions. Treating every cascade as a noisy move would keep many positions permanently unstable, greatly increase the branching factor and undermine the purpose of quiescence search.

This design therefore favours a narrow definition of tactical instability: immediate captures are resolved, while broader cascade tactics remain part of the regular alpha-beta search.

### Opening strategy

During the placement phase, the agent uses a rule-based strategy rather than searching the rapidly expanding transition into the movement phase. It attempts to build a central L-shaped formation while accounting for legal cells, distance from opposing stacks and fallback positions.

This gives the agent:

- fast and deterministic opening decisions;
- central influence;
- aligned stacks with future cascade potential;
- a controlled transition into adversarial search.

### Performance-oriented representation

- Each board is stored in a 64-byte `bytearray`.
- Colour and stack height are encoded with bit flags.
- Moves are packed into integers containing the action type, source and target.
- Immutable board bytes are used as transposition-table keys.
- Neighbour and distance information is precomputed where practical.
- Shallow transposition entries are removed to control memory use.

## Strengths

- Combines several complementary search optimisations rather than relying on raw depth.
- Uses compact board and move representations suited to a large search tree.
- Makes a deliberate trade-off between tactical accuracy and quiescence branching.
- Separates placement strategy from movement-phase search.
- Uses an explainable, feature-based evaluation function.
- Accounts for material trades, immediate danger, spatial control and endgame priorities.
- Preserves multiple experimental agents, showing the evolution of search and heuristic ideas.
- Includes unit tests for placement, movement, eating, merging and cascading rules.

## Weaknesses and future improvements

- **Limited experimental evidence:** the repository does not summarise tournament win rates, average search depth, node counts or timing results.
- **Hand-tuned evaluation:** heuristic weights were manually selected and would benefit from systematic self-play optimisation.
- **Predictable opening:** the deterministic L-shaped placement strategy may be exploitable by an opponent designed specifically against it.
- **Narrow quiescence scope:** excluding cascades controls branching effectively, but some forcing cascade sequences can still cross the regular search horizon.
- **Basic emergency fallback:** if depth one cannot finish, the first generated legal move is selected instead of a fast one-ply heuristic choice.
- **Duplicated experiments:** many complete agent variants make the repository difficult to navigate and maintain.
- **Incomplete search testing:** the existing tests focus on game mechanics rather than pruning correctness, timeout behaviour, transposition bounds and heuristic edge cases.
- **Reproducibility:** benchmark scripts, pinned dependencies and a consolidated experiment report would make results easier to verify.

## Running Part B

From the Part B package directory, run the primary agent against another included agent:

```bash
cd ProjectPartB/part_b
python -m referee agent_ab-q1 agent_alphabeta
```

Run the available action tests:

```bash
python -m unittest tests/test_actions.py
```

Use the referee help command to view time, memory, logging and display options:

```bash
python -m referee --help
```

## Repository structure

```text
COMP30024-AI/
├── ProjectPartA/
│   └── part_a/search/        # A* solver and heuristic
├── ProjectPartB/
│   └── part_b/
│       ├── agent_ab-q1/      # Main game-playing agent
│       ├── agent_*/          # Experimental agent variants
│       ├── referee/          # Local match runner
│       └── tests/            # Game-mechanics tests
└── README.md
```

## Project report

The original report and experiment discussion are available in [Google Docs](https://docs.google.com/document/d/1yDYI_FLAxAkE17ZgcFnsJ6eOmdo9PLEGtZFjbeRr3-s/edit?usp=sharing).

> This repository is an academic project. Course framework code and specifications remain subject to their original ownership and usage conditions.
