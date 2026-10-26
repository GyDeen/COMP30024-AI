# Shared agent mechanics

All agents now import their byte board and action rules from this package:

```python
from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import place, move, eat, cascade
```

`board.py` contains the baseline's existing board class and encoding constants.
`action.py` contains the existing transition functions and packed-move helpers.
`features/` contains shared feature calculations and geometry. Search, evaluation
profiles, placement strategies and time budgets stay in each agent.
Keep these shared mechanics stable during evaluator experiments: the baseline
now depends on the same shared files as the experimental agents.

Previously, four agents used a board without `turn_count`. They now share the
baseline board, including that field; their existing agent-level turn counters
remain unchanged. The extra field costs one slot per board for those agents.
Primitive `place`, `move`, `eat` and `cascade` operations copy the board and do
not increment its turn count. Packed `execute_move` increments it once, exactly
as it did for the baseline. This preserves the two calling conventions.

The per-agent `board.py` and `action.py` copies have been removed. Import the
shared modules directly in new code. `Agents.common` does not import a playing
agent, so the extractor can reuse its encoding constants without loading search.

## Shared heuristic features

The 15 agents with `heuristic.py` modules import matching feature calculations
from `Agents.common.features`:

- `geometry.py`: centre weights, distances and adjacency, independent of search.
- `movement.py`: baseline movement features, their column order, and `feature_vector`.
- `movement_variants.py`: different efficiency, positioning, safety and hunting
  formulas shared by subsets of agents.
- `placement.py` and `placement_variants.py`: shared placement measurements and
  the control-zone / compact-four alternatives.
- `closeout.py`: shared closeout and maximum-height-control features.

Agents alias their chosen variant to the existing function name. For example,
an agent can import `calc_efficiency_friendly_only as calc_efficiency` while the
baseline imports `movement.calc_efficiency`. The original argument lists,
weight gating and score arithmetic are retained. Unique formulas remain local;
the TD and bespoke `evaluation.py` implementations remain separate.

Dataset preparation uses the same baseline functions directly:

```python
from Agents.common.features.movement import FEATURE_NAMES, feature_vector

values = feature_vector(board, root_color=1, is_red_turn=True)
```

`values` is a tuple of eight unweighted numbers in `FEATURE_NAMES` order, with
no NumPy, PyTorch or agent-search dependency. The extractor converts it to its
chosen array dtype. Baseline weight snapshots stay in `training/baseline.py`;
changing a feature formula affects every agent that imports it.

Run from `part_b`, so the `Agents` and `referee` packages are visible:

```bash
./.venv/bin/python -m referee Agents.agent_baseline Agents.agent_ab-q1 -v 1
./.venv/bin/python -m unittest tests.test_actions tests.test_shared_game
./.venv/bin/python -m unittest Agents.common.features.test_features
```

The TD trainer also supports module execution:

```bash
./.venv/bin/python -m Agents.agent_td.td_training --help
```

The packed action representation retains its existing limitations: a direction
is represented by a neighbouring target cell. Use the primitive `cascade`
function for outward edge cascades. This extraction does not alter move rules
or search generators.
