# Shared agent mechanics

All agents now import their byte board and action rules from this package:

```python
from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import place, move, eat, cascade
```

`board.py` contains the baseline's existing board class and encoding constants.
`action.py` contains the existing transition functions and packed-move helpers.
Search, evaluation, placement strategies and time budgets stay in each agent.
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

Run from `part_b`, so the `Agents` and `referee` packages are visible:

```bash
./.venv/bin/python -m referee Agents.agent_baseline Agents.agent_ab-q1 -v 1
./.venv/bin/python -m unittest tests.test_actions tests.test_shared_game
```

The TD trainer also supports module execution:

```bash
./.venv/bin/python -m Agents.agent_td.td_training --help
```

The packed action representation retains its existing limitations: a direction
is represented by a neighbouring target cell. Use the primitive `cascade`
function for outward edge cascades. This extraction does not alter move rules
or search generators.
