# Archived agents

These experimental agents were moved out of the active working set. Active
agents remain in `Agents/agent_ab-q1`, `Agents/agent_baseline` and `Agents/teacher`;
shared mechanics and features remain in `Agents/common`.

Import an archived agent explicitly through `Agents.archive.<folder>`. For example:

```bash
./.venv/bin/python -m referee Agents.agent_ab-q1 Agents.archive.agent_alphabeta
```

The arena lists active agent folders only. The game generator can still run
archived agents when their full import path is specified. Existing saved game
metadata retains the original paths used when those games were played.

Archived folders:

- `agent_11`
- `agent_135wings`
- `agent_ab-q`
- `agent_ab-q1-with-random`
- `agent_ab-with-quienscence`
- `agent_ab-with-quienscence1`
- `agent_ab-with-quienscence2`
- `agent_ab-with-quienscence3`
- `agent_ab-without-quienscence`
- `agent_ab-without-quienscence1`
- `agent_ab14`
- `agent_ab16`
- `agent_ab1wne`
- `agent_ab_with_quienscence_place3`
- `agent_alphabeta`
- `agent_faststupid`
- `agent_td`
