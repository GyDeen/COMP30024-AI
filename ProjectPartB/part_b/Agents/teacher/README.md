# Collect games from the teaching agent

The bundled teaching agent can play through the normal Python referee without
a browser or arena server. The adapter runs the existing JS worker, WASM and
trained weights from `battlefield_page/az/` in Node.js. It does not copy or
reimplement the teacher's search or neural network.

```text
teacher / agent_ab-q1
    → Python referee and action log
    → training.DataExtraction.extract_game
    → training.DataExtraction.extract_features
    → GameGeneration/runs/<run>/features/*.npz
```

From `part_b`, preview or generate the prepared six-game batch:

```bash
./.venv/bin/python -m GameGeneration.generate --config GameGeneration/teacher.toml --dry-run
./.venv/bin/python -m GameGeneration.generate --config GameGeneration/teacher.toml
```

The configuration plays `agent_ab-q1` against levels 3 and 4, then level 3
against level 4. Each pairing swaps colours. Edit its matchups to choose other
opponents. Available specifications are `Agents.teacher:Level0Agent`
through `Agents.teacher:Level5Agent`; `Agents.teacher` defaults
to level 3. The budgets match the bundled arena: 0, 50, 150, 400, 800 and 1600
simulations. Two teacher specifications enable teacher-versus-teacher games.

Node.js must be available on `PATH`; no npm packages or external service are
needed. This adapter was checked with the already installed Node.js 26.0.0.
All model assets are read from disk; the adapter does not fetch them over HTTP.

Completed games save ordinary referee logs, complete board archives in
`games/`, and our existing feature archives in `features/`. The `features`
field in each result identifies its training archive. Forfeits and unfinished
games produce no training archive. Outcomes are the actual final game result.
The teacher's displayed value estimates are not saved as targets.

These are newly generated games. The bundled model does not include the
teaching team's original training dataset. Repeated deterministic pairings can
produce identical games; changing the configured seed alone does not diversify
this teacher's normal choices. Check whole game trajectories before increasing
the batch size or splitting training and validation data.

The adapter keeps the worker alive for a game, forwards every accepted action
and retains its board history and search tree between moves. The arena currently
supplies a fresh snapshot when choosing a move, resetting the worker's search
tree and repetition history. Consequently, the same level need not produce the
same game through both interfaces.

The teacher adapter counts its Node.js CPU against the referee's remaining
whole-game allowance, in addition to Python CPU. Referee memory tracking covers
the Python process, not the Node.js child; `teacher.toml` disables that limit.
The configured wall timeout stops the entire match process group. These settings
are for data collection, not a claim of tournament resource parity.

Run the integration checks:

```bash
./.venv/bin/python -m unittest GameGeneration.tests.test_generation GameGeneration.tests.test_teacher -b -v
```
