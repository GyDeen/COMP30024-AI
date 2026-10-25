# Repeatable Cascade game generation

This folder runs configured matches through the existing referee and records
what happened. Valid completed games also become compact datasets through the
existing `DataExtraction` module. It does not train agents or assign outcome
scores. Agent strategies, shared mechanics and referee source are unchanged.

## 1. Preview the games before running

From `part_b`:

```bash
cd "/Users/deenzhou/Documents/Unimelb/COMP30024 AI/ProjectPartB/part_b"
./.venv/bin/python -m GameGeneration.generate --dry-run
```

This displays the agents, colours and seeds without starting a game or creating
a run folder. The default configuration is `GameGeneration/config.toml`.

## 2. Configure the batch

```toml
games_per_matchup = 2
swap_colours = true
base_seed = 42
cpu_seconds = 180.0
memory_mb = 250.0
wall_seconds = 600.0

[[matchups]]
agent_a = "Agents.agent_baseline"
agent_b = "Agents.agent_ab-q1-with-random"
```

| Setting | Meaning |
|---|---|
| `games_per_matchup` | TOTAL games for each listed matchup, not games per colour |
| `swap_colours` | Alternate which participant is Red; requires an even game count |
| `base_seed` | Determines the planned per-agent seeds; an unsigned 32-bit integer |
| `cpu_seconds` | Referee CPU allowance per agent over the whole game |
| `memory_mb` | Referee memory allowance per agent; zero disables it |
| `wall_seconds` | Maximum elapsed time for the entire match, including startup |
| `matchups` | Agent pairings, each with an `agent_a` and `agent_b` |

With two games and colour swapping enabled, the first game has A as Red and B
as Blue; the second reverses those colours. Each participant keeps its seed
across that pair. The next pair receives new seeds. The schedule is repeatable
for unchanged settings, including the order of matchup entries.

Add more `[[matchups]]` entries for more opponents. A specification may be a
package exporting `Agent` or a module/class such as
`Agents.agent_td.program:Agent`. Using the same agent for A and B enables
self-play. This version runs games sequentially to avoid CPU contention.

Memory measurement is provided by the referee on Linux. It is unavailable on
macOS; configuring a limit does not make that platform enforce it. CPU time and
wall time are distinct: a sleeping or stalled process may use little CPU while
still exceeding the match's wall-time limit. Recorded duration is wall time,
not measured agent CPU usage.

## 3. Generate the batch

```bash
./.venv/bin/python -m GameGeneration.generate \
  --config GameGeneration/config.toml \
  --output GameGeneration/runs/pilot-001
```

The output directory must be new. Existing runs are never overwritten or
resumed. If `--output` is omitted, a unique dated directory is created under
`GameGeneration/runs`. No new package installation is needed.

The default real-agent pair can take several minutes. Start with two games
before increasing the count. Reducing the CPU budget changes the agent's
search behaviour and can cause forfeits; it is not equivalent to a full-budget
performance evaluation.

`GameGeneration/pilot.toml` provides a shorter baseline-versus-itself pipeline
check with a 12-second CPU budget per agent. Use it to check the workflow, not
to assess full-budget strength or collect a diverse training set.

## 4. What a run saves

```text
pilot-001/
    config.toml                 Settings used for this run
    manifest.json               Run ID, start time and configuration filename
    schedule.json               Planned games, agents, colours and seeds
    results.jsonl               One raw result per attempted game
    summary.json                Counts and progress, without outcome scores
    games/
        game-000001.log          Referee's game events
        game-000001.console.txt  Referee and agent diagnostics
        game-000001.npz          Board-history dataset, for valid normal games
        ...
```

The `.npz` archives have exactly the existing extraction format: `game_id`,
`winner`, `ending_reason` and the `uint8` board array. Game IDs include the run
identifier so different runs do not accidentally share training-game IDs.

Each `results.jsonl` record stores:

| Field | Meaning |
|---|---|
| `game_id`, `number`, `matchup_index` | Game identity, one-based game number, zero-based matchup index |
| `red_agent`, `blue_agent` | Actual playing agents assigned to the colours |
| `red_seed`, `blue_seed` | Random seed assigned to each agent |
| `status` | Completed, forfeited or failed execution category |
| `winner` | Verified natural winner or referee-declared forfeit winner; otherwise null |
| `reported_winner` | Logged winner, even if the game later fails validation |
| `ending_reason` | Natural ending or forfeit cause; otherwise null |
| `turns` | Accepted actions counted from the log; null if not determined after an interrupted/aborted launch |
| `return_code` | Referee process exit code; null if it could not be launched |
| `wall_seconds` | Elapsed match duration, including startup and shutdown |
| `error` | Failure description, or null |
| `log`, `console`, `positions` | Paths relative to this run; positions is null without a valid saved dataset |

Status meanings:

- `completed`: naturally finished, replay-verified game with a saved dataset.
- `forfeit`: the referee awarded a winner after an illegal action, agent error,
  CPU limit, memory limit or agent response timeout. No training dataset is saved.
- `wall_timeout`: the whole match exceeded the configured wall limit.
- `referee_error`: the referee failed or did not produce its expected log.
- `invalid_log`: a final result or game trajectory could not be validated.
- `launch_error`: the match could not be started or its console output created.
- `dataset_error`: the game finished naturally but its dataset could not be saved.
- `interrupted`: the user interrupted a running match.

The raw log and console files are retained where available. A forfeit win is
not silently treated as a naturally won board position. A missing result is
not treated as a draw. Results are flushed and progress is saved after each
game, so earlier completed games remain available after interruption.

`summary.json` reports the number scheduled, number attempted, status counts,
and Red/Blue/draw counts for completed games. Colour totals are not yet an
agent ranking. In particular, self-play records do not measure the agent's
strength against other agents.

## 5. Seeds and repeatability

The referee runs each agent in a separate process. `seeded_agent.py` supplies
the configured real agent and seed through that existing interface. The
adapter seeds Python's global random generator and NumPy's legacy global
generator before construction and resets them afterwards. That handles the
current random agent's constructor reseeding from the clock.

Seeds are not a guarantee of identical games: custom private random generators,
later reseeding and search decisions affected by timing can still differ.
Deterministic agents may also play identical games despite different seeds.
The generator does not inject exploratory moves into frozen playing agents.
Use multiple opponents and inspect game diversity before treating a batch as
a useful training collection.

The run manifest records only its identifier, start time and configuration
filename. Per-game records identify the agents, colours, seeds, duration and
outcome. Git revisions, source changes and environment versions are not tracked.

## 6. Evaluation scores and training rewards

Generation records raw outcomes only. `evaluation_scores` belongs to a future
evaluation configuration, and `training_rewards` belongs to the trainer. They
must remain separate even if their initial numeric values match. Neither is
accepted as a game-generation setting or baked into the result records.

The saved outcomes allow later rescoring without replaying games. An evaluator
must state its treatment of forfeits and keep runner failures visible rather
than silently converting them into wins, losses or draws. Evaluation comparisons
should use fixed scoring settings and equal resource budgets.

## 7. Function purposes

- `load_config(path)` returns validated settings for a generation experiment.
- `plan_games(config, run_id)` returns the exact agent/colour/seed assignments.
- `run_match(spec, config, output_dir)` produces one game's artifacts and raw result.
- `read_outcome(log_path, return_code, game_id)` identifies its result and returns
  a dataset only for a valid naturally finished game.
- `generate(config_path, output_dir)` produces a complete batch directory and
  returns the progress summary.
- `summarize(results, scheduled, interrupted)` reports raw counts without assigning
  performance scores or training rewards.

Command exit status is zero when every scheduled game completed naturally,
one when matches failed or forfeited, two for invalid configuration/output,
and 130 after user interruption. A completed process is not by itself evidence
that every game was usable: inspect `summary.json` and the raw results.

## 8. Run the checks

```bash
./.venv/bin/python -m unittest GameGeneration.tests.test_generation -b -v
```

The checks include actual referee subprocess games with fast fixture agents,
colour/seed pairing, global RNG seeding, compact datasets, illegal actions,
agent crashes, wall timeouts and incomplete-log handling. Fixture agents are
only for runner verification; they are not performance opponents.
