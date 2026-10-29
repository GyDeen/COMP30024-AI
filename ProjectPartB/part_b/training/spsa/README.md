# SPSA training with PyTorch

Start with [01_training.ipynb](01_training.ipynb). It shows one real training
iteration using the project's existing game runner and standard PyTorch tools.

This is **game-based parameter tuning**: its weight updates come from newly
played matches. It does not fit the saved feature dataset. The archive loaded
in the notebook is only used to illustrate a forward pass. Agents choose all
their moves, including the initial placements, during each match.

| Component | Implementation |
|---|---|
| Evaluator coefficients | `nn.Linear(8, 1, bias=False)` |
| Parameter update | `torch.optim.SGD(maximize=True)` |
| Learning-rate decay | `torch.optim.lr_scheduler.LambdaLR` |
| Save and resume | `state_dict`, `torch.save`, `torch.load` |
| Probe and gradient estimate | Small SPSA-specific tensor operations |
| Match evaluation | Existing referee, agents and feature extraction |

PyTorch does not include an SPSA optimiser. Match outcomes come from external
processes, so autograd cannot differentiate them. We estimate the gradient from
two weight variants and assign it to `model.weight.grad` before calling the
built-in SGD optimiser. There is no fabricated differentiable loss and no
`backward()` call on a disconnected match score.

## Package layout

```text
spsa/
    01_training.ipynb   Visible steps for one training iteration
    model.py            Linear layer and named weight conversion
    optimizer.py        SPSA probes, gradient estimate and SGD update
    matches.py          Paired games, results and action CPU measurements
    config.py           Typed experiment settings
    config.toml         Starting weights, opponents and budgets
    trainer.py          Repeated iterations and PyTorch checkpoints
    __main__.py         Command-line entry point
```

The model uses CPU float64 coefficients with shape \((1,8)\), retaining the
previous trainer's coefficient precision. The game agent receives named Python
floats and continues using the existing search and feature implementations.
A feature batch used for a model forward pass must also be float64, with shape
\((N,8)\).

## Run in Jupyter

From `part_b` in the dev container:

```bash
PYTHONPATH="$PWD" python -m jupyterlab --no-browser --ip=0.0.0.0
```

Open the notebook with **Python (Cascade)**. The working directory is
`training/spsa`. Paths in its cells are relative to that directory.
The game cell runs the actual configured matches. With the default settings,
one iteration plays eight games. Choose a fresh output name for a new run.

Inspect the model weights, the probe vectors, the match scores and the estimated
gradient before running the optimiser step. The notebook shows `zero_grad`,
`.grad` assignment, `step` and the scheduler explicitly. Its checkpoints can be
continued by the command-line trainer with the same configuration.

## Command-line training

```bash
python -m training.spsa train --output logs/spsa-torch-001 --iterations 1
python -m training.spsa train --output logs/spsa-torch-001 --resume --iterations 20
```

`--iterations` is the total desired count, including completed updates. The
configuration defaults to [config.toml](config.toml). Material is held fixed to
anchor the score scale, alignment stays disabled, and the other six weights are
tuned. Both probes use matched opponent assignments and seeds, with
adjacent colour swaps. The control uses the original weights and the same search
settings. Additional normal project agents can be configured as opponents.

`move_seconds` caps wall time per search. `remaining_divisor` controls the
remaining-clock allocation and `max_depth` caps nominal iterative deepening.
Quiescence may extend beyond that depth. The referee's `cpu_seconds` allowance
is separate and applies to each agent over the entire game. Instrumented action
CPU usage excludes updates and process startup.

## SPSA-specific calculation

For mean match score and random signs on the tunable coordinates:

\[
J(\mathbf w)=\frac{W+\tfrac12D}{N},\qquad
\widehat{\mathbf g}_k=
\frac{J(\mathbf w^+)-J(\mathbf w^-)}{2c_k}\boldsymbol\Delta_k.
\]

The estimated gradient goes into SGD with `maximize=True`. Near a weight bound,
the probe center moves inward to keep both probes symmetric. After the update,
tunable coordinates are projected onto the configured bounds. Frozen coordinates
have zero gradients and remain fixed. Equal scores give a zero gradient; an
inward adjustment of the probe center may still occur at a boundary.

Repeating deterministic games or changing only their seeds does not provide
independent strategic examples.
Time-limited search can also change completed depths between repeated games.

## Checkpoints and validation

Each run saves `config.json`, `checkpoint.pt` and the exported `weights.json`.
The PyTorch checkpoint contains model, optimiser and scheduler state dictionaries,
the completed iteration count and the experiment configuration. An interrupted
iteration can reuse completed games. Failures and forfeits are retained and stop
training before an update is committed. Use one process per output directory.

The new `.pt` checkpoints are for this PyTorch trainer. Start a new run rather
than resuming an earlier NumPy/JSON checkpoint.

Evaluate frozen weights using a separate seed schedule:

```bash
python -m training.spsa evaluate \
  --weights logs/spsa-torch-001/checkpoint.pt \
  --output logs/spsa-torch-001-validation
```

`weights.json` is also accepted. Per-game attempts retain referee logs, console
output, raw board archives and extracted features. A single favourable batch
does not establish improved playing strength. Compare
W/D/L results and CPU use across varied games.
