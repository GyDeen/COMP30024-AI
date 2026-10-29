# Cascade training algorithms

Each training algorithm has its own Python package. Its implementation,
configuration, notebooks and documentation belong inside that package.

```text
training/
    DataExtraction/          Shared game and feature extraction
    baseline.py              Shared baseline feature order and weights
    td_lambda/               PyTorch learning workflow
        __init__.py
        README.md
        01_weighted_evaluator.ipynb
    spsa/                    SPSA using PyTorch
        __init__.py
        __main__.py
        01_training.ipynb
        model.py
        optimizer.py
        matches.py
        config.py
        trainer.py
        config.toml
        README.md
```

## Current learning workflow

Start with [the TD(lambda) package](td_lambda/README.md) and
[the evaluator notebook](td_lambda/01_weighted_evaluator.ipynb).
The notebook currently covers the PyTorch model, predictions and gradients.
Delayed-reward targets and actual TD training are the next steps.

From `part_b` in the dev container, launch JupyterLab:

```bash
PYTHONPATH="$PWD" python -m jupyterlab --no-browser --ip=0.0.0.0
```

Open the notebook with **Python (Cascade)**. JupyterLab starts its kernel in
`training/td_lambda`, and the notebook uses `../..` as the project root.
Its `feature_path` points to the recorded short-budget game against teacher
level 3. Change it to inspect another extracted feature archive.

## SPSA with PyTorch

The [SPSA notebook](spsa/01_training.ipynb) shows a full match-based update using
`nn.Linear`, `torch.optim.SGD`, `LambdaLR` and PyTorch checkpoints. The probes and
gradient estimate remain SPSA-specific. See the [package guide](spsa/README.md).

The command-line entry point is:

```bash
python -m training.spsa train --output logs/spsa-torch-001 --iterations 1
```

Its default configuration is now `training/spsa/config.toml`.
