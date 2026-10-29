# TD(lambda) with PyTorch

Start with [01_weighted_evaluator.ipynb](01_weighted_evaluator.ipynb).
The notebook loads the existing feature dataset, represents the weighted
heuristic using `nn.Linear`, and inspects PyTorch's automatic gradients.

Current stage: model setup and forward/backward passes. TD(lambda) targets and
weight training have not been implemented yet. We will add them here using
PyTorch and TorchRL components, one step at a time.

Launch JupyterLab from `part_b`:

```bash
PYTHONPATH="$PWD" python -m jupyterlab --no-browser --ip=0.0.0.0
```

Open the notebook with **Python (Cascade)**. Its working directory is
`training/td_lambda`, so `../..` refers to the project root.
