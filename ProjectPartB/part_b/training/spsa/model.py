from collections.abc import Mapping
import json
from pathlib import Path

import torch
from torch import nn

from Agents.common.features.movement import FEATURE_NAMES
from Agents.common.features.weighted import validate_weights


def make_model(weights: Mapping[str, float]) -> nn.Linear:
    """Return a CPU float64 Linear(8, 1) with weights shaped (1, 8).

    Float64 retains the original trainer's coefficient precision. Feature
    batches passed to this layer should also be float64 with shape (N, 8).
    The playing agent receives exported Python floats and does not use Torch.
    """
    values = torch.tensor(list(validate_weights(weights).values()), dtype=torch.float64)
    model = nn.Linear(len(FEATURE_NAMES), 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(values.unsqueeze(0))
    return model


def to_weights(values: torch.Tensor) -> dict[str, float]:
    """Export an eight-element tensor as named Python floats for the agent."""
    return validate_weights(dict(zip(
        FEATURE_NAMES, values.detach().flatten().tolist(), strict=True,
    )))


def model_weights(model: nn.Linear) -> dict[str, float]:
    """Return an independent named snapshot of the layer's coefficients."""
    return to_weights(model.weight)


def load_weights(path: Path) -> dict[str, float]:
    """Read exported weights.json or a PyTorch training checkpoint.pt."""
    if path.suffix == ".pt":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return to_weights(checkpoint["model_state_dict"]["weight"])
    return validate_weights(json.loads(path.read_text())["weights"])
