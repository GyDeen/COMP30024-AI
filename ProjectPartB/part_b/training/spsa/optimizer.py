"""The small SPSA-specific gradient estimator, followed by PyTorch SGD."""

from dataclasses import dataclass

import torch
from torch import nn

from Agents.common.features.movement import FEATURE_NAMES

from .config import SPSAConfig


@dataclass(frozen=True)
class Proposal:
    """CPU float64 vectors of shape (8,); frozen coordinates have direction zero."""

    center: torch.Tensor
    plus: torch.Tensor
    minus: torch.Tensor
    direction: torch.Tensor
    perturbation: float


def propose(model: nn.Linear, config: SPSAConfig, iteration: int) -> Proposal:
    """Make symmetric probes for a zero-based iteration using Torch tensors.

    Move the probe center inward near bounds so both variants remain feasible.
    Frozen weights are neither perturbed nor projected.
    """
    center = model.weight.detach().flatten().clone()
    active = torch.tensor([name in config.tune_features for name in FEATURE_NAMES])
    generator = torch.Generator().manual_seed(config.seed + iteration)
    signs = torch.randint(0, 2, center.shape, generator=generator) * 2 - 1
    direction = signs.to(dtype=center.dtype) * active
    c = config.perturbation / (iteration + 1) ** config.perturbation_decay
    center[active] = center[active].clamp(config.lower_bound + c, config.upper_bound - c)
    return Proposal(center, center + c * direction, center - c * direction, direction, c)


def estimate_gradient(proposal: Proposal, plus_score: float, minus_score: float) -> torch.Tensor:
    """Estimate the gradient of mean match score; return float64 shape (8,).

    Match results are external to autograd. This estimate is supplied directly
    to Parameter.grad rather than calling backward() on a disconnected score.
    """
    return (plus_score - minus_score) / (2 * proposal.perturbation) * proposal.direction


def apply_update(model: nn.Linear, optimizer: torch.optim.SGD, proposal: Proposal,
                 plus_score: float, minus_score: float, config: SPSAConfig) -> torch.Tensor:
    """Supply the SPSA gradient to SGD(maximize=True), then enforce bounds."""
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.weight.copy_(proposal.center.unsqueeze(0))
    gradient = estimate_gradient(proposal, plus_score, minus_score)
    model.weight.grad = gradient.unsqueeze(0)
    optimizer.step()
    active = proposal.direction != 0
    with torch.no_grad():
        model.weight[0, active] = model.weight[0, active].clamp(
            config.lower_bound, config.upper_bound,
        )
    return gradient
