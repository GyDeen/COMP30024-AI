"""Experiment configuration and recorded opening prefixes."""

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from Agents.common.features.movement import FEATURE_NAMES
from Agents.common.features.weighted import validate_weights
from Agents.tune_agent.spsa.program import INITIAL_WEIGHTS

CANDIDATE = "Agents.tune_agent.spsa:Agent"
CONTROL = "Agents.tune_agent.spsa:BaselineAgent"


@dataclass
class SPSAConfig:
    """Weights are named feature coefficients; all durations are seconds."""

    iterations: int = 20
    pairs_per_opponent: int = 2
    seed: int = 42
    opponents: list[str] = field(default_factory=lambda: [CONTROL])
    training_openings: list[list[str]] = field(default_factory=lambda: [[]])
    validation_openings: list[list[str]] = field(default_factory=lambda: [[]])
    weights: dict[str, float] = field(default_factory=lambda: dict(INITIAL_WEIGHTS))
    tune_features: list[str] = field(default_factory=lambda: [
        name for name in FEATURE_NAMES if name not in {"material", "alignment"}
    ])
    learning_rate: float = 0.01
    perturbation: float = 0.05
    learning_decay: float = 0.602
    perturbation_decay: float = 0.101
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    move_seconds: float = 0.2
    remaining_divisor: float = 300.0
    max_depth: int = 20
    cpu_seconds: float = 180.0
    wall_seconds: float = 600.0
    memory_mb: float = 0.0

    def __post_init__(self) -> None:
        for name in ("iterations", "pairs_per_opponent", "max_depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for name in ("learning_rate", "perturbation", "learning_decay", "perturbation_decay",
                     "move_seconds", "remaining_divisor", "cpu_seconds", "wall_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lower_bound", "upper_bound", "memory_mb"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.upper_bound - self.lower_bound <= 2 * self.perturbation:
            raise ValueError("Bounds must leave room for both perturbations")
        self.weights = validate_weights(self.weights)
        if (not self.tune_features or len(set(self.tune_features)) != len(self.tune_features)
                or not set(self.tune_features) <= set(FEATURE_NAMES)):
            raise ValueError("tune_features must be unique known feature names")
        if "material" in self.tune_features or self.weights["material"] <= 0:
            raise ValueError("Keep a positive material weight fixed to anchor the score scale")
        if any(not self.lower_bound <= self.weights[name] <= self.upper_bound
               for name in self.tune_features):
            raise ValueError("Initial tunable weights must lie inside the bounds")
        if (not isinstance(self.opponents, list) or not self.opponents
                or any(not isinstance(x, str) or not x for x in self.opponents)):
            raise ValueError("At least one opponent import specification is required")
        if CANDIDATE in self.opponents or "Agents.tune_agent.spsa" in self.opponents:
            raise ValueError("Use BaselineAgent as the control, not the trainable Agent")


def load_config(path: Path) -> SPSAConfig:
    """Load a TOML configuration, rejecting unknown fields."""
    with path.open("rb") as source:
        values = tomllib.load(source)
    unknown = set(values) - set(SPSAConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown SPSA fields: {sorted(unknown)}")
    return SPSAConfig(**values)


