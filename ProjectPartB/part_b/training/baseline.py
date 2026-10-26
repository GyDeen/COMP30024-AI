"""Shared feature order and fixed baseline weights for training experiments.

Weights are a snapshot of Agents.agent_ab-q1.heuristic.PROFILES. Rows follow
BASELINE_PROFILE_NAMES; columns follow FEATURE_NAMES. Update this reference
explicitly if the baseline changes. Copy weights before training modifies them.
"""

import numpy as np

from Agents.common.features.movement import FEATURE_NAMES


FEATURE_SOURCE = "Agents.common.features.movement"
PERSPECTIVES = ("RED", "BLUE")
BASELINE_PROFILE_NAMES = (
    "phalanx_balanced",
    "phalanx_aggressive",
    "hunt_aggressive",
    "hunt_balanced",
    "hunt_defensive",
    "endgame",
)
BASELINE_WEIGHTS = np.array([
    [0.40, 0.15, 0.30, 0.00, 0.10, 0.20, 0.10, 0.10],
    [0.40, 0.15, 0.10, 0.00, 0.20, 0.10, 0.20, 0.15],
    [0.25, 0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.10],
    [0.30, 0.10, 0.15, 0.20, 0.10, 0.20, 0.15, 0.10],
    [0.30, 0.20, 0.10, 0.15, 0.05, 0.25, 0.00, 0.10],
    [0.50, 0.05, 0.10, 0.00, 0.00, 0.15, 0.20, 0.00],
], dtype=np.float32)
BASELINE_WEIGHTS.setflags(write=False)
