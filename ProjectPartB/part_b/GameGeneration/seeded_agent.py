"""Referee-compatible adapter for an assigned agent and random seed."""

import importlib
import os
import random

import numpy as np

from referee.game import PlayerColor


class Agent:
    
    def __init__(self, color: PlayerColor, **referee):
        spec = os.environ[f"CASCADE_{color.name}_AGENT"]
        seed = int(os.environ[f"CASCADE_{color.name}_SEED"])
        random.seed(seed)
        np.random.seed(seed)
        module_name, separator, class_name = spec.partition(":")
        module = importlib.import_module(module_name)
        agent_class = getattr(module, class_name if separator else "Agent")
        self._agent = agent_class(color, **referee)
        # Some current agents reseed from the clock in their constructors.
        random.seed(seed)
        np.random.seed(seed)

    def action(self, **referee):
        """Return the configured agent's next action under the referee budget."""
        return self._agent.action(**referee)

    def update(self, color, action, **referee):
        """Notify the configured agent of an accepted move by either player."""
        return self._agent.update(color, action, **referee)
