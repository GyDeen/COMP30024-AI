"""Reuse the short-budget search with instance-local weights and search limits."""

import json
import math
import os
import time

from Agents.common.action import translate_to_Action
from Agents.common.features.weighted import make_evaluator
from Agents.tune_agent.short_budget_baseline.program import Agent as OriginalAgent
from Agents.tune_agent.short_budget_baseline import search
from referee.game import Action, PlayerColor

# Exact Python-float snapshot of q1's phalanx_aggressive profile.
INITIAL_WEIGHTS: dict[str, float] = {
    "material": 0.40, "positioning": 0.15, "efficiency": 0.10,
    "alignment": 0.0, "threaten_enemy": 0.20, "safety": 0.10,
    "hunting": 0.20, "control": 0.15,
}


class Agent(OriginalAgent):
    """Read CASCADE_TUNING_RED/BLUE JSON, or use the initial short-budget setup.

    Settings contain weights, move_seconds, remaining_divisor and max_depth.
    The recursive alpha-beta and quiescence implementations are reused intact.
    Search uses wall time, matching the existing engine. The referee separately
    enforces whole-game CPU limits. An action-level CPU measurement is logged.
    """

    fixed_weights = False

    def __init__(self, color: PlayerColor, **referee: object) -> None:
        super().__init__(color, **referee)
        settings = json.loads(os.environ.get(f"CASCADE_TUNING_{color.name}", "{}"))
        unknown = set(settings) - {"weights", "move_seconds", "remaining_divisor", "max_depth"}
        if unknown:
            raise ValueError(f"Unknown tuning settings: {sorted(unknown)}")
        weights = INITIAL_WEIGHTS if self.fixed_weights else settings.get("weights", INITIAL_WEIGHTS)
        self.evaluate = make_evaluator(weights)
        self.move_seconds = float(settings.get("move_seconds", 0.2))
        self.remaining_divisor = float(settings.get("remaining_divisor", 300.0))
        self.max_depth = settings.get("max_depth", 20)
        if any(not math.isfinite(x) or x <= 0 for x in (self.move_seconds, self.remaining_divisor)):
            raise ValueError("Search time settings must be finite and positive")
        if type(self.max_depth) is not int or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

    def action(self, **referee: object) -> Action:
        """Return the last fully completed iteration, with a legal fallback."""
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        self.manage_tt_mem()
        is_red = self._color == PlayerColor.RED
        remaining_value = referee.get("time_remaining", 180.0)
        remaining = float(remaining_value) if isinstance(remaining_value, (int, float)) else 180.0
        remaining = max(0.0, remaining - (time.process_time() - cpu_start))
        budget = min(self.move_seconds, remaining / self.remaining_divisor,
                     max(0.0, remaining - 0.01))
        completed_depth = 0
        if self.board.turn_count < 8:
            best_move = search._rule_based_placement(self.board, is_red)
        else:
            moves = search.generate_possible_moves(self.board, is_red)
            if not moves:
                raise RuntimeError("Referee requested an action with no legal moves")
            best_move = moves[0]
            start = time.perf_counter()
            for depth in range(1, self.max_depth + 1):
                try:
                    move, _score = search.search_all_move(
                        self.board, depth, float("-inf"), float("inf"),
                        int(is_red), True, is_red, start, budget, self.evaluate, self.tt,
                    )
                except TimeoutError:
                    break
                best_move = move
                completed_depth = depth
        action = translate_to_Action(best_move)
        print("TUNING_METRIC " + json.dumps({
            "colour": self._color.name, "turn_count": self.board.turn_count,
            "completed_depth": completed_depth, "action": str(action),
            "search_budget_seconds": budget,
            "cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.perf_counter() - wall_start,
        }), flush=True)
        return action


class BaselineAgent(Agent):
    """Same search and timing as Agent, always using the initial weights."""

    fixed_weights = True
