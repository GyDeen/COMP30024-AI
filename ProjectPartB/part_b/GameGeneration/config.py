"""Experiment settings and colour-balanced match schedules."""

from dataclasses import dataclass
import math
from pathlib import Path
import random
import re
import tomllib


@dataclass(frozen=True)
class Matchup:
    """The two agent import specifications to compare in one matchup."""

    agent_a: str
    agent_b: str


@dataclass(frozen=True)
class GenerationConfig:
    """Game-generation settings; no evaluation scores or training rewards."""

    games_per_matchup: int
    swap_colours: bool
    base_seed: int
    cpu_seconds: float
    memory_mb: float
    wall_seconds: float
    matchups: tuple[Matchup, ...]


@dataclass(frozen=True)
class GameSpec:
    """A scheduled game's identity, agents, colours and random seeds."""

    game_id: str
    number: int
    matchup_index: int
    red_agent: str
    blue_agent: str
    red_seed: int
    blue_seed: int


def load_config(path: Path) -> GenerationConfig:
    """
    The file must specify all GenerationConfig fields. Agent specifications
    accept 'package.module' or 'package.module:ClassName'. An even game count
    is required when colours are swapped. Invalid or unknown settings raise
    ValueError.
    """
    with Path(path).open("rb") as source:
        data = tomllib.load(source)
    required = set(GenerationConfig.__dataclass_fields__)
    if set(data) != required:
        raise ValueError(f"Config fields: missing {sorted(required - set(data))}; "
                         f"unknown {sorted(set(data) - required)}")
    if type(data["games_per_matchup"]) is not int or data["games_per_matchup"] <= 0:
        raise ValueError("games_per_matchup must be a positive integer")
    if type(data["swap_colours"]) is not bool:
        raise ValueError("swap_colours must be true or false")
    if data["swap_colours"] and data["games_per_matchup"] % 2:
        raise ValueError("Colour swapping requires an even games_per_matchup")
    if type(data["base_seed"]) is not int or not 0 <= data["base_seed"] < 2**32:
        raise ValueError("base_seed must be an integer from 0 through 4294967295")
    for name in ["cpu_seconds", "memory_mb", "wall_seconds"]:
        value = data[name]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (name != "memory_mb" and value == 0):
            raise ValueError(f"{name} must be positive (memory_mb may be 0 to disable it)")
        data[name] = float(value)
    if not isinstance(data["matchups"], list) or not data["matchups"]:
        raise ValueError("At least one [[matchups]] entry is required")
    matchups = []
    for item in data["matchups"]:
        if not isinstance(item, dict) or set(item) != {"agent_a", "agent_b"}:
            raise ValueError("Each matchup requires exactly agent_a and agent_b")
        for spec in item.values():
            if not isinstance(spec, str) or re.fullmatch(
                r"[A-Za-z_]\w*(?:[.\-][A-Za-z_0-9]+)*(?::[A-Za-z_]\w*)?", spec
            ) is None:
                raise ValueError(f"Invalid agent import specification: {spec!r}")
            if spec.partition(":")[0] == "GameGeneration.seeded_agent":
                raise ValueError("Select a playing agent, not the seed adapter")
        matchups.append(Matchup(**item))
    data["matchups"] = tuple(matchups)
    return GenerationConfig(**data)


def plan_games(config: GenerationConfig, run_id: str) -> list[GameSpec]:
    """Return the ordered games and seeds to run for these settings.

    Adjacent colour-swapped games reuse each participant's seed. A new pair
    gets new seeds. Identical settings produce identical assignments and seeds;
    run_id only changes game identifiers.
    """
    rng = random.Random(config.base_seed)
    games = []
    for matchup_index, matchup in enumerate(config.matchups):
        for index in range(config.games_per_matchup):
            if not config.swap_colours or index % 2 == 0:
                seed_a, seed_b = rng.getrandbits(32), rng.getrandbits(32)
            reversed_colours = config.swap_colours and index % 2 == 1
            number = len(games) + 1
            games.append(GameSpec(
                game_id=f"{run_id}-game-{number:06d}",
                number=number,
                matchup_index=matchup_index,
                red_agent=matchup.agent_b if reversed_colours else matchup.agent_a,
                blue_agent=matchup.agent_a if reversed_colours else matchup.agent_b,
                red_seed=seed_b if reversed_colours else seed_a,
                blue_seed=seed_a if reversed_colours else seed_b,
            ))
    return games
