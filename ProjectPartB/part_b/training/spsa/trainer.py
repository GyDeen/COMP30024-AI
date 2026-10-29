"""Train SPSA probes with a PyTorch model, SGD and state_dict checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

from GameGeneration.generate import save_json
from .config import SPSAConfig, load_config
from .matches import play_batch, schedule
from .model import load_weights, make_model, model_weights, to_weights
from .optimizer import apply_update, propose


def save_checkpoint(path: Path, model: nn.Linear, optimizer: torch.optim.SGD,
                    scheduler: torch.optim.lr_scheduler.LambdaLR,
                    config: SPSAConfig, completed_iterations: int) -> None:
    """Save PyTorch state dictionaries and export named weights beside them."""
    checkpoint = {
        "completed_iterations": completed_iterations,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    save_json(path.with_name("weights.json"), {
        "completed_iterations": completed_iterations, "weights": model_weights(model),
    })


def train(config: SPSAConfig, output: Path, *, resume: bool = False,
          iterations: int | None = None) -> dict[str, Any]:
    """Run paired matches and use their SPSA gradient estimates in PyTorch SGD.

    output is a new run directory, or an existing PyTorch run with resume=True.
    iterations is the total desired count, including already completed updates.
    """
    target = config.iterations if iterations is None else iterations
    if target < 1:
        raise ValueError("iterations must be positive")
    model = make_model(config.weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, maximize=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda iteration: (iteration + 1) ** (-config.learning_decay),
    )
    completed = 0
    if resume:
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != asdict(config):
            raise ValueError("Configuration differs from the saved run; use --iterations to extend it")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        completed = checkpoint["completed_iterations"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "config.json", asdict(config))
    save_checkpoint(output / "checkpoint.pt", model, optimizer, scheduler, config, completed)

    for iteration in range(completed, target):
        proposal = propose(model, config, iteration)
        directory = output / f"iteration-{iteration + 1:04d}"
        directory.mkdir(exist_ok=True)
        proposal_record = {
            "iteration": iteration + 1, "weights_before": model_weights(model),
            "center": to_weights(proposal.center), "plus": to_weights(proposal.plus),
            "minus": to_weights(proposal.minus), "direction": proposal.direction.tolist(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "perturbation": proposal.perturbation,
        }
        proposal_path = directory / "proposal.json"
        if proposal_path.exists() and json.loads(proposal_path.read_text()) != proposal_record:
            raise ValueError("Saved proposal differs from the PyTorch checkpoint")
        save_json(proposal_path, proposal_record)
        scores: dict[str, dict[str, Any]] = {}
        order = ("plus", "minus") if iteration % 2 == 0 else ("minus", "plus")
        for label in order:
            scores[label] = play_batch(
                to_weights(getattr(proposal, label)), config, directory / label, iteration,
            )
        gradient = apply_update(model, optimizer, proposal,
                                scores["plus"]["score"], scores["minus"]["score"], config)
        scheduler.step()
        completed = iteration + 1
        save_json(directory / "update.json", {
            **proposal_record, "results": scores, "estimated_gradient": gradient.tolist(),
            "weights_after": model_weights(model),
        })
        save_checkpoint(output / "checkpoint.pt", model, optimizer, scheduler, config, completed)
        print(f"Iteration {completed}: plus={scores['plus']['score']:.3f}, "
              f"minus={scores['minus']['score']:.3f}; PyTorch checkpoint saved", flush=True)
    return {"completed_iterations": completed, "weights": model_weights(model)}


def evaluate(config: SPSAConfig, weights: dict[str, float], output: Path) -> dict[str, Any]:
    """Evaluate fixed named weights on a separate seed schedule, without updating."""
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", asdict(config))
    save_json(output / "weights.json", {"weights": weights})
    return play_batch(weights, config, output / "validation", 0, validation=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, help="Total iterations, including completed ones")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weights", type=Path, help="weights.json or checkpoint.pt for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Preview scheduled games")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "evaluate" and (args.resume or args.iterations is not None):
            raise ValueError("--resume and --iterations apply only to training")
        if args.command == "train" and args.weights is not None:
            raise ValueError("Set initial weights in the configuration for a new training run")
        if args.command == "evaluate" and args.weights is None:
            raise ValueError("evaluate requires --weights")
        if args.dry_run:
            count = args.iterations if args.iterations is not None else config.iterations
            games = 2 * config.pairs_per_opponent * len(config.opponents)
            print(json.dumps({
                "configuration": asdict(config),
                "scheduled_games": games if args.command == "evaluate" else count * 2 * games,
                "example_schedule": schedule(config, 0, args.command == "evaluate"),
            }, indent=2))
        elif args.command == "train":
            train(config, args.output, resume=args.resume, iterations=args.iterations)
        else:
            print(json.dumps(evaluate(config, load_weights(args.weights), args.output), indent=2))
    except KeyboardInterrupt:
        print("Interrupted. Resume the run to reuse its completed games.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"SPSA: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
