"""Isolated PPO/auxiliary-phase composition with complete-cycle checkpoints."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from koikoi.published_train import PublishedTrainer
from scripts.collect_score_auxiliary import collect_outcomes, compact_evidence
from scripts.score_auxiliary import ScoreAuxiliary


@dataclass(frozen=True)
class AuxiliaryConfig:
    kind: str = "score"
    interval: int = 25
    matches: int = 128
    epochs: int = 6
    minibatch: int = 128
    max_steps: int = 448
    deal_seed: int = 105110001
    action_seed: int = 105210001
    reservoir_seed: int = 105310001
    order_seed: int = 105410001

    def __post_init__(self):
        if self.kind not in ("none", "win", "score"):
            raise ValueError("Unknown auxiliary condition")
        for key in ("interval", "matches", "epochs", "minibatch", "max_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Auxiliary sizes must be positive integers")
        for key in ("deal_seed", "action_seed", "reservoir_seed", "order_seed"):
            if (
                type(getattr(self, key)) is not int
                or not 0 <= getattr(self, key) < 2**31
            ):
                raise ValueError("Invalid auxiliary seed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {
        name: sha(name)
        for name in (
            "scripts/score_auxiliary.py",
            "scripts/collect_score_auxiliary.py",
            "scripts/score_auxiliary_trainer.py",
        )
    }


class ScoreAuxiliaryTrainer:
    def __init__(self, ppo, config):
        if type(ppo) is not PublishedTrainer or not isinstance(config, AuxiliaryConfig):
            raise TypeError("Expected published PPO and auxiliary configuration")
        if ppo.config.score_temperature != 0.05 or not ppo.config.endgame:
            raise ValueError(
                "Auxiliary collection requires T0.05 and public endgame masks"
            )
        if ppo.counters["updates"]:
            raise ValueError("New auxiliary training requires a fresh PPO trainer")
        self.ppo, self.config = ppo, config
        self._attach()
        self.counters = dict.fromkeys(
            (
                "cycles",
                "decisions",
                "rounds",
                "matches",
                "retained_rows",
                "example_presentations",
                "optimizer_steps",
                "optimizer_steps_discard",
                "optimizer_steps_pick",
                "optimizer_steps_koi",
            ),
            0,
        )
        self.last_collection = None
        self.last_metrics = None
        self._complete = True

    def _attach(self):
        self.auxiliary = ScoreAuxiliary(self.ppo.model.actor)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.auxiliary.actor.parameters(), "lr": self.ppo.config.lr},
                {
                    "params": self.auxiliary.heads.parameters(),
                    "lr": self.ppo.config.value_lr,
                },
            ],
            eps=1e-5,
        )
        self.source_hashes = sources()
        self._config = asdict(self.config)

    def _check_complete(self):
        if not self._complete:
            raise RuntimeError(
                "Incomplete update; restore the last complete checkpoint"
            )
        if asdict(self.config) != self._config:
            raise ValueError("Auxiliary configuration changed")
        expected = self.ppo.counters["policy_updates"] // self.config.interval
        if self.counters["cycles"] != expected:
            raise ValueError("Auxiliary/PPO cycle counts disagree")

    def update(self):
        self._check_complete()
        self._complete = False
        metrics = self.ppo.update()
        completed = self.ppo.counters["policy_updates"]
        if not metrics["critic_only"] and completed % self.config.interval == 0:
            self.last_metrics = self._auxiliary_phase()
        self._complete = True
        self._check_complete()
        return metrics | {"auxiliary_cycles": self.counters["cycles"]}

    def _auxiliary_phase(self):
        c = self.config
        cycle = self.counters["cycles"]
        batch = collect_outcomes(
            self.ppo.model.actor,
            count=c.matches,
            deal_seed=c.deal_seed + cycle * c.matches,
            action_seed=c.action_seed + cycle,
            reservoir_seed=c.reservoir_seed + cycle,
            max_steps=c.max_steps,
        )
        self.last_collection = compact_evidence(batch)
        self.last_collection["actor_sha256"] = self.ppo.model.actor.checkpoint()[
            "current"
        ]["parameters_sha256"]
        for key, count in batch["work"].items():
            self.counters[key] += count
        n = len(batch["targets"])
        self.counters["retained_rows"] += n
        totals = dict.fromkeys(("outcome_nll", "within_outcome_nll", "clone_kl"), 0.0)
        presentations = 0
        if c.kind != "none":
            generator = torch.Generator(device="cpu").manual_seed(c.order_seed + cycle)
            for _ in range(c.epochs):
                for ids in torch.randperm(n, generator=generator).split(c.minibatch):
                    features, phases, legal, targets, reference = (
                        batch[key][ids].to(self.ppo.config.device)
                        for key in (
                            "features",
                            "phases",
                            "legal",
                            "targets",
                            "reference_log_probs",
                        )
                    )
                    loss, metrics = self.auxiliary.auxiliary_loss(
                        features, phases, legal, targets, reference, c.kind
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite auxiliary loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.auxiliary.parameters(), 0.5, error_if_nonfinite=True
                    )
                    self.optimizer.step()
                    self.counters["optimizer_steps"] += 1
                    for name, selected in (
                        ("discard", phases == 0),
                        ("pick", (phases == 1) | (phases == 2)),
                        ("koi", phases == 3),
                    ):
                        self.counters["optimizer_steps_" + name] += int(selected.any())
                    for key, value in metrics.items():
                        totals[key] += float(value) * len(ids)
                    presentations += len(ids)
        self.counters["example_presentations"] += presentations
        self.counters["cycles"] += 1
        return {
            "kind": c.kind,
            "cycle": cycle + 1,
            "retained_rows": n,
            "example_presentations": presentations,
            "mean_training_losses": {
                key: value / presentations for key, value in totals.items()
            }
            if presentations
            else None,
            "collection_work": batch["work"],
        }

    def save(self, directory):
        """Immutable two-file checkpoint; manifest written last marks completion."""
        self._check_complete()
        if sources() != self.source_hashes:
            raise ValueError("Auxiliary source changed during training")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.ppo.save(directory / "ppo.pt")
        payload = {
            "format": "score_auxiliary_state_v1",
            "config": asdict(self.config),
            "heads": self.auxiliary.heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "counters": self.counters,
            "last_collection": self.last_collection,
            "last_metrics": self.last_metrics,
        }
        torch.save(payload, directory / "auxiliary.pt")
        manifest = {
            "format": "score_auxiliary_checkpoint_v1",
            "config": asdict(self.config),
            "ppo_counters": self.ppo.counters,
            "auxiliary_counters": self.counters,
            "sources": self.source_hashes,
            "files": {
                name: sha(directory / name) for name in ("ppo.pt", "auxiliary.pt")
            },
        }
        with (directory / "manifest.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def restore(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["format"] != "score_auxiliary_checkpoint_v1"
            or manifest["sources"] != sources()
        ):
            raise ValueError("Checkpoint format/source mismatch")
        if set(manifest["files"]) != {"ppo.pt", "auxiliary.pt"} or any(
            sha(directory / name) != digest
            for name, digest in manifest["files"].items()
        ):
            raise ValueError("Checkpoint file digest mismatch")
        payload = torch.load(
            directory / "auxiliary.pt", map_location="cpu", weights_only=False
        )
        if (
            payload["format"] != "score_auxiliary_state_v1"
            or payload["config"] != manifest["config"]
        ):
            raise ValueError("Auxiliary state/config mismatch")
        obj = cls.__new__(cls)
        obj.ppo = PublishedTrainer.restore(directory / "ppo.pt")
        obj.config = AuxiliaryConfig(**payload["config"])
        obj._attach()  # Zero initialization preserves the restored CPU RNG.
        obj.auxiliary.heads.load_state_dict(payload["heads"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.counters = payload["counters"]
        obj.last_collection, obj.last_metrics = (
            payload["last_collection"],
            payload["last_metrics"],
        )
        obj._complete = True
        if (
            obj.ppo.counters != manifest["ppo_counters"]
            or obj.counters != manifest["auxiliary_counters"]
        ):
            raise ValueError("Checkpoint counters disagree")
        obj._check_complete()
        return obj
