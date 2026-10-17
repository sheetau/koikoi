"""Fixed, update-free gradient diagnostic on two immutable PPO parents.

Measures raw first-order gradient interactions, not Adam updates or strength.
No checkpoint selection, teacher, evaluation deals or parameter updates.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.train import Config, Trainer, provenance

BASE = Path("runs/objective_gradient_diagnostic")
SOURCE = "1c36895f2b5164c4c0a86862e2270364a74c2ff0ee4d950f4b0c4cd970575914"
PARENTS = (
    (
        "seed1",
        "runs/ppo8_v3/update_00005000.pt",
        "d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5",
        6400001,
    ),
    (
        "seed10001",
        "runs/ppo8_seed10001_captured/update_00005000.pt",
        "edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d",
        6410001,
    ),
)
TERMS = ("policy", "entropy", "value", "card_policy", "koi_policy")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vectors(losses, parameters):
    result = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        result[name] = torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for p, g in zip(parameters, grads)
            ]
        ).detach()
    return result


def terms(trainer, batch):
    logits, value, _ = trainer.actor(batch["obs"], batch["mask"])
    logprobs = logits.log_softmax(-1)
    selected = logprobs.gather(1, batch["action"][:, None]).squeeze(-1)
    ratio = (selected - batch["old_logp"]).exp()
    advantage = batch["advantage"]
    policy = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - trainer.config.clip, 1 + trainer.config.clip) * advantage,
    )
    koi = batch["mask"][:, 48:].any(-1)
    return {
        "policy": policy.mean(),
        "entropy": trainer.config.entropy * (logprobs.exp() * logprobs).sum(-1).mean(),
        "value": trainer.config.value_weight
        * (value - batch["return"]).square().mean(),
        "card_policy": (policy * (~koi)).mean(),
        "koi_policy": (policy * koi).mean(),
    }


def summarize(gradients):
    gradients = {k: np.asarray(v, dtype=np.float64) for k, v in gradients.items()}
    norms = {k: float(np.linalg.norm(v)) for k, v in gradients.items()}
    comparisons = {}
    for left, right in (
        ("policy", "value"),
        ("policy", "entropy"),
        ("card_policy", "koi_policy"),
        ("koi_policy", "value"),
    ):
        dot = float(gradients[left] @ gradients[right])
        product = norms[left] * norms[right]
        comparisons[f"{left}_vs_{right}"] = {
            "dot": dot,
            "cosine": dot / product if product else None,
        }
    return {"norms": norms, "comparisons": comparisons}


def verify_production_gradient(trainer, batch, parameters):
    """Compare to actual production backward/clipping, suppressing Adam.step."""
    gradients = vectors(terms(trainer, batch), parameters)
    total = gradients["policy"] + gradients["entropy"] + gradients["value"]
    norm = torch.linalg.vector_norm(total)
    expected = total * torch.clamp(0.5 / (norm + 1e-6), max=1)
    step = trainer.optimizer.step
    trainer.optimizer.step = lambda: None
    try:
        metrics = trainer.minibatch_step(batch)
    finally:
        trainer.optimizer.step = step
    actual = torch.cat(
        [
            (torch.zeros_like(p) if p.grad is None else p.grad).flatten()
            for p in parameters
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-7)
    torch.testing.assert_close(metrics[5], norm, rtol=2e-4, atol=2e-7)
    trainer.optimizer.zero_grad(set_to_none=True)
    return float((actual - expected).abs().max())


def main():
    assert provenance()["source_sha256"] == SOURCE
    for _, path, digest, _ in PARENTS:
        assert sha(path) == digest
    BASE.mkdir(exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {
        "status": "running",
        "source_sha256": SOURCE,
        "script_sha256": sha(__file__),
        "protocol": {
            "parents": PARENTS,
            "rollouts_per_parent": 4,
            "envs": 1024,
            "horizon": 128,
            "chunk": 8192,
            "updates": 0,
            "endgame": True,
            "gradient_terms": TERMS,
            "interpretation": "Raw Euclidean gradients before Adam or clipping; "
            "loss coefficients included. Phase gradients use full-batch denominator. "
            "Self-play sampled trajectories; no win-rate inference or teacher.",
        },
        "cases": {},
    }
    (BASE / "protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    arrays = {}
    for label, path, digest, seed in PARENTS:
        config = Config(
            profile="paper8",
            seed=seed,
            envs=1024,
            horizon=128,
            endgame=True,
            cpu_threads=2,
            cuda_graph=False,
            cuda_rollout=False,
        )
        trainer = Trainer(config)
        trainer.initialize_weights(path)
        initial = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        named = list(trainer.model.named_parameters())
        parameters = [p for _, p in named]
        body = torch.cat(
            [
                torch.full((p.numel(),), name.startswith("body."), device=config.device)
                for name, p in named
            ]
        ).bool()
        case = {"config": asdict(config), "parent_sha256": digest, "rollouts": []}
        for repeat in range(4):
            batch = trainer.collect()
            raw = batch["advantage"]
            adv_mean, adv_std = float(raw.mean()), float(raw.std(unbiased=False))
            batch["advantage"] = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8)
            count = len(raw)
            sums = {
                k: torch.zeros(sum(p.numel() for p in parameters), device=config.device)
                for k in TERMS
            }
            check = None
            for start in range(0, count, 8192):
                data = {k: v[start : start + 8192] for k, v in batch.items()}
                grads = vectors(terms(trainer, data), parameters)
                for key in TERMS:
                    sums[key] += grads[key] * (len(data["obs"]) / count)
                if start == 0:
                    check = verify_production_gradient(trainer, data, parameters)
            torch.testing.assert_close(
                sums["policy"],
                sums["card_policy"] + sums["koi_policy"],
                rtol=2e-4,
                atol=2e-7,
            )
            legal = batch["mask"].sum(-1)
            row = {
                "index": repeat,
                "rows": count,
                "forced_rows": int((legal == 1).sum()),
                "koi_rows": int(batch["mask"][:, 48:].any(-1).sum()),
                "advantage_mean": adv_mean,
                "advantage_std": adv_std,
                "production_gradient_max_error": check,
                "counters": trainer.counters.copy(),
            }
            for scope, selected in (("all", torch.ones_like(body)), ("body", body)):
                scoped = {k: v[selected].cpu().numpy() for k, v in sums.items()}
                row[scope] = summarize(scoped)
                for key, value in scoped.items():
                    arrays[f"{label}_{repeat}_{scope}_{key}"] = value
            case["rollouts"].append(row)
            print(
                json.dumps({"case": label, "rollout": repeat, "body": row["body"]}),
                flush=True,
            )
        assert not trainer.optimizer.state
        assert trainer.counters["updates"] == 0
        for name, value in trainer.model.state_dict().items():
            assert torch.equal(initial[name], value), name
        case["weights_unchanged"] = True
        case["optimizer_empty"] = True
        case["counters"] = trainer.counters.copy()
        case["parameter_layout"] = [(name, list(p.shape)) for name, p in named]
        report["cases"][label] = case
        del trainer, batch, initial
    np.savez_compressed(BASE / "gradients.npz", **arrays)
    report["raw_sha256"] = sha(BASE / "gradients.npz")
    report["status"] = "complete"
    report["limitations"] = (
        "Eight correlated trajectory batches from two fixed actors. "
        "Raw gradients and local first-order geometry do not establish Adam behavior, "
        "causal training failure, or match superiority. No complete trajectory replay."
    )
    (BASE / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
