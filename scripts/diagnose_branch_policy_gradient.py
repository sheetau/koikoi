"""Fixed small diagnostic of direct action-outcome gradient signals."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from koikoi.env import BatchEnv
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.branch_policy_gradient import (
    BranchWorkspace,
    collect_roots,
    expected_return_loss,
    probabilities,
)
from scripts.run_arbitration_study import PARENTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value


def statistics(probs, counts, samples):
    # samples[replicate, root, action], logit gradients of individual root terms.
    p = probs.double()
    q = samples.double()
    gradient = p * (q - (p * q).sum(-1, keepdim=True))
    weighted = gradient * counts[None, :, None]
    variable = (
        q.masked_fill(~(p > 0), -2).max(-1).values
        != q.masked_fill(~(p > 0), 2).min(-1).values
    )
    return {
        "conditional_logit_variance_per_root": weighted.var(0, unbiased=True).sum(-1),
        "unweighted_conditional_logit_variance_per_root": gradient.var(
            0, unbiased=True
        ).sum(-1),
        "roots_with_any_outcome_difference": int(variable.any(0).sum()),
        "replicate_root_outcome_difference_fraction": float(variable.double().mean()),
        "mean_gradient_squared_norm": float(weighted.mean(0).square().sum(-1).mean()),
    }


def run(base):
    base = Path(base)
    if base.exists():
        raise ValueError("Use a fresh diagnostic directory; completed runs are closed")
    base.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(110000001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    files = (
        "scripts/branch_policy_gradient.py",
        "scripts/diagnose_branch_policy_gradient.py",
        "scripts/run_arbitration_study.py",
        "docs/BRANCH_POLICY_GRADIENT.rst",
        "tests/test_branch_policy_gradient.py",
    )
    plan = {
        "format": "branch_policy_gradient_diagnostic_v1",
        "source": provenance(),
        "bindings": {name: sha(name) for name in files},
        "parents": [{"path": name, "sha256": digest} for name, digest in PARENTS],
        "device": "cuda",
        "source_matches_per_parent": 64,
        "replicates_per_mode": 8,
        "temperature": 0.05,
        "source_seeds": [110110001, 110210001],
        "action_seed_offset": 1000000,
        "reservoir_seed_offset": 2000000,
        "branch_seed_rule": "110410001 + parent_index*1000000 + mode_index*10000 + replicate",
        "modes": ["shared_actions", "independent_actions"],
        "scope": "conditional logit-gradient variance and cost; no strength inference",
        "training_updates": 0,
    }
    for parent in plan["parents"]:
        if sha(parent["path"]) != parent["sha256"]:
            raise ValueError("Parent identity mismatch")
    write(base / "protocol.json", plan)
    summary = {"protocol_sha256": sha(base / "protocol.json"), "parents": []}
    workspace = BranchWorkspace(64 * 8, "cuda")
    start = time.perf_counter()
    for index, parent in enumerate(plan["parents"]):
        seed = plan["source_seeds"][index]
        directory = base / f"parent{index + 1}"
        directory.mkdir()
        payload = torch.load(parent["path"], map_location="cpu", weights_only=False)
        actor = PublishedActor(payload["actor"], "cuda")
        initial_digest = actor.checkpoint()["current"]["parameters_sha256"]
        env = BatchEnv(64, "cuda", seed=seed, debug=True)

        def rng(value):
            return torch.Generator(device="cuda").manual_seed(value)

        data = collect_roots(env, actor, rng(seed + 1000000), rng(seed + 2000000))
        torch.save(cpu(data), directory / "source.pt")
        features, phases, legal = workspace.root_inputs(data["states"])
        with torch.no_grad():
            probs = probabilities(actor, features, phases, legal)
        torch.save(
            cpu({"probabilities": probs, "phases": phases, "legal": legal}),
            directory / "root_policy.pt",
        )
        result = {
            "parent": parent,
            "source_work": data["work"],
            "roots": len(probs),
            "zero_weight_roots": int((data["eligible_decisions"] == 0).sum()),
            "eligible_decisions": int(data["eligible_decisions"].sum()),
            "phase_counts": torch.bincount(phases, minlength=4).cpu().tolist(),
            "mean_legal_actions": float(legal.float().sum(-1).mean()),
            "modes": {},
        }
        for mode_index, mode in enumerate(plan["modes"]):
            outcomes, total = [], dict.fromkeys(data["work"], 0)
            mode_start = time.perf_counter()
            for replicate in range(plan["replicates_per_mode"]):
                branch = workspace.run(
                    data["states"],
                    actor,
                    rng(110410001 + index * 1000000 + mode_index * 10000 + replicate),
                    shared_actions=mode == "shared_actions",
                )
                torch.save(cpu(branch), directory / f"{mode}_{replicate}.pt")
                outcomes.append(branch["q"])
                for key in total:
                    total[key] += branch["work"][key]
                print(
                    json.dumps(
                        {
                            "parent": index + 1,
                            "mode": mode,
                            "replicate": replicate + 1,
                            "branch_work": total,
                        }
                    ),
                    flush=True,
                )
            stacked = torch.stack(outcomes)
            stats = statistics(probs, data["eligible_decisions"], stacked)
            torch.save(cpu(stats), directory / f"{mode}_statistics.pt")
            actor.zero_grad(set_to_none=True)
            differentiable = probabilities(actor, features, phases, legal)
            expected_return_loss(
                differentiable, stacked.mean(0), data["eligible_decisions"]
            ).backward()
            gradients = [p.grad for p in actor.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("Nonfinite or missing actor gradients")
            result["modes"][mode] = {
                "branch_work": total,
                "elapsed_seconds": time.perf_counter() - mode_start,
                "mean_conditional_logit_variance": float(
                    stats["conditional_logit_variance_per_root"].mean()
                ),
                "mean_unweighted_conditional_logit_variance": float(
                    stats["unweighted_conditional_logit_variance_per_root"].mean()
                ),
                "roots_with_any_outcome_difference": stats[
                    "roots_with_any_outcome_difference"
                ],
                "replicate_root_outcome_difference_fraction": stats[
                    "replicate_root_outcome_difference_fraction"
                ],
                "mean_gradient_squared_norm": stats["mean_gradient_squared_norm"],
                "parameter_gradient_norm": sum(
                    float(g.double().square().sum()) for g in gradients
                )
                ** 0.5,
            }
        shared = result["modes"]["shared_actions"]["mean_conditional_logit_variance"]
        independent = result["modes"]["independent_actions"][
            "mean_conditional_logit_variance"
        ]
        result["conditional_variance_ratio_shared_over_independent"] = (
            shared / independent if independent else None
        )
        if actor.checkpoint()["current"]["parameters_sha256"] != initial_digest:
            raise RuntimeError("Diagnostic mutated the parent actor")
        result["unchanged_parameter_sha256"] = initial_digest
        summary["parents"].append(result)
        write(base / "progress.json", summary)
        del actor
    if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
        raise RuntimeError("Production sources changed during diagnostic")
    if any(sha(name) != digest for name, digest in plan["bindings"].items()):
        raise RuntimeError("Diagnostic sources changed during execution")
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["artifacts"] = {
        str(path.relative_to(base)): sha(path) for path in sorted(base.rglob("*.pt"))
    }
    summary["status"] = "complete"
    write(base / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/branch_policy_gradient_diagnostic")
    run(parser.parse_args().output)
