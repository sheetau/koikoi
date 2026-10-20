"""Post-hoc full-actor gradient variance from the closed diagnostic's labels."""

import json
from pathlib import Path

import numpy as np
import torch

from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.branch_policy_gradient import (
    BranchWorkspace,
    expected_return_loss,
    probabilities,
)
from scripts.train_branch_policy_gradient import setup, sha, write


def run():
    source = Path("runs/branch_policy_gradient_diagnostic")
    output = Path("runs/branch_parameter_variance")
    if output.exists():
        raise ValueError("Use this saved-data analysis only once")
    audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed":
        raise ValueError("Expected a completed diagnostic audit")
    for path, digest in audit["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"Diagnostic binding changed: {path}")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    output.mkdir()
    setup(110060001)
    write(
        output / "protocol.json",
        {
            "analysis": "post-hoc conditional variance of batch-mean full-parameter gradients",
            "diagnostic_audit_sha256": sha(source / "audit.json"),
            "source": provenance(),
            "bindings": {
                path: sha(path)
                for path in (
                    __file__,
                    "scripts/branch_policy_gradient.py",
                    "scripts/train_branch_policy_gradient.py",
                )
            },
            "selection": "All eight saved replicates, both coupling modes, both parents; no training or new simulations.",
            "scope": "Conditions on the same 64 root worlds per parent; excludes between-root/world sampling variance. Descriptive eight-replicate estimates.",
        },
    )
    workspace = BranchWorkspace(64, "cuda")
    result = {"parents": [], "actor_backward_passes": 0, "new_simulation_work": 0}
    for index, item in enumerate(summary["parents"]):
        directory = source / f"parent{index + 1}"
        payload = torch.load(
            item["parent"]["path"], map_location="cpu", weights_only=False
        )
        actor = PublishedActor(payload["actor"], "cuda")
        initial_digest = actor.checkpoint()["current"]["parameters_sha256"]
        data = torch.load(
            directory / "source.pt", map_location="cpu", weights_only=False
        )
        policy = torch.load(
            directory / "root_policy.pt", map_location="cpu", weights_only=False
        )
        features, phases, legal = workspace.root_inputs(data["states"].cuda())
        counts = data["eligible_decisions"].cuda()
        with torch.no_grad():
            p = probabilities(actor, features, phases, legal)
            if not torch.equal(p.cpu(), policy["probabilities"]):
                raise RuntimeError(
                    "Root probabilities differ from the original diagnostic"
                )
        parent = {"parent": index + 1, "modes": {}}
        for mode in ("shared_actions", "independent_actions"):
            vectors = []
            for replicate in range(8):
                branch = torch.load(
                    directory / f"{mode}_{replicate}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                actor.zero_grad(set_to_none=True)
                p = probabilities(actor, features, phases, legal)
                expected_return_loss(p, branch["q"].cuda(), counts).backward()
                vector = torch.cat(
                    [
                        (
                            parameter.grad
                            if parameter.grad is not None
                            else torch.zeros_like(parameter)
                        )
                        .detach()
                        .reshape(-1)
                        .cpu()
                        for parameter in actor.parameters()
                    ]
                )
                if not torch.isfinite(vector).all():
                    raise RuntimeError("Nonfinite actor gradient")
                vectors.append(vector)
                result["actor_backward_passes"] += 1
            gradients = torch.stack(vectors)
            target = output / f"parent{index + 1}_{mode}.pt"
            torch.save(gradients, target)
            # NumPy and Torch independently calculate the scalar summary.
            values = gradients.numpy().astype(np.float64)
            variance = float(values.var(axis=0, ddof=1).sum())
            mean_norm = float(np.square(values.mean(axis=0)).sum())
            torch_variance = float(gradients.double().var(0, unbiased=True).sum())
            if not np.isclose(variance, torch_variance, rtol=1e-12, atol=1e-10):
                raise RuntimeError("Variance calculations disagree")
            parent["modes"][mode] = {
                "conditional_parameter_gradient_variance": variance,
                "mean_parameter_gradient_squared_norm": mean_norm,
                "parameters": gradients.shape[1],
                "replicates": len(gradients),
                "gradient_file": str(target),
                "gradient_sha256": sha(target),
            }
            del values, gradients, vectors
        parent["shared_over_independent_variance"] = (
            parent["modes"]["shared_actions"]["conditional_parameter_gradient_variance"]
            / parent["modes"]["independent_actions"][
                "conditional_parameter_gradient_variance"
            ]
        )
        if actor.checkpoint()["current"]["parameters_sha256"] != initial_digest:
            raise RuntimeError("Analysis mutated the actor")
        parent["unchanged_parameter_sha256"] = initial_digest
        result["parents"].append(parent)
        print(json.dumps(parent), flush=True)
        del actor
    result["status"] = "complete"
    result["protocol_sha256"] = sha(output / "protocol.json")
    write(output / "summary.json", result)


if __name__ == "__main__":
    run()
