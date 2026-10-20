"""Decompose existing frozen-root gradients by fixed actor parameter blocks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.published_actor import PublishedActor
from koikoi.train import provenance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run():
    base = Path("runs/branch_parameter_variance")
    output = Path("runs/branch_gradient_blocks")
    if output.exists():
        raise ValueError("Use a fresh output; do not repeat this saved-data analysis")
    source = read(base / "summary.json")
    source_protocol = read(base / "protocol.json")
    assert provenance()["source_sha256"] == source_protocol["source"]["source_sha256"]
    diagnostic = read("runs/branch_policy_gradient_diagnostic/summary.json")
    protocol = {
        "scope": "Post-hoc reuse of all eight frozen-root gradient replicates, both parents and coupling modes",
        "source_summary_sha256": sha(base / "summary.json"),
        "source_protocol_sha256": sha(base / "protocol.json"),
        "diagnostic_summary_sha256": sha(
            "runs/branch_policy_gradient_diagnostic/summary.json"
        ),
        "implementation_sha256": sha(__file__),
        "blocks": {
            "output_weights": "Parameter names ending in .out.weight",
            "output_bias": "Parameter names ending in .out.bias",
            "encoder": "All remaining actor parameters",
        },
        "metrics": "Variance trace, squared norm of replicate mean, and mean norm minus variance/8",
        "interpretation": "Raw Euclidean gradients at fixed roots; not Adam update geometry, new samples or a strength estimate",
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for index, parent in enumerate(source["parents"]):
        parent_record = diagnostic["parents"][index]["parent"]
        assert sha(parent_record["path"]) == parent_record["sha256"]
        payload = torch.load(
            parent_record["path"], map_location="cpu", weights_only=False
        )
        actor = PublishedActor(payload["actor"], "cpu")
        assert (
            actor.checkpoint()["current"]["parameters_sha256"]
            == parent["unchanged_parameter_sha256"]
        )
        groups = {name: [] for name in protocol["blocks"]}
        offset = 0
        names = []
        for name, parameter in actor.named_parameters():
            count = parameter.numel()
            group = (
                "output_weights"
                if name.endswith(".out.weight")
                else "output_bias"
                if name.endswith(".out.bias")
                else "encoder"
            )
            groups[group].append(np.arange(offset, offset + count))
            names.append({"name": name, "offset": offset, "count": count})
            offset += count
        groups = {name: np.concatenate(parts) for name, parts in groups.items()}
        assert [len(groups[k]) for k in protocol["blocks"]] == [768, 3, 4018944]
        result = {
            "parent": index + 1,
            "parameter_order": names,
            "modes": {},
        }
        for mode, metadata in parent["modes"].items():
            path = Path(metadata["gradient_file"])
            assert sha(path) == metadata["gradient_sha256"]
            values = torch.load(path, map_location="cpu", weights_only=True).numpy()
            assert values.shape == (8, offset)
            blocks = {}
            for name, indices in groups.items():
                selected = values[:, indices].astype(np.float64)
                variance = float(selected.var(0, ddof=1).sum())
                mean_norm = float(np.square(selected.mean(0)).sum())
                blocks[name] = {
                    "parameters": len(indices),
                    "variance_trace": variance,
                    "mean_gradient_squared_norm": mean_norm,
                    "noise_corrected_mean_squared_norm": mean_norm - variance / 8,
                }
            np.testing.assert_allclose(
                sum(row["variance_trace"] for row in blocks.values()),
                metadata["conditional_parameter_gradient_variance"],
                rtol=1e-12,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                sum(row["mean_gradient_squared_norm"] for row in blocks.values()),
                metadata["mean_parameter_gradient_squared_norm"],
                rtol=1e-12,
                atol=1e-10,
            )
            result["modes"][mode] = {
                "gradient_sha256": metadata["gradient_sha256"],
                "blocks": blocks,
            }
        results.append(result)
    summary = {
        "status": "complete",
        "protocol_sha256": sha(output / "protocol.json"),
        "parents": results,
        "new_simulator_decisions": 0,
        "new_backward_passes": 0,
        "new_optimizer_updates": 0,
        "new_independent_samples": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for item in results:
        print(json.dumps({k: v for k, v in item.items() if k != "parameter_order"}))


if __name__ == "__main__":
    run()
