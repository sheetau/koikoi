"""Independently check saved Point-initialization work and actual optimizer states."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from koikoi.upstream_batch import BatchPublishedAgent
from scripts.audit_pretrained_ppo_screen import tensor_digest, validate_accounting


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, arm, update):
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert update in (protocol["warmup"], protocol["updates"])
    assert arm in protocol["configs"]
    for name, expected in protocol["inputs"].items():
        assert sha(name) == expected, name
    folder = root / arm
    config = protocol["configs"][arm]
    # Read only the closed prefix needed by this immutable snapshot. Later rows
    # may still be written; no hash of a moving full metrics file is certified.
    lines = (folder / "metrics.jsonl").read_bytes().splitlines(keepends=True)[:update]
    assert len(lines) == update and all(line.endswith(b"\n") for line in lines)
    raw = [json.loads(line) for line in lines]
    assert [row["update"] for row in raw] == list(range(1, update + 1))
    rows = [row["metrics"] | row["counters"] for row in raw]
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
    path = folder / f"update_{update:08d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == "phase_temperature_ppo_v1"
    assert json.dumps(payload["config"], sort_keys=True) == json.dumps(
        config, sort_keys=True
    )
    assert initial["config"] == payload["config"]
    assert payload["provenance"]["phase_temperature_source"] == sha(
        "scripts/phase_temperature_ppo.py"
    )
    source = hashlib.sha256()
    for source_path in sorted(Path("koikoi").rglob("*")):
        if source_path.suffix in (".py", ".h", ".cpp", ".cu"):
            source.update(source_path.as_posix().encode())
            source.update(source_path.read_bytes())
    assert payload["provenance"]["source_sha256"] == source.hexdigest()
    reference = BatchPublishedAgent(config["initial_variant"], "cpu").checkpoint()
    for key in ("variant", "upstream_revision", "model_hashes", "parameters_sha256"):
        assert payload["actor"]["initialization"][key] == reference[key]
    assert payload["actor"]["initialization"] == initial["actor"]["initialization"]
    assert (
        payload["actor"]["initialization"]
        == payload["provenance"]["published_initialization"]
    )
    digest = tensor_digest(payload["actor"]["current"]["weights"])
    assert digest == payload["actor"]["current"]["parameters_sha256"]
    initial_digest = tensor_digest(initial["actor"]["current"]["weights"])
    assert initial_digest == reference["parameters_sha256"]
    assert (digest == initial_digest) == (update <= protocol["warmup"])
    assert all(torch.isfinite(t).all() for t in payload["value"].values())
    assert payload["runtime"]["torch"] == torch.__version__
    assert payload["runtime"]["device"] == "cuda:0"
    for key, expected in {
        "cpu_threads": 2,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }.items():
        assert payload["runtime"][key] == expected
    validate_accounting(payload, rows, update)
    groups, state = payload["optimizer"]["param_groups"], payload["optimizer"]["state"]
    assert (
        len(groups) == 2
        and len(groups[0]["params"]) == 90
        and len(groups[1]["params"]) == 6
    )
    assert groups[0]["lr"] == 3e-6 and groups[1]["lr"] == 1e-4
    assert groups[0]["eps"] == groups[1]["eps"] == 1e-5
    ids = groups[0]["params"] + groups[1]["params"]
    assert len(set(ids)) == 96 and set(state) <= set(ids)
    expected_steps = []
    for name, size in (("discard", 30), ("pick", 30), ("koi", 30), ("value", 6)):
        expected_steps.extend([payload["counters"]["optimizer_steps_" + name]] * size)
    for key, steps in zip(ids, expected_steps):
        if not steps:
            assert key not in state
        else:
            assert int(state[key]["step"]) == steps
            assert torch.isfinite(state[key]["exp_avg"]).all()
            assert torch.isfinite(state[key]["exp_avg_sq"]).all()
    result = {
        "sha256": sha(path),
        "parameters_sha256": digest,
        "counters": payload["counters"],
    }
    assert payload["counters"]["decisions"] == update * 16384
    if update == protocol["updates"]:
        assert payload["counters"]["decisions"] == protocol["decisions_per_arm"]
    return {
        "status": "passed",
        "arm": arm,
        "update": update,
        "protocol_sha256": sha(protocol_path),
        "auditor_sha256": sha(__file__),
        "helper_sha256": sha("scripts/audit_pretrained_ppo_screen.py"),
        "metrics_prefix_rows": update,
        "metrics_prefix_sha256": hashlib.sha256(b"".join(lines)).hexdigest(),
        "initial_sha256": sha(folder / "initial.pt"),
        "checkpoint": result,
        "scope": "Saved metrics prefix, simulator episode counters, model identities and finite tensors, per-parameter Adam steps/moments, warmup and decision accounting. No trajectory, neural-inference or training replay and no playing-strength evidence.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing checkpoint audits")
    torch.set_num_threads(2)
    result = audit(args.input, args.arm, args.update)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
