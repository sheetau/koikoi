"""Training-split target descriptives by phase and remaining match horizon."""

import json
from pathlib import Path

import torch

from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import read, sha


def summarize(public, wins):
    if public.ndim != 2 or public.shape[1] != 562 or wins.shape != (len(public), 2):
        raise ValueError("Expected aligned public roots and paired win labels")
    rounds = (public[:, 534] * 12).round().long()
    phases = public[:, 528:532].argmax(-1)
    if not ((rounds >= 0) & (rounds < 8)).all():
        raise ValueError("Expected paper8 round indices")
    if not torch.isin(wins, torch.tensor([-1.0, 0.0, 1.0])).all():
        raise ValueError("Invalid signed win labels")
    delta = (wins[:, 1].double() - wins[:, 0].double()) / 2
    groups = {
        "early_cards": (rounds < 6) & (phases != 3),
        "late_cards": (rounds >= 6) & (phases != 3),
        "early_koi": (rounds < 6) & (phases == 3),
        "late_koi": (rounds >= 6) & (phases == 3),
    }
    assert torch.stack(list(groups.values())).sum(0).eq(1).all()
    groups = {"all": torch.ones_like(rounds, dtype=torch.bool), **groups}
    result = {}
    for name, mask in groups.items():
        values = delta[mask]
        result[name] = {
            "roots": len(values),
            "different_win_outcomes": int((values != 0).sum()),
            "point_better_realized_outcomes": int((values > 0).sum()),
            "parent_better_realized_outcomes": int((values < 0).sum()),
            "point_minus_parent_win_score": float(values.mean())
            if len(values)
            else None,
        }
    return result


def run(output=Path("runs/opponent_target_stage_diagnostic.json")):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(2)
    base = Path("runs/opponent_target_study")
    completed = read(base / "results.json")
    inputs = {
        str(Path(__file__)): sha(__file__),
        str(base / "results.json"): sha(base / "results.json"),
        "scripts/paired_rollout_targets.py": sha("scripts/paired_rollout_targets.py"),
    }
    reports = {}
    for index in (1, 2):
        path = base / "data" / f"parent{index}" / "opponent_preserving/targets.pt"
        digest = sha(path)
        assert (
            digest
            == completed[str(index)]["collection"]["conditions"]["opponent_preserving"][
                "dataset_sha256"
            ]
        )
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["format"] == "paired_expert_targets_v1"
        keep = ~match_split(data["match_groups"])
        inputs[str(path)] = digest
        reports[str(index)] = summarize(
            data["public_observations"][keep], data["outcomes"][keep]
        )
    report = {
        "inputs": inputs,
        "parents": reports,
        "scope": "Training source-match split only; early=round indices0--5, late=6--7. Descriptive realized outcomes for one changed action followed by frozen continuation. Not public-policy oracle values, independent root samples, causal horizon effects or repeated-policy match strength. No model fit, threshold selection or simulator work.",
    }
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"output": str(output), "sha256": sha(output), "parents": reports})
    )


if __name__ == "__main__":
    run()
