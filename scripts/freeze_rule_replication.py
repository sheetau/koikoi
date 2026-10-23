"""Freeze one independent precision replication; the failed cohort stays closed."""

import argparse
from copy import deepcopy
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write
from scripts.study_rule_variants import checked


def freeze(base, original=Path("runs/rule_variants_v1")):
    if base.exists():
        raise FileExistsError(base)
    p = deepcopy(checked(original))
    result = read(original / "confirmation_summary.json")
    selection = read(original / "development_summary.json")
    if result["criterion_met"] or result["selected"] != "winner_takes_yaku":
        raise ValueError(
            "This replication follows the recorded failed selected-rule comparison"
        )
    if not all(c["mean_gain"] > 0 for c in result["comparisons"].values()):
        raise ValueError("The rationale requires both positive point estimates")
    base.mkdir(parents=True)
    (base / "development_summary.json").write_bytes(
        (original / "development_summary.json").read_bytes()
    )
    p.update(
        scope="one independent precision replication of the same transferred winner_takes_yaku comparison; original failed confirmation remains unchanged",
        confirmation={"seed": 240300000, "pairs": 8192},
        rationale="Original paired means are positive for both parents but joint 2048-pair criterion failed; quadruple the new fixed sample for roughly half the sampling uncertainty",
        selection_imported_from=str(original / "development_summary.json"),
        preceding_failed_result=str(original / "confirmation_summary.json"),
        extension="none; exactly one new fixed replication; do not pool or relabel the preceding failed study",
    )
    for path in (
        Path(__file__),
        original / "development_summary.json",
        original / "confirmation_summary.json",
        base / "development_summary.json",
    ):
        p["inputs"][str(path)] = sha(path)
    write(base / "protocol.json", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.base)
