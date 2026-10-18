"""Summarize only a completed, independently audited multi-round public-memory comparison."""

import argparse
import json
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write

PROTOCOL_SHA = "95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1"


def summarize(root):
    if any((root / name).exists() for name in ("summary.md", "summary_receipt.json")):
        raise ValueError("Refusing to overwrite the completed summary")
    assert sha(root / "protocol.json") == PROTOCOL_SHA
    assert read(root / "process.json")["status"] == "complete"
    audit = read(root / "independent_audit.json")
    assert audit["status"] == "passed"
    expected = {
        (parent, comparison, scope)
        for parent in (1, 2)
        for comparison in (
            "public-minus-masked",
            "public-minus-parent",
            "masked-minus-parent",
        )
        for scope in ("mixture", "wp", "point")
    }
    assert len(audit["contrasts"]) == 18
    assert {
        (r["parent"], r["comparison"], r["scope"]) for r in audit["contrasts"]
    } == expected
    assert audit["new_neural_inferences"] == audit["new_independent_matches"] == 0
    for path, digest in audit["inputs"].items():
        assert sha(path) == digest, path
    produced = read(root / "analysis.json")
    assert produced["contrasts"] == audit["contrasts"]
    assert produced["replicated_gain"] == audit["replicated_gain"]
    primary = [
        r
        for r in audit["contrasts"]
        if r["scope"] == "mixture" and r["comparison"] != "masked-minus-parent"
    ]
    assert len(primary) == 4
    assert audit["replicated_gain"] == all(
        r["win"]["interval95"][0] > 0 for r in primary
    )
    lines = [
        "# Multi-round public-memory comparison",
        "",
        (
            "All four training arms and all 24,576 strict full matches completed. "
            "The same 2,048 fresh deals are shared across policies and parents."
        ),
        "",
        (
            f"The predeclared balanced-mixture replication criterion {'passes' if audit['replicated_gain'] else 'fails'}. "
            "Individual opponent cells are reported below; the primary criterion concerns the equal WP/Point mixture."
        ),
        "",
        "| Parent | Contrast | Opponent scope | Win change (pp), 95% CI | Point change, 95% CI |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in audit["contrasts"]:
        cells = []
        for name, scale in (("win", 100), ("point", 1)):
            data = row[name]
            lo, hi = data["interval95"]
            cells.append(
                f"{scale * data['change']:+.3f} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
            )
        lines.append(
            f"| {row['parent']} | {row['comparison']} | {row['scope']} | "
            + " | ".join(cells)
            + " |"
        )
    training = {
        k: sum(c[k] for c in audit["training_counters"])
        for k in (
            "decisions",
            "learner_decisions",
            "opponent_decisions",
            "conditional_round_endings",
            "conditional_tail_completions",
            "rounds",
            "matches",
        )
    }
    assert len(audit["training_counters"]) == 4
    assert all(
        c["updates"] == 520 and c["warmup_updates"] == 20 and c["policy_updates"] == 500
        for c in audit["training_counters"]
    )
    root_replay = audit["root_replay_work"]
    assert root_replay == {
        "decisions": 212212,
        "round_endings": 14101,
        "completed_matches": 192,
    }
    evaluation = audit["evaluation_replay_work"]
    assert training["matches"] == training["rounds"] == 0
    assert training["conditional_tail_completions"] == 266240
    assert evaluation["completed_matches"] == 24576
    calls = {"wp": 0, "point": 0}
    distinct_deals = set()
    for path in root.glob("eval_parent*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                distinct_deals.add(row["seed"])
                if row["memory"] is not None:
                    for name in calls:
                        calls[name] += row["memory"]["prediction_calls"][name]
    assert len(distinct_deals) == 2048
    lines += [
        "",
        "## Accounting",
        "",
        (
            f"New training: {training['decisions']:,} decisions "
            f"({training['learner_decisions']:,} learner / {training['opponent_decisions']:,} opponent), "
            f"{training['conditional_tail_completions']:,} conditional multi-round completions, "
            f"{training['conditional_round_endings']:,} conditional round endings, zero full training matches."
        ),
        "",
        (
            f"Actual evaluation: {evaluation['decisions']:,} decisions, {evaluation['round_endings']:,} round endings, "
            f"{evaluation['completed_matches']:,} complete matches. Public model checks add {sum(calls.values()):,} "
            f"strict predictions ({calls['wp']:,} WP / {calls['point']:,} Point)."
        ),
        "",
        (
            f"Independent replay repeats those recorded matches and checks {audit['strict_input_checks']:,} historical "
            "inputs/masks against the original builder. It adds no neural inference or independent deals."
        ),
        "",
        (
            f"Independent root verification separately repeats {root_replay['decisions']:,} source decisions, "
            f"{root_replay['round_endings']:,} source round endings and {root_replay['completed_matches']:,} early full completions. "
            "The four arms reuse all 855/876 natural public-stopping roots; source generation and identification are accounted "
            "in the preceding studies. CPU paper compilation/rendering overlapped training; timings are not isolated latency benchmarks."
        ),
        "",
        "## Evidence",
        "",
        f"Protocol SHA256: `{PROTOCOL_SHA}`.",
        "",
        f"Independent audit SHA256: `{sha(root / 'independent_audit.json')}`.",
        "",
        (
            "All initial/warmup comparisons, frozen prefix/opponent checks, public-code joins and Adam counts pass. "
            "No model is automatically promoted; the canonical game and reserved 101M deals remain unchanged."
        ),
        "",
    ]
    (root / "summary.md").write_bytes("\n".join(lines).encode("utf-8"))
    write(
        root / "summary_receipt.json",
        {
            "status": "complete",
            "protocol_sha256": PROTOCOL_SHA,
            "independent_audit_sha256": sha(root / "independent_audit.json"),
            "summary_sha256": sha(root / "summary.md"),
            "replicated_gain": audit["replicated_gain"],
            "training_work": training,
            "evaluation_work": evaluation,
            "root_replay_work": root_replay,
            "prediction_calls": calls,
            "distinct_deals": len(distinct_deals),
            "summarizer_sha256": sha(__file__),
        },
    )
    print(
        json.dumps(
            {
                "replicated_gain": audit["replicated_gain"],
                "training_work": training,
                "evaluation_work": evaluation,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output)
