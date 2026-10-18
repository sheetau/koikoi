"""Report only a completed,independently audited chance-renewal comparison."""

import argparse
import json
from pathlib import Path

from scripts.renewed_continuation_analysis import decisions, load_points
from scripts.run_arbitration_study import read, sha, write


def require_complete(root):
    if any((root / name).exists() for name in ("summary.md", "summary_receipt.json")):
        raise ValueError("Refusing to overwrite an existing summary")
    process = root / "process.json"
    if not process.exists() or read(process)["status"] != "complete":
        raise ValueError("Complete producer output required")
    path = root / "independent_audit.json"
    if not path.exists() or read(path)["status"] != "passed":
        raise ValueError("Passed independent audit required")
    audit = read(path)
    for name, digest in audit["inputs"].items():
        if sha(name) != digest:
            raise ValueError(f"Changed audited input: {name}")
    plan, produced = read(root / "protocol.json"), read(root / "analysis.json")
    assert plan["kind"] == "renewed_continuation_study_v1"
    assert (
        plan["updates"],
        plan["pairs_per_opponent"],
        plan["bootstrap_seed"],
        plan["bootstrap_replicates"],
    ) == (520, 1024, 95310001, 10000)
    assert produced["status"] == "complete"
    assert produced["contrasts"] == audit["contrasts"]
    assert produced["decisions"] == audit["decisions"] == decisions(audit["contrasts"])
    assert audit["new_neural_inferences"] == audit["new_independent_matches"] == 0
    assert audit["distinct_training_chance_keys"] == 133120
    assert audit["distinct_evaluation_deals"] == 2048
    assert audit["evaluation_replay_work"]["completed_matches"] == 40960
    assert audit["root_replay_work"] == {
        "decisions": 212212,
        "round_endings": 14101,
        "completed_matches": 192,
    }
    assert len(audit["training_counters"]) == 4
    for counter in audit["training_counters"]:
        assert (
            counter["updates"],
            counter["warmup_updates"],
            counter["policy_updates"],
        ) == (520, 20, 500)
        assert (
            counter["conditional_tail_completions"]
            == counter["fresh_conditional_deals"]
            == 66560
        )
        assert counter["matches"] == counter["rounds"] == 0
        assert (
            counter["decisions"]
            == counter["learner_decisions"] + counter["opponent_decisions"]
        )
    # Validate every complete grid identity rather than trusting a total alone.
    load_points(root)
    return audit


def render(audit, training, calls, protocol_sha, audit_sha):
    verdicts = decisions(audit["contrasts"])
    assert verdicts == audit["decisions"]
    lines = [
        "# Chance-renewal comparison",
        "",
        "All four renewed-training arms and all 40,960 full evaluation matches completed. The five policies and two parents share 2,048 distinct evaluation deals.",
        "",
        "| Prespecified claim | Result |",
        "| --- | --- |",
    ]
    for key, label in (
        (
            "replicated_strength_gain",
            "Primary: renewed-public beats original and fixed-public in the balanced opponent mixture at both parents",
        ),
        (
            "replicated_public_memory_gain",
            "Secondary: renewed-public beats renewed-masked at both parents",
        ),
        (
            "replicated_positive_interaction",
            "Secondary: chance renewal increases the public-memory benefit at both parents",
        ),
    ):
        lines.append(f"| {label} | {'Pass' if verdicts[key] else 'Fail'} |")
    lines += [
        "",
        "Each decision requires strictly positive lower 95% paired win bounds for every listed condition. Secondary claims do not replace the primary criterion. These comparisons establish neither a new general learning principle nor performance against arbitrary opponents.",
        "",
        "## All paired comparisons",
        "",
        "Win changes are percentage points. Point changes use the match point-difference scale. The mixture gives equal weight to WP and Point;opponent-specific cells are secondary. All ten pairwise comparisons and the interaction are retained.",
        "",
        "| Parent | Comparison | Scope | Win change (pp), 95% CI | Point change, 95% CI |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in audit["contrasts"]:
        values = []
        for metric, scale in (("win", 100), ("point", 1)):
            value = row[metric]
            lower, upper = value["interval95"]
            values.append(
                f"{value['change'] * scale:+.3f} [{lower * scale:+.3f}, {upper * scale:+.3f}]"
            )
        lines.append(
            f"| {row['parent']} | {row['comparison']} | {row['scope']} | {' | '.join(values)} |"
        )
    evaluation, roots = audit["evaluation_replay_work"], audit["root_replay_work"]
    lines += [
        "",
        "## Accounting",
        "",
        f"New training: {training['decisions']:,} decisions ({training['learner_decisions']:,} learner / {training['opponent_decisions']:,} opponent), {training['conditional_tail_completions']:,} conditional completions and {training['conditional_round_endings']:,} conditional round endings. These are zero full training matches. The 266,240 fresh conditional-deal constructions use 133,120 distinct chance keys shared within each parent's public/masked pair.",
        "",
        "The fixed-bank controls retain their earlier endpoints. Their preceding training is not counted again as new training. Equal completions do not imply equal decisions or optimizer steps.",
        "",
        f"Actual evaluation: {evaluation['decisions']:,} decisions, {evaluation['round_endings']:,} round endings and {evaluation['completed_matches']:,} full matches. Public identification adds {sum(calls.values()):,} strict predictions ({calls['wp']:,} WP / {calls['point']:,} Point).",
        "",
        f"Independent replay repeats the recorded evaluation trajectories and checks {audit['strict_input_checks']:,} historical inputs/masks. It adds zero neural predictions or independent deals. Separate root reconstruction repeats {roots['decisions']:,} source decisions, {roots['round_endings']:,} source round endings and {roots['completed_matches']:,} early full completions.",
        "",
        "## Evidence",
        "",
        f"Protocol SHA256: `{protocol_sha}`.",
        "",
        f"Independent audit SHA256: `{audit_sha}`.",
        "",
        "Saved root-index schedules match the fixed controls. Initial states,paired warmup,ancestry,public metadata,fresh chance accounting and optimizer counts pass their independent checks. The canonical human-play model and reserved 101M deals remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def summarize(root):
    audit = require_complete(root)
    keys = (
        "decisions",
        "learner_decisions",
        "opponent_decisions",
        "conditional_round_endings",
        "conditional_tail_completions",
        "fresh_conditional_deals",
        "matches",
        "rounds",
    )
    training = {key: sum(c[key] for c in audit["training_counters"]) for key in keys}
    calls, deals, matches = {"wp": 0, "point": 0}, set(), 0
    for path in sorted(root.glob("eval_parent*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                matches += 1
                deals.add(row["seed"])
                if row["memory"] is not None:
                    for name in calls:
                        count = row["memory"]["prediction_calls"][name]
                        assert type(count) is int and count >= 0
                        calls[name] += count
    assert matches == 40960 and len(deals) == 2048
    protocol_sha, audit_sha = (
        sha(root / "protocol.json"),
        sha(root / "independent_audit.json"),
    )
    text = render(audit, training, calls, protocol_sha, audit_sha)
    for path, digest in audit["inputs"].items():
        assert sha(path) == digest, path
    with (root / "summary.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    write(
        root / "summary_receipt.json",
        {
            "status": "complete",
            "protocol_sha256": protocol_sha,
            "independent_audit_sha256": audit_sha,
            "summary_sha256": sha(root / "summary.md"),
            "summarizer_sha256": sha(__file__),
            "decisions": audit["decisions"],
            "training_work": training,
            "evaluation_work": audit["evaluation_replay_work"],
            "root_replay_work": audit["root_replay_work"],
            "prediction_calls": calls,
            "distinct_evaluation_deals": len(deals),
            "distinct_training_chance_keys": 133120,
        },
    )
    print(
        json.dumps(
            {
                "decisions": audit["decisions"],
                "summary_sha256": sha(root / "summary.md"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output)
