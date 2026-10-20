"""Export checked root-action and phase totals for the passive diagnostic."""

from collections import Counter
from pathlib import Path

from scripts.export_belief_tail import read
from scripts.export_full_source_results import table
from scripts.train_branch_policy_gradient import sha, write


def export():
    base = Path("runs/refused_tail_profile")
    report = Path("docs/REFUSED_TAIL_PROFILE_RESULTS.rst")
    receipt = Path("runs/refused_tail_profile_export.json")
    if report.exists() or receipt.exists():
        raise ValueError("Use fresh outputs")
    complete, protocol = read(base / "complete.json"), read(base / "protocol.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    assert complete["protocol_sha256"] == sha(base / "protocol.json")
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    assert all(
        sha(base / path) == digest for path, digest in complete["artifacts"].items()
    )
    assert [r["source"] for r in complete["records"]] == protocol["source_indices"]
    work, phases, support = Counter(), Counter(), Counter()
    rows = []
    for record in complete["records"]:
        support.update(record["support"]["counts"])
        for name, plan in record["plans"].items():
            assert (
                sum(plan["phase_turn_decisions"].values()) == plan["work"]["decisions"]
            )
            assert (
                sum(a["work"]["decisions"] for a in plan["root_actions"])
                == plan["work"]["decisions"]
            )
            work.update(
                {
                    k: plan["work"][k]
                    for k in (
                        "decisions",
                        "round_endings",
                        "completions",
                        "represented_decisions",
                        "represented_completions",
                    )
                }
            )
            phases.update(plan["phase_turn_decisions"])
            for action in plan["root_actions"]:
                assert ("q" in action) == (action["status"] == "complete")
                assert (
                    sum(action["phase_turn_decisions"].values())
                    == action["work"]["decisions"]
                )
                rows.append(
                    [
                        record["source"],
                        name,
                        action["action"],
                        action["status"],
                        action.get("q", "-"),
                        action["work"]["decisions"],
                    ]
                )
    last_two = sum(v for k, v in phases.items() if int(k) // 10 >= 14)
    text = (
        "Passive profile of remaining tail refusals\n"
        "=========================================\n\n"
        "All five failures reproduce the closed recovery's work, policy-call\n"
        "counts and stopping stage exactly. Every parent-first root action\n"
        "completes; exhaustion occurs in later action comparisons. Source36\n"
        "also reproduces its complete fixed control before optimized refusal.\n\n"
    )
    text += table(
        "All evaluated root actions",
        [
            "Source",
            "Plan",
            "Action",
            "Status",
            "Exact Q if complete",
            "Physical decisions",
        ],
        rows,
    )
    text += table(
        "Successful physical transitions by turn and phase",
        ["Turn (zero based)", "Phase", "Decisions"],
        [
            [int(k) // 10, int(k) % 10, v]
            for k, v in sorted(phases.items(), key=lambda x: int(x[0]))
        ],
    )
    text += (
        f"The final two turns account for {last_two:,}/{work['decisions']:,} "
        f"transitions ({100 * last_two / work['decisions']:.4f}%).\n"
        "Phase0 is discard,1 hand capture,2 draw capture,3 koi. Root action\n"
        "values are exact only for completed actions; no value is assigned\n"
        "to the interrupted candidates. There is no new complete optimized\n"
        "solution or actual-tail assessment, and no new strength result.\n\n"
    )
    text += table(
        "Additional diagnostic work", ["Counter", "Count"], list(work.items())
    )
    text += table(
        "Additional history-support work", ["Counter", "Count"], list(support.items())
    )
    text += (
        "Planning, represented prefixes and history replay are separate counters.\n"
        "No source match, training update or independent sample was added.\n"
        "Timing includes instrumentation and is not a speed comparison. Saved\n"
        "support worlds and partial action caches are bound for subsequent checks.\n\n"
        f"Complete SHA256: {sha(base / 'complete.json')}.\n"
        f"Protocol SHA256: {complete['protocol_sha256']}.\n"
    )
    report.write_bytes(text.encode("utf-8"))
    write(
        receipt,
        {
            "status": "complete",
            "exporter_sha256": sha(__file__),
            "complete_sha256": sha(base / "complete.json"),
            "report_sha256": sha(report),
            "work": dict(work),
            "history": dict(support),
            "phase_turn_decisions": dict(phases),
        },
    )
    print(dict(work))


if __name__ == "__main__":
    export()
