"""Export manuscript tables from immutable audits, without reading live scores."""

import hashlib
import json
from pathlib import Path

OUTPUT = Path("docs/paper/generated")
PINNED = {
    "development": (
        "runs/cold_ppo_screen/independent_audit.json",
        "6b8548605b50226738227ce83dba1a2e946efd282461b0fa8d48132a42cfe9ba",
    ),
    "earlier_screen": (
        "runs/pretrained_ppo_screen/independent_audit.json",
        "643432440c451836559ab398d45a095f21043c4f04b60e26ec470dfa09d8c982",
    ),
    "final_protocol": (
        "runs/final_published_test/protocol.json",
        "4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4",
    ),
    "feature_compatibility": (
        "runs/published_batch_independent_audit.json",
        "e2a9567be3f05bf6a72ff6cbf3c56383463e9cd07d535f8bd90cddcb7a9599d4",
    ),
    "search_replication": (
        "runs/ppo8_v3/all_search192_1000_independent_audit.json",
        "51f2fdf02504c23fa55e92a40e796537246718dcbdcf5aef7f1e16222214ed5f",
    ),
    "search_confirmation": (
        "runs/confirm1024_search192_independent_audit.json",
        "e4a28dc9792271e2e8dab760f72725f4f16821288d2e37e85cf09c3e51e96b9e",
    ),
    "selected_world_diagnostic": (
        "runs/search_resampling_diagnostic_audit.json",
        "031644fcb0f6297bda3bde53301110c0050aa8c7fa244357a3932d67d491b6ad",
    ),
    "supported_world_diagnostic": (
        "runs/supported_values_independent_audit.json",
        "e1cca166e4ee9348624211be925f275db2c8ec4bdc483bf3bdf434649640775e",
    ),
}
SCALE = Path("runs/ppo12_scale_seed30001_captured/audit_100m_independent.json")
SCALE_MODEL = "613ea0d992a54a0a18ab1848f106abe074c5d74640d76c3a86a131448aa29cce"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def interval(values, factor=1, digits=3):
    return f"[{values[0] * factor:.{digits}f}, {values[1] * factor:.{digits}f}]"


def export(output=OUTPUT):
    inputs, evidence = {}, {}
    for name, (path, digest) in PINNED.items():
        if sha(path) != digest:
            raise ValueError(f"Changed audited input: {path}")
        evidence[name] = read(path)
        inputs[path] = digest
    dev = evidence["development"]
    assert dev["status"] == evidence["earlier_screen"]["status"] == "passed"
    assert dev["decision"]["selected_arm"] == "warm"
    assert dev["decision"]["canonical_case"] == "seed6710001_warm"
    for folder, audit in (
        (Path("runs/cold_ppo_screen"), dev),
        (Path("runs/pretrained_ppo_screen"), evidence["earlier_screen"]),
    ):
        assert sha(folder / "summary.json") == audit["summary_sha256"]
        for case, endpoint in audit["endpoints"].items():
            path = folder / case / "evaluation.json"
            assert sha(path) == endpoint["report_sha256"]
            inputs[str(path)] = sha(path)

    scale = read(SCALE)
    assert (
        sha(SCALE) == "d9447712468ac9cec4eedea842898445aacf2e74aa8adaa69baa36f84cb51c51"
    )
    assert scale["checkpoint_sha256"] == SCALE_MODEL
    assert scale["episode_round_decision_accounting_and_finiteness"]
    assert scale["means_intervals_and_paired_difference_recomputed"]
    assert scale["counters"] == {
        "updates": 69267,
        "decisions": 18157928448,
        "rounds": 1200025665,
        "matches": 100001204,
    }
    inputs[str(SCALE)] = sha(SCALE)
    evidence["scale"] = scale

    rows, paired, budgets = [], [], []
    for seed in (6710001, 6720001):
        for arm in ("warm", "cold"):
            case = f"seed{seed}_{arm}"
            r = dev["endpoints"][case]
            rows.append(
                f"{seed} & {arm} & {100 * r['score_mean']:.3f} & "
                f"{interval(r['score_ci95'], 100)} & {r['point_mean']:.4f} \\\\"
            )
            c = dev["checkpoints"][case + ":520"]["counters"]
            budgets.append(
                f"{seed} & {arm} & {c['matches']:,} & {c['rounds']:,} & "
                f"{c['decisions']:,} \\\\"
            )
        r = dev["paired"][str(seed)]
        paired.append(
            f"{seed} & {100 * r['score_mean']:.3f} & "
            f"{interval(r['score_ci95'], 100)} & {r['point_mean']:.4f} \\\\"
        )

    final_path = Path("runs/final_published_test/independent_audit.json")
    final_rows = []
    if final_path.exists():
        final = read(final_path)
        protocol = evidence["final_protocol"]
        assert final["status"] == "passed"
        assert final["auditor_sha256"] == sha("scripts/audit_final_published_test.py")
        assert final["development_auditor_sha256"] == sha(
            "scripts/audit_cold_ppo_screen.py"
        )
        assert final["statistics_helper_sha256"] == sha(
            "scripts/audit_round_continuation_screen.py"
        )
        assert final["protocol_sha256"] == PINNED["final_protocol"][1]
        assert final["summary_sha256"] == sha(final_path.with_name("summary.json"))
        assert set(final["endpoints"]) == set(protocol["jobs"])
        for name, job in protocol["jobs"].items():
            r = final["endpoints"][name]
            path = final_path.parent / name / "evaluation.json"
            assert sha(path) == r["report_sha256"]
            inputs[str(path)] = sha(path)
            final_rows.append(
                f"{job['case'].split('_')[0][4:]} & {job['variant']} & "
                f"{100 * r['score_mean']:.3f} & {interval(r['score_ci95'], 100)} & "
                f"{r['point_mean']:.4f} \\\\"
            )
        inputs[str(final_path)] = sha(final_path)
        evidence["final"] = final
        status = "audited_final_results_available"
    else:
        # No live log, provisional summary, or incomplete endpoint is read here.
        evidence["final"] = None
        status = "draft_final_results_pending"
    bundle = {
        "status": status,
        "exporter_sha256": sha(__file__),
        "inputs": inputs,
        "evidence": evidence,
        "scope": "transcription of audited artifacts, not a new statistical or trajectory audit",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content, columns, header in (
        (
            "development_table.tex",
            rows,
            "rlrrr",
            r"Training seed & Setting & Score (\%) & 95\% interval (\%) & Point diff.",
        ),
        (
            "paired_table.tex",
            paired,
            "rrrr",
            r"Training seed & Cold minus warm (pp) & 95\% interval (pp) & Point diff. change",
        ),
        (
            "budget_table.tex",
            budgets,
            "rlrrr",
            r"Training seed & Setting & Matches & Round endings & Decisions",
        ),
        (
            "final_table.tex",
            final_rows,
            "rlrrr",
            r"Training seed & Opponent & Score (\%) & 95\% interval (\%) & Point diff.",
        ),
    ):
        # Include the complete alignment: LaTeX's input hooks between the last
        # row and an external bottomrule can otherwise start a spurious cell.
        table = (
            [
                rf"\begin{{tabular}}{{{columns}}}",
                r"\toprule",
                header + r" \\",
                r"\midrule",
                *content,
                r"\bottomrule",
                r"\end{tabular}",
            ]
            if content
            else []
        )
        (output / name).write_text(
            "\n".join(table) + "\n", encoding="utf-8", newline="\n"
        )
    (output / "evidence.json").write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print({"status": status, "bound_artifacts": len(inputs), "output": str(output)})


if __name__ == "__main__":
    export()
