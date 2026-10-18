"""Export audited tail adaptation and public revelation results for the paper."""

from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def main():
    tail_path = Path("runs/final_round_training_study/independent_audit.json")
    identity_path = Path("runs/revealed_opponent_identification/independent_audit.json")
    expected = {
        tail_path: "2f5b65a1fbb5e036c757c106cd9ee9294effef878a0688081ad29d987cd10278",
        identity_path: "c83ea431e670ef91b3fb53d780993e6bcc3ebd92a1df54387ae627f583096fc2",
    }
    inputs = {}
    for path, digest in expected.items():
        assert sha(path) == digest
        report = read(path)
        assert report["status"] == "passed"
        for filename, bound in report["inputs"].items():
            assert sha(filename) == bound, filename
            inputs[filename] = bound
        inputs[str(path)] = digest
    tail, identity = read(tail_path), read(identity_path)
    assert not tail["replicated_gain"]
    groups = identity["groups"]
    assert sum(g["identified_correct"] for g in groups.values()) == 1746
    assert sum(g["ambiguous"] for g in groups.values()) == 302
    assert (
        sum(
            g["identified_wrong"] + g["inconsistent"] + g["true_model_excluded"]
            for g in groups.values()
        )
        == 0
    )
    assert (
        sum(g["correctly_identified_final_round_survivors"] for g in groups.values())
        == 1598
    )
    assert sum(g["final_round_survivors"] for g in groups.values()) == 1856
    folder = Path("docs/paper/generated")
    outputs = {}
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Opponent & Win change (pp) & Point change \\",
        r"\midrule",
    ]
    for r in tail["contrasts"]:
        cells = []
        for metric, scale in (("win", 100), ("point", 1)):
            data = r[metric]
            lo, hi = data["interval95"]
            cells.append(
                f"{scale * data['change']:+.3f} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
            )
        rows.append(
            f"{r['parent']} & {'WP' if r['opponent'] == 'wp' else 'Point'} & "
            + " & ".join(cells)
            + r" \\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    path = folder / "final_tail_table.tex"
    path.write_bytes("\n".join(rows).encode("utf-8"))
    outputs[str(path)] = sha(path)
    rows = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Parent & Opponent & Identified & Unresolved & Identified at final root \\",
        r"\midrule",
    ]
    for parent in (1, 2):
        for opponent in ("wp", "point"):
            g = groups[f"parent{parent}_{opponent}"]
            rows.append(
                f"{parent} & {'WP' if opponent == 'wp' else 'Point'} & {g['identified_correct']}/512 & {g['ambiguous']} & {g['correctly_identified_final_round_survivors']}/{g['final_round_survivors']}"
                + r" \\"
            )
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    path = folder / "public_revelation_table.tex"
    path.write_bytes("\n".join(rows).encode("utf-8"))
    outputs[str(path)] = sha(path)
    inputs[str(Path(__file__))] = sha(__file__)
    write(
        folder / "public_revelation_evidence.json",
        {
            "status": "passed",
            "inputs": inputs,
            "outputs": outputs,
            "tail_contrasts": tail["contrasts"],
            "tail_training_work": tail["training_work"],
            "tail_evaluation_work": tail["evaluation_work"],
            "identification": groups,
            "strict_input_checks": identity["input_check_count_verified"],
            "scope": "Completed studies only; no public-memory strength claim.",
        },
    )
    print({"inputs": len(inputs), "outputs": outputs})


if __name__ == "__main__":
    main()
