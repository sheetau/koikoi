"""Conditional V5A continuation and qualification, serial and non-overwriting."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.export_public_stock_adaptation import audit_training, checked_endpoint
from scripts.export_public_stock_results import read, sha, write
from scripts.public_stock_adaptation import (
    EVALUATE,
    make_protocol,
    training_config,
    verify_protocol,
)


def run(module, options, log):
    command = [sys.executable, "-X", "utf8", "-m", module]
    for key, value in options.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    print(json.dumps({"command": command}), flush=True)
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    base = Path(args.out)
    if base.exists() or Path(args.evidence).exists():
        raise FileExistsError("V5A outputs already exist; inspect before recovery")
    study = make_protocol()
    verify_protocol(study)
    base.mkdir(parents=True, exist_ok=False)
    write(base / "protocol.json", study)
    audits = {}
    for label in study["systems"]:
        verify_protocol(study)
        run("scripts.continue_public_stock_residual", training_config(base, study, label), base / f"train_{label}.log")
        audits[label] = audit_training(base, study, label, {})
        print(json.dumps({"trained_and_audited": label}), flush=True)
    write(base / "training_audit.json", {"protocol_sha256": sha(base / "protocol.json"), "systems": audits})
    audit_digest = sha(base / "training_audit.json")
    for label, row in study["systems"].items():
        other = study["systems"][f"{row['profile']}_parent{1 - row['parent']}"]["checkpoint"]
        new = audits[label]["checkpoint"]
        for kind, checkpoint, opponent in (("new_own", new, row["checkpoint"]), ("new_other", new, other), ("old_other", row["checkpoint"], other)):
            verify_protocol(study)
            if sha(base / "training_audit.json") != audit_digest or any(
                sha(a["checkpoint"]) != a["checkpoint_sha256"] for a in audits.values()
            ):
                raise ValueError("Trained policies or pre-qualification audit changed")
            name = f"qualification_{label}_{kind}"
            folder = base / name
            run("scripts.evaluate_public_stock_residual", {
                **EVALUATE, "checkpoint": checkpoint, "opponent": opponent, "out": str(folder),
            }, base / f"{name}.log")
            checked_endpoint(folder, study, checkpoint, opponent, row["profile"], {})
            print(json.dumps({"completed": name}), flush=True)
    run("scripts.export_public_stock_adaptation", {"study": str(base), "out": args.evidence}, base / "export.log")
    result = read(args.evidence)
    write(base / "finished.json", {"status": "complete", "evidence": args.evidence, "sha256": sha(args.evidence)})
    print(json.dumps({k: result[k] for k in ("qualified_rules", "eligible_search_finalists")}), flush=True)


if __name__ == "__main__":
    main()
