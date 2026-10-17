"""Run existing independent analyses after each fixed study finishes."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def jobs():
    return [
        (
            "arbitration",
            Path("runs/arbitration_study"),
            [("scripts.analyze_arbitration_study", [])],
        ),
        (
            "paired",
            Path("runs/paired_arbitration_study"),
            [("scripts.analyze_paired_arbitration_study", [])]
            + [
                (
                    "scripts.diagnose_paired_targets",
                    [
                        "--dataset",
                        f"runs/paired_arbitration_study/parent{index}/targets.pt",
                        "--checkpoints",
                        f"runs/paired_arbitration_study/parent{index}_full/final.pt",
                        f"runs/paired_arbitration_study/parent{index}_no_match/final.pt",
                        "--output",
                        f"runs/paired_arbitration_study/parent{index}/held_out_diagnostic.json",
                    ],
                )
                for index in (1, 2)
            ],
        ),
        (
            "static",
            Path("runs/static_expert_controls"),
            [("scripts.analyze_static_expert_controls", [])],
        ),
    ]


def run(base):
    if base.exists():
        raise ValueError("Use a fresh audit receipt directory")
    base.mkdir(parents=True)
    specifications = jobs()
    commands = {
        module: sha(Path(module.replace(".", "/") + ".py"))
        for _, _, commands in specifications
        for module, _ in commands
    }
    write(
        base / "plan.json",
        {
            "runner_sha256": sha(__file__),
            "analysis_scripts": commands,
            "scope": "Automatic read-only analysis of fixed completed endpoints; no training, evaluation, selection or deployment changes.",
        },
    )
    try:
        for label, study, steps in specifications:
            write(base / "process.json", {"status": "waiting", "study": label})
            while True:
                status = read(study / "process.json")["status"]
                if status == "complete":
                    break
                if status == "failed":
                    raise RuntimeError(f"{label} study failed")
                time.sleep(10)
            for number, (module, arguments) in enumerate(steps):
                source = Path(module.replace(".", "/") + ".py")
                if sha(source) != commands[module]:
                    raise RuntimeError(f"Analysis changed: {module}")
                write(
                    base / "process.json",
                    {"status": "analyzing", "study": label, "module": module},
                )
                with (base / f"{label}_{number}.log").open(
                    "x", encoding="utf-8"
                ) as log:
                    subprocess.run(
                        [sys.executable, "-m", module, *arguments],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            write(base / f"{label}_complete.json", {"finished": time.time()})
    except BaseException as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise
    write(base / "process.json", {"status": "complete", "finished": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/extension_audits"))
    run(parser.parse_args().base)


if __name__ == "__main__":
    main()
