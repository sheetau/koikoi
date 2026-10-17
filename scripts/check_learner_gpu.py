"""Validate CUDA learner PPO when the existing GPU slot becomes free.

Optionally launch the already fixed learner-return comparison after all gates
pass. A fresh capability directory is required; a failed check is never retried
or bypassed automatically. No files in the existing study are changed.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.learner_response import LearnerConfig, LearnerTrainer
from scripts.run_arbitration_study import read, sha, write
from scripts.run_response_study import checked as original_checked

SOURCE = Path("runs/response_study")
BASE = Path("runs/learner_response_capability_cuda")
TEST = "tests/test_learner_response.py::test_warmup_restart_and_frozen_weights[cuda]"


def train_slot_free(process, expected):
    if process["status"] == "failed":
        raise RuntimeError("Existing response study failed")
    jobs = [j for j in process["jobs"] if j["mode"] == "train"]
    return (
        len(jobs) == len(expected)
        and {j["case"] for j in jobs} == set(expected)
        and all(j.get("returncode") == 0 for j in jobs)
    )


def run(wait=False, launch=False):
    if BASE.exists():
        raise ValueError("Use a fresh capability directory; inspect any previous run")
    prior = original_checked(SOURCE)
    protocol_sha = sha(SOURCE / "protocol.json")
    paths = [
        Path(__file__),
        Path("docs/LEARNER_RETURN_STUDY.rst"),
        Path("tests/test_learner_response.py"),
        Path("tests/test_learner_return_study.py"),
    ]
    paths += [
        Path("scripts") / name
        for name in (
            "learner_response.py",
            "learner_returns.py",
            "run_learner_return_study.py",
        )
    ]
    inputs = {str(path): sha(path) for path in paths}
    BASE.mkdir(parents=True)
    receipt = {
        "status": "waiting_for_training_slot",
        "source_protocol_sha256": protocol_sha,
        "verification_inputs": inputs,
        "started": time.time(),
        "launch_study": launch,
    }
    write(BASE / "process.json", receipt)
    try:
        while True:
            state = read(SOURCE / "process.json")
            if state["protocol_sha256"] != protocol_sha:
                raise ValueError("Existing study protocol changed")
            if train_slot_free(state, prior["cases"]):
                break
            if not wait:
                raise RuntimeError("Existing GPU trainer is still active; use --wait")
            time.sleep(5)
        original_checked(SOURCE)
        if not all(sha(path) == digest for path, digest in inputs.items()):
            raise ValueError("Verification source changed while waiting")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        receipt.update(status="checking_cuda_restart", gpu_started=time.time())
        write(BASE / "process.json", receipt)
        command = [sys.executable, "-m", "pytest", TEST, "-q", "-rA"]
        with (BASE / "restart_test.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        test_output = (BASE / "restart_test.log").read_text(encoding="utf-8")
        if "1 passed" not in test_output or "skipped" in test_output:
            raise RuntimeError("CUDA restart test did not actually pass")
        receipt.update(
            status="checking_production_size",
            restart_test_command=command,
            restart_test_sha256=sha(BASE / "restart_test.log"),
        )
        write(BASE / "process.json", receipt)
        config = LearnerConfig(
            device="cuda",
            envs=128,
            horizon=128,
            minibatch=512,
            epochs=3,
            critic_width=256,
            critic_warmup_updates=1,
            seed=20295001,
        )
        trainer = LearnerTrainer.from_parent(config, prior["parents"][0]["path"])
        assert not trainer.optimizer.state
        initial = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
        start = time.perf_counter()
        with (BASE / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as log:
            for index in range(3):
                row = (
                    trainer.update()
                    | trainer.counters
                    | {"elapsed_seconds": time.perf_counter() - start}
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                actor = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
                assert (actor == initial) == (index == 0)
                assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        c = trainer.counters
        assert (
            c["decisions"] == 49152 == c["learner_decisions"] + c["published_decisions"]
        )
        assert (
            c["learner_decisions"]
            == c["valid_learner_decisions"] + c["dropped_learner_tails"]
        )
        assert c["dropped_learner_tails"] <= 384
        assert c["warmup_updates"] == 1 and c["policy_updates"] == 2
        assert c["optimizer_steps_value"] + c["skipped_empty_minibatches"] == 288
        # Phase absence is legitimate after learner filtering; compare each real
        # tensor step with the executed phase count, rather than assuming192.
        steps = {}
        for name, parameter in trainer.model.named_parameters():
            state = trainer.optimizer.state.get(parameter)
            steps[name] = 0 if state is None else int(state["step"])
            if name.startswith("value."):
                expected = c["optimizer_steps_value"]
            else:
                phase = name.split(".")[2]
                phase_name = {"0": "discard", "1": "pick", "3": "koi"}[phase]
                expected = c["optimizer_steps_" + phase_name]
                assert 0 < expected <= 192
            assert steps[name] == expected
            if state:
                assert all(torch.isfinite(value).all() for value in state.values())
        trainer.save(BASE / "final.pt")
        original_checked(SOURCE)
        assert all(sha(path) == digest for path, digest in inputs.items())
        report = {
            "purpose": "CUDA capability: one warmup and two policy updates, not strength evidence",
            "checks_passed": True,
            "exact_restart_passed": True,
            "verification_inputs": inputs,
            "restart_test_command": command,
            "restart_test_sha256": sha(BASE / "restart_test.log"),
            "config": asdict(config),
            "counters": c,
            "opponent_work": trainer.opponent_work,
            "provenance": trainer.provenance,
            "runtime": trainer.runtime,
            "initial_actor": initial,
            "final_actor": actor,
            "optimizer_steps": steps,
            "checkpoint_sha256": sha(BASE / "final.pt"),
            "metrics_sha256": sha(BASE / "metrics.jsonl"),
        }
        write(BASE / "audit.json", report)
        receipt.update(
            status="complete", ended=time.time(), audit_sha256=sha(BASE / "audit.json")
        )
        write(BASE / "process.json", receipt)
        print(
            json.dumps(
                {"capability": "passed", "audit_sha256": receipt["audit_sha256"]}
            ),
            flush=True,
        )
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), ended=time.time())
        write(BASE / "process.json", receipt)
        raise
    if launch:
        # Release this process's large trainer before the research subprocess.
        del trainer
        torch.cuda.empty_cache()
        subprocess.run(
            [sys.executable, "-m", "scripts.run_learner_return_study"], check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--launch-study", action="store_true")
    args = parser.parse_args()
    run(args.wait, args.launch_study)


if __name__ == "__main__":
    main()
