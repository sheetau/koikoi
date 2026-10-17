"""Fixed CPU deployment-call timing on cached development states, not matches."""

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.evaluate import make_agent
from koikoi.layout import PHASE
from koikoi.reference import ReferenceEnv
from scripts import run_final_published_test as final
from scripts.run_pretrained_ppo_screen import read, sha, write

BASE = Path("runs/final_latency")
CORPUS = Path("runs/published_batch_validation.npz")
CORPUS_SHA = "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
FINAL_SHA = "4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4"
AUDITOR = Path("scripts/audit_final_latency.py")
REPEATS, WARMUPS = 8, 2


def spread(indices, count):
    if len(indices) < count or count < 2:
        raise ValueError("Insufficient cached states for fixed sampling")
    return indices[np.arange(count) * (len(indices) - 1) // (count - 1)]


def sample_groups(states, sources, guards):
    random = sources >= 0
    groups = {"mixed": spread(np.flatnonzero(random), 1024).tolist()}
    for phase in range(4):
        groups[f"unguarded_phase{phase}"] = spread(
            np.flatnonzero(random & (guards < 0) & (states[:, PHASE] == phase)), 64
        ).tolist()
    groups["guarded"] = spread(np.flatnonzero(random & (guards >= 0)), 32).tolist()
    return groups


def cached():
    if sha(CORPUS) != CORPUS_SHA:
        raise ValueError("Cached corpus identity changed")
    with np.load(CORPUS, allow_pickle=False) as data:
        return {
            name: data[name].copy() for name in ("states", "sources", "masks", "guards")
        }


def project_workers():
    # Read-only inventory; never terminate another process to obtain a timing.
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command], text=True
        ).strip()
        records = json.loads(output) if output else []
        records = records if isinstance(records, list) else [records]
        rows = [(int(r["ProcessId"]), r.get("CommandLine") or "") for r in records]
    else:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        rows = [
            (int(line.split(None, 1)[0]), line.split(None, 1)[1])
            for line in output.splitlines()
            if len(line.split(None, 1)) == 2
        ]
    pattern = r"(?:^|\s)-m\s+(?:koikoi\.(?:evaluate|batch_evaluate|train|published_train|benchmark)\b|scripts\.run_)"
    return [
        {"pid": pid, "command": cmd}
        for pid, cmd in rows
        if pid != os.getpid() and re.search(pattern, cmd)
    ]


def terminal_gate():
    manifest = read(final.BASE / "process.json")
    if (
        manifest.get("status") != "complete_awaiting_audit"
        or len(manifest["jobs"]) != 4
    ):
        raise RuntimeError("Wait for the original final evaluation and complete audit")
    if not all(
        j.get("status") == "complete" and j.get("returncode") == 0
        for j in manifest["jobs"].values()
    ):
        raise RuntimeError("Every original final job must finish successfully")
    audit_path = final.BASE / "independent_audit.json"
    audit = read(audit_path)
    assert audit["status"] == "passed"
    assert manifest["protocol_sha256"] == audit["protocol_sha256"] == FINAL_SHA
    assert audit["auditor_sha256"] == sha(final.AUDITOR)
    assert audit["summary_sha256"] == sha(final.BASE / "summary.json")
    workers = project_workers()
    if workers:
        raise RuntimeError(f"Project workloads still active; do not measure: {workers}")
    return sha(audit_path)


def prepare():
    if BASE.exists():
        raise FileExistsError(BASE)
    f = final.validate_frozen()
    assert sha(final.BASE / "protocol.json") == FINAL_SHA
    data = cached()
    groups = sample_groups(data["states"], data["sources"], data["guards"])
    models = {case: row["checkpoint"] for case, row in f["models"].items()}
    agents = models | {"rl_wp": "rl-wp-assisted", "rl_point": "rl-point-assisted"}
    p = {
        "runner_sha256": sha(__file__),
        "auditor_sha256": sha(AUDITOR),
        "final_protocol_sha256": FINAL_SHA,
        "source_sha256": final.cold.SOURCE,
        "corpus_sha256": CORPUS_SHA,
        "models": f["models"],
        "agents": agents,
        "groups": groups,
        "indices": sorted({i for rows in groups.values() for i in rows}),
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "cpu_threads": 2,
        "device": "cpu",
        "deterministic": True,
        "endgame": True,
        "scope": "Single agent.act(reference_env) call including feature construction, guard and inference; excludes agent loading, reference construction, simulator transitions, correctness checks and process inventories. No new matches or final deal states.",
        "interpretation": "Descriptive cached-state timings, not a policy-induced deployment distribution or strength evidence. Groups overlap; repetitions and states are not independent confidence samples. Agent order rotates across8 passes. Project workload absence is checked at pass boundaries; unrelated OS activity remains uncontrolled.",
    }
    BASE.mkdir(exist_ok=False)
    write(BASE / "protocol.json", p)
    print(
        {
            "prepared": True,
            "unique_states": len(p["indices"]),
            "protocol_sha256": sha(BASE / "protocol.json"),
        }
    )


def validate():
    p = read(BASE / "protocol.json")
    assert p["runner_sha256"] == sha(__file__) and p["auditor_sha256"] == sha(AUDITOR)
    assert p["repeats"] == REPEATS and p["warmups"] == WARMUPS
    assert p["cpu_threads"] == 2 and p["device"] == "cpu"
    assert p["deterministic"] and p["endgame"]
    assert p["source_sha256"] == final.cold.SOURCE
    assert p["corpus_sha256"] == CORPUS_SHA
    assert p["final_protocol_sha256"] == sha(final.BASE / "protocol.json") == FINAL_SHA
    f = final.validate_frozen()
    assert p["models"] == f["models"]
    assert p["agents"] == {
        case: row["checkpoint"] for case, row in f["models"].items()
    } | {"rl_wp": "rl-wp-assisted", "rl_point": "rl-point-assisted"}
    data = cached()
    assert p["groups"] == sample_groups(data["states"], data["sources"], data["guards"])
    assert p["indices"] == sorted({i for rows in p["groups"].values() for i in rows})
    return p, data


def statistics(values):
    milliseconds = np.asarray(values, dtype=np.float64).flatten() / 1e6
    return {
        "calls": len(milliseconds),
        "mean_ms": float(milliseconds.mean()),
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.quantile(milliseconds, 0.95)),
        "p99_ms": float(np.quantile(milliseconds, 0.99)),
        "max_ms": float(milliseconds.max()),
    }


def deployment_agent(spec):
    agent = make_agent(spec, 817001, deterministic=True, device="cpu")
    # Match evaluate --endgame exactly: candidates also have their checkpoint
    # guard internally; assisted comparator construction already wraps once.
    return (
        agent
        if spec in ("rl-wp-assisted", "rl-point-assisted")
        else EndgameAgent(agent)
    )


def run():
    # The live-study guard precedes any model load or output-file creation.
    audit_sha = terminal_gate()
    p, data = validate()
    for name in ("report.json", "raw.npz", "process.json"):
        if (BASE / name).exists():
            raise FileExistsError("Timing already started; inspect existing artifacts")
    write(
        BASE / "process.json",
        {
            "pid": os.getpid(),
            "status": "running",
            "protocol_sha256": sha(BASE / "protocol.json"),
        },
    )
    try:
        measure(p, data, audit_sha)
    except BaseException as exc:
        write(
            BASE / "process.json",
            {
                "pid": os.getpid(),
                "status": "failed",
                "error": type(exc).__name__ + ": " + str(exc),
                "protocol_sha256": sha(BASE / "protocol.json"),
            },
        )
        raise


def measure(p, data, audit_sha):
    torch.set_num_threads(2)
    torch.manual_seed(817001)
    indices = np.array(p["indices"])
    states, masks, guards = (
        data[name][indices] for name in ("states", "masks", "guards")
    )
    environments = []
    for state in states:
        env = ReferenceEnv.__new__(ReferenceEnv)
        env.state = state.copy()
        environments.append(env)
    agents, load_seconds = {}, {}
    for name, spec in p["agents"].items():
        start = time.perf_counter()
        agents[name] = deployment_agent(spec)
        if name in p["models"]:
            assert agents[name].agent.checkpoint_sha256 == p["models"][name]["sha256"]
        load_seconds[name] = time.perf_counter() - start
    names = list(agents)
    expected = np.empty((len(names), len(indices)), dtype=np.int64)

    def check(row, action):
        assert 0 <= action < 50 and masks[row, action]
        assert guards[row] < 0 or action == guards[row]
        np.testing.assert_array_equal(environments[row].state, states[row])

    for a, name in enumerate(names):
        for i, env in enumerate(environments):
            expected[a, i] = agents[name].act(env)
            check(i, expected[a, i])
    for _ in range(WARMUPS):
        for a, name in enumerate(names):
            for i, env in enumerate(environments):
                action = agents[name].act(env)
                assert action == expected[a, i]
                check(i, action)
    times = np.empty((len(names), REPEATS, len(indices)), dtype=np.int64)
    actions = np.empty_like(times)
    passes = []
    for repeat in range(REPEATS):
        assert not project_workers(), "Another project workload appeared"
        order = [(repeat + i) % len(names) for i in range(len(names))]
        started = time.time()
        for a in order:
            for i, env in enumerate(environments):
                start = time.perf_counter_ns()
                action = agents[names[a]].act(env)
                times[a, repeat, i] = time.perf_counter_ns() - start
                actions[a, repeat, i] = action
                assert action == expected[a, i]
                check(i, action)
        assert not project_workers(), "Another project workload appeared"
        passes.append(
            {
                "order": [names[a] for a in order],
                "started": started,
                "finished": time.time(),
                "project_workers_before": [],
                "project_workers_after": [],
            }
        )
    assert np.all(times > 0)
    assert terminal_gate() == audit_sha
    validate()
    np.savez_compressed(
        BASE / "raw.npz",
        indices=indices,
        elapsed_ns=times,
        actions=actions,
        expected_actions=expected,
    )
    position = {int(index): i for i, index in enumerate(indices)}
    summaries = {
        name: {
            group: statistics(times[a, :, [position[i] for i in rows]])
            for group, rows in p["groups"].items()
        }
        for a, name in enumerate(names)
    }
    report = {
        "status": "complete_awaiting_audit",
        "protocol_sha256": sha(BASE / "protocol.json"),
        "raw_sha256": sha(BASE / "raw.npz"),
        "final_audit_sha256": audit_sha,
        "agents": names,
        "summaries": summaries,
        "passes": passes,
        "load_seconds": load_seconds,
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_threads": torch.get_num_threads(),
        },
        "measured_calls": int(times.size),
        "warmup_calls": len(names) * WARMUPS * len(indices),
        "validation_calls": len(names) * len(indices),
        "simulated_matches": 0,
    }
    write(BASE / "report.json", report)
    write(
        BASE / "process.json",
        {
            "pid": os.getpid(),
            "status": "complete_awaiting_audit",
            "protocol_sha256": sha(BASE / "protocol.json"),
        },
    )
    print({"complete": True, "measured_calls": int(times.size)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == "__main__":
    main()
