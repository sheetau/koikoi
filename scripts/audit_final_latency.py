"""Audit cached-state timing records; does not rerun or certify wall-clock timings."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path("runs/final_latency")
CORPUS = Path("runs/published_batch_validation.npz")
CORPUS_SHA = "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
FINAL_SHA = "4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def groups_from_records(states, sources, guards):
    sets = {"mixed": [], **{f"unguarded_phase{i}": [] for i in range(4)}, "guarded": []}
    for i, (state, source, guard) in enumerate(
        zip(states, sources, guards, strict=True)
    ):
        if source < 0:
            continue
        sets["mixed"].append(i)
        sets["guarded" if guard >= 0 else f"unguarded_phase{int(state[99])}"].append(i)
    selected = {}
    for name, rows in sets.items():
        count = 1024 if name == "mixed" else 32 if name == "guarded" else 64
        assert len(rows) >= count
        selected[name] = [
            rows[(len(rows) - 1) * i // (count - 1)] for i in range(count)
        ]
    return selected


def statistics(values):
    rows = sorted(float(v) / 1e6 for v in np.asarray(values).ravel())
    assert rows and all(math.isfinite(v) and v > 0 for v in rows)

    def quantile(q):
        x = (len(rows) - 1) * q
        i = math.floor(x)
        return rows[i] + (x - i) * (rows[min(i + 1, len(rows) - 1)] - rows[i])

    return {
        "calls": len(rows),
        "mean_ms": math.fsum(rows) / len(rows),
        "median_ms": quantile(0.5),
        "p95_ms": quantile(0.95),
        "p99_ms": quantile(0.99),
        "max_ms": rows[-1],
    }


def verify_arrays(p, data, raw):
    groups = groups_from_records(data["states"], data["sources"], data["guards"])
    assert p["groups"] == groups
    indices = sorted({i for rows in groups.values() for i in rows})
    assert p["indices"] == indices
    np.testing.assert_array_equal(raw["indices"], indices)
    n = len(indices)
    elapsed, actions, expected = (
        raw[key] for key in ("elapsed_ns", "actions", "expected_actions")
    )
    assert elapsed.shape == actions.shape == (4, 8, n) and expected.shape == (4, n)
    assert np.issubdtype(elapsed.dtype, np.integer) and np.all(elapsed > 0)
    assert np.issubdtype(actions.dtype, np.integer) and np.issubdtype(
        expected.dtype, np.integer
    )
    assert np.all((actions >= 0) & (actions < 50))
    assert np.all(actions == expected[:, None, :])
    for a in range(4):
        for i, original in enumerate(indices):
            action = int(expected[a, i])
            assert data["masks"][original, action]
            guard = int(data["guards"][original])
            assert guard < 0 or action == guard
    return indices


def main():
    p, r, m = [
        read(BASE / name) for name in ("protocol.json", "report.json", "process.json")
    ]
    assert p["runner_sha256"] == sha("scripts/benchmark_final_latency.py")
    assert p["auditor_sha256"] == sha(__file__)
    assert r["protocol_sha256"] == m["protocol_sha256"] == sha(BASE / "protocol.json")
    assert r["status"] == m["status"] == "complete_awaiting_audit"
    assert p["repeats"] == 8 and p["warmups"] == 2
    assert p["cpu_threads"] == r["runtime"]["cpu_threads"] == 2
    assert p["device"] == "cpu" and p["deterministic"] and p["endgame"]
    assert p["corpus_sha256"] == sha(CORPUS) == CORPUS_SHA
    assert (
        sha("runs/final_published_test/protocol.json")
        == p["final_protocol_sha256"]
        == FINAL_SHA
    )
    final = read("runs/final_published_test/protocol.json")
    assert (
        p["models"] == final["models"]
        and p["source_sha256"] == final["provenance"]["source_sha256"]
    )
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == p["source_sha256"]
    for path, value in final["bindings"].items():
        assert sha(path) == value
    assert p["agents"] == {
        case: row["checkpoint"] for case, row in final["models"].items()
    } | {"rl_wp": "rl-wp-assisted", "rl_point": "rl-point-assisted"}
    for row in p["models"].values():
        assert sha(row["checkpoint"]) == row["sha256"]
    assert r["final_audit_sha256"] == sha(
        "runs/final_published_test/independent_audit.json"
    )
    final_audit = read("runs/final_published_test/independent_audit.json")
    assert (
        final_audit["status"] == "passed"
        and final_audit["protocol_sha256"] == FINAL_SHA
    )
    assert final_audit["auditor_sha256"] == sha("scripts/audit_final_published_test.py")
    assert final_audit["summary_sha256"] == sha(
        "runs/final_published_test/summary.json"
    )
    assert r["raw_sha256"] == sha(BASE / "raw.npz")
    with (
        np.load(CORPUS, allow_pickle=False) as data,
        np.load(BASE / "raw.npz", allow_pickle=False) as raw,
    ):
        indices = verify_arrays(p, data, raw)
        measured_pass_seconds = [
            int(raw["elapsed_ns"][:, i, :].sum()) / 1e9 for i in range(8)
        ]
        positions = {original: index for index, original in enumerate(indices)}
        assert r["agents"] == list(p["agents"]) and len(r["agents"]) == 4
        assert set(r["summaries"]) == set(r["agents"])
        recomputed = {}
        for a, name in enumerate(r["agents"]):
            assert set(r["summaries"][name]) == set(p["groups"])
            recomputed[name] = {}
            for group, rows in p["groups"].items():
                stats = statistics(
                    raw["elapsed_ns"][a, :, [positions[i] for i in rows]]
                )
                assert set(stats) == set(r["summaries"][name][group])
                for key, value in stats.items():
                    assert math.isclose(
                        value,
                        r["summaries"][name][group][key],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                recomputed[name][group] = stats
    assert r["measured_calls"] == 4 * 8 * len(indices)
    assert r["warmup_calls"] == 4 * 2 * len(indices)
    assert r["validation_calls"] == 4 * len(indices) and r["simulated_matches"] == 0
    assert len(r["passes"]) == 8
    assert set(r["load_seconds"]) == set(r["agents"])
    assert all(
        math.isfinite(value) and value > 0 for value in r["load_seconds"].values()
    )
    previous = -math.inf
    for i, row in enumerate(r["passes"]):
        assert row["order"] == [r["agents"][(i + j) % 4] for j in range(4)]
        assert math.isfinite(row["started"]) and math.isfinite(row["finished"])
        assert previous <= row["started"] < row["finished"]
        assert measured_pass_seconds[i] <= row["finished"] - row["started"]
        assert row["project_workers_before"] == row["project_workers_after"] == []
        previous = row["finished"]
    output = {
        "status": "passed",
        "scope": "artifact identities, fixed cached sampling, raw timing/action accounting and independent descriptive statistics; not an independent timing or inference rerun; project workload absence is observed at pass boundaries only",
        "protocol_sha256": sha(BASE / "protocol.json"),
        "report_sha256": sha(BASE / "report.json"),
        "auditor_sha256": sha(__file__),
        "summaries": recomputed,
        "unique_cached_states": len(indices),
        "simulated_matches": 0,
    }
    (BASE / "independent_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print({"status": "passed", "unique_cached_states": len(indices)})


if __name__ == "__main__":
    main()
