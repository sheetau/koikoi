"""Recompute corpus identities, raw numerical summaries and timing arithmetic.

No imports from the validation runner or batch policy. Inference itself is not
independently rerun here; feature parity and numerical results are finite tests.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from koikoi.layout import BANKRUPT, KOI, PHASE, PLAYER, POINTS, ROUND, ROUNDS, TERMINAL
from koikoi.reference import ReferenceEnv

BASE = Path("runs/published_batch_validation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    report = json.loads(BASE.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["raw_sha256"] == sha(BASE.with_suffix(".npz"))
    assert report["script_sha256"] == sha("scripts/validate_published_batch.py")
    package = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            package.update(path.as_posix().encode())
            package.update(path.read_bytes())
    assert package.hexdigest() == report["provenance"]["source_sha256"]
    for variant, teacher in report["teacher"].items():
        for model, digest in teacher["models"].items():
            assert (
                sha(f"external/KoiKoi-AI/model_agent/{model}_rl_{variant}.pt") == digest
            )
    p = report["protocol"]
    assert p["matches"] == 100 and p["first_seed"] == 6200100
    assert p["action_rng"] == 62021 and p["continue_probability"] == 0.7
    assert p["batch_size"] == 256 and p["cpu_threads"] == 2 and not p["tf32"]
    with np.load(BASE.with_suffix(".npz")) as raw:
        states, masks, guards = raw["states"], raw["masks"], raw["guards"]
        assert states.dtype == np.int32 and states.shape[1] == 208
        assert masks.shape == (len(states), 50) and masks.dtype == bool
        rng = np.random.default_rng(62021)
        cursor, rounds = 0, 0
        for match, seed in enumerate(range(6200100, 6200200)):
            env = ReferenceEnv(seed)
            while env.state[PHASE] != TERMINAL:
                np.testing.assert_array_equal(env.state, states[cursor])
                assert raw["sources"][cursor] == seed
                if env.state[PHASE] == KOI:
                    action = 49 if rng.random() < 0.7 else 48
                else:
                    action = int(rng.choice(np.where(env.legal())[0]))
                result = env.step(action)
                rounds += int(result[1])
                cursor += 1
            np.testing.assert_array_equal(env.state, raw["terminal_states"][match])
        assert cursor == report["counters"]["trajectory_decisions"]
        assert rounds == report["counters"]["rounds"]
        assert rounds == sum(int(s[ROUND]) + 1 for s in raw["terminal_states"])
        cached_path = Path("runs/rollout_opponent_roots.json")
        assert sha(cached_path) == report["cached_roots_sha256"]
        roots = json.loads(cached_path.read_text(encoding="utf-8"))["roots"]
        assert len(roots) == report["counters"]["cached_roots"] == 40
        for row in roots:
            np.testing.assert_array_equal(states[cursor], row["state"])
            assert (
                hashlib.sha256(states[cursor].tobytes()).hexdigest()
                == row["state_sha256"]
            )
            assert raw["sources"][cursor] == -1
            cursor += 1
        assert len(states) == cursor + len(raw["hidden_pairs"])
        assert len(raw["hidden_pairs"]) == report["counters"]["hidden_variants"] == 128
        np.testing.assert_array_equal(
            raw["hidden_pairs"][:, 1], np.arange(cursor, len(states))
        )
        np.testing.assert_array_equal(raw["sources"][cursor:], -2)
        assert len(states) == report["counters"]["states"]
        for index, state in enumerate(states):
            env = ReferenceEnv.__new__(ReferenceEnv)
            env.state = state.copy()
            np.testing.assert_array_equal(masks[index], env.legal())
            expected = -1
            if state[PHASE] == KOI:
                player = int(state[PLAYER])
                gain = env.yaku(player)
                own = int(state[POINTS + player]) + gain
                other = int(state[POINTS + 1 - player]) - gain
                if state[BANKRUPT] and other <= 0:
                    expected = 48
                elif state[ROUND] + 1 == state[ROUNDS]:
                    expected = 48 if own > other else 49 if own < other else -1
            assert guards[index] == expected
        for key in ("cpu_features_sha256", "cuda_features_sha256"):
            np.testing.assert_array_equal(raw[key], raw["oracle_features_sha256"])
        for parent, hidden in raw["hidden_pairs"]:
            np.testing.assert_array_equal(
                raw["oracle_features_sha256"][parent],
                raw["oracle_features_sha256"][hidden],
            )
            a, b = (
                ReferenceEnv.__new__(ReferenceEnv),
                ReferenceEnv.__new__(ReferenceEnv),
            )
            a.state, b.state = states[parent].copy(), states[hidden].copy()
            np.testing.assert_array_equal(a.observe(), b.observe())
            np.testing.assert_array_equal(masks[parent], masks[hidden])
        summaries = {}
        for variant in ("wp", "point"):
            oracle = raw[f"oracle_{variant}"]
            assert np.isfinite(oracle[masks]).all()
            assert np.isneginf(oracle[~masks]).all()
            for device in ("cpu", "cuda"):
                name = f"{device}_{variant}"
                output = raw[name]
                assert np.isfinite(output[masks]).all()
                assert np.isneginf(output[~masks]).all()
                different, assisted, margins = [], 0, []
                maximum = 0.0
                by_phase = {
                    str(phase): {"states": 0, "action_disagreements": 0}
                    for phase in range(4)
                }
                for i in range(len(states)):
                    legal = np.flatnonzero(masks[i])
                    expected = int(legal[np.argmax(oracle[i, legal])])
                    actual = int(legal[np.argmax(output[i, legal])])
                    assert (
                        expected == oracle[i].argmax() and actual == output[i].argmax()
                    )
                    phase = str(states[i, PHASE])
                    by_phase[phase]["states"] += 1
                    maximum = max(
                        maximum,
                        float(np.abs(output[i, legal] - oracle[i, legal]).max()),
                    )
                    if actual != expected:
                        different.append(i)
                        by_phase[phase]["action_disagreements"] += 1
                        top = sorted(oracle[i, legal])
                        margins.append(float(top[-1] - top[-2]))
                        assisted += int(guards[i] < 0)
                computed = {
                    "states": len(states),
                    "action_disagreements": len(different),
                    "assisted_action_disagreements": assisted,
                    "max_legal_logit_error": maximum,
                    "by_phase": by_phase,
                    "disagreement_indices": different,
                    "oracle_margins_at_disagreements": margins,
                }
                assert computed == report["comparisons"][name]
                summaries[name] = computed
    times = report["timings_wp_unassisted"]
    assert len(times) == 68
    for row in times:
        ordered = sorted(row["seconds"])
        count = len(ordered)
        expected_repeats = {"features": 20, "policy": 5, "strict_policy_serial": 3}[
            row["operation"]
        ]
        assert count == expected_repeats and ordered[0] > 0
        middle = (
            ordered[count // 2]
            if count % 2
            else sum(ordered[count // 2 - 1 : count // 2 + 1]) / 2
        )
        assert row["median_seconds"] == middle
        assert row["states_per_second"] == row["batch"] / middle
    assert (
        report["counters"]["training_matches"]
        == report["counters"]["strength_matches"]
        == 0
    )
    result = {
        "passed": True,
        "report_sha256": sha(BASE.with_suffix(".json")),
        "raw_sha256": sha(BASE.with_suffix(".npz")),
        "auditor_sha256": sha(__file__),
        "source_sha256": package.hexdigest(),
        "states": len(states),
        "summaries": summaries,
        "timing_rows": len(times),
        "scope": "Independent replay of random corpus; cached-state hashes; reference masks/guards/public observations; stored feature digest equality; independent raw-logit and latency arithmetic. Inference and timing measurements are not independently rerun.",
    }
    out = Path("runs/published_batch_independent_audit.json")
    out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"passed": True, "states": len(states), "output": str(out)}))


if __name__ == "__main__":
    check()
