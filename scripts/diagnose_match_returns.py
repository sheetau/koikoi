"""Fixed complete-match diagnostic of trace decay and horizon bootstrapping.

Frozen sampled self-play, no optimizer or checkpoint selection. Store every
action and native-state digest so a CPU auditor can replay the entire corpus.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.estimators import native_gae
from koikoi.layout import PHASE, PLAYER, ROUND
from koikoi.model import sample_action
from koikoi.train import Config, Trainer, provenance

BASE = Path("runs/match_return_diagnostic")
SOURCE = "1c36895f2b5164c4c0a86862e2270364a74c2ff0ee4d950f4b0c4cd970575914"
PARENTS = (
    (
        "seed1",
        "runs/ppo8_v3/update_00005000.pt",
        "d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5",
        6420001,
    ),
    (
        "seed10001",
        "runs/ppo8_seed10001_captured/update_00005000.pt",
        "edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d",
        6430001,
    ),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def statistics(arrays):
    valid = arrays["valid"]
    lengths = valid.sum(0)
    distance = lengths[None] - np.arange(len(valid))[:, None] - 1
    masks = {
        "all": valid,
        "rounds_1_2": valid & (arrays["round"] < 2),
        "rounds_3_6": valid & (arrays["round"] >= 2) & (arrays["round"] < 6),
        "rounds_7_8": valid & (arrays["round"] >= 6),
        "terminal_distance_gt64": valid & (distance > 64),
        "terminal_distance_le16": valid & (distance <= 16),
    }
    result = {}
    for group, mask in masks.items():
        truth = arrays["mc"][mask].astype(np.float64)
        value = arrays["value"][mask].astype(np.float64)
        truth_advantage = truth - value
        row = {"rows": int(mask.sum()), "estimators": {}}
        if not len(truth):
            result[group] = row
            continue
        row["mean_terminal_trace_weight"] = float(np.mean(0.95 ** distance[mask]))
        row["mc_advantage_std"] = float(truth_advantage.std())
        for key in ("gae95_full", "gae95_h128", "gae1_h128"):
            estimate = arrays[key][mask].astype(np.float64)
            advantage = estimate - value
            row["estimators"][key] = {
                "return_rmse_vs_realized_mc": float(
                    np.sqrt(np.mean((estimate - truth) ** 2))
                ),
                "advantage_std": float(advantage.std()),
                "advantage_sign_disagreement": float(
                    np.mean(advantage * truth_advantage < 0)
                ),
                "advantage_correlation": float(
                    np.corrcoef(advantage, truth_advantage)[0, 1]
                ),
            }
        delta = arrays["gae95_h128"][mask] - arrays["gae95_full"][mask]
        row["h128_vs_full95_rmse"] = float(
            np.sqrt(np.mean(delta.astype(np.float64) ** 2))
        )
        result[group] = row
    return result


@torch.no_grad()
def collect(path, seed):
    trainer = Trainer(
        Config(
            profile="paper8",
            envs=1024,
            horizon=128,
            seed=seed,
            cpu_threads=2,
            endgame=True,
        )
    )
    trainer.initialize_weights(path)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    initial = trainer.env.state.cpu().numpy().copy()
    records = {
        k: []
        for k in ("action", "player", "value", "valid", "phase", "round", "result")
    }
    digests = []
    for _ in range(1024):
        state = trainer.env.state
        valid = state[:, PHASE] != 4
        if not valid.any():
            break
        digests.append(state_digest(state))
        obs, mask = trainer.observe()
        logits, value, _ = trainer.actor(obs, mask)
        # Padded terminal rows consume RNG but neither act nor contribute returns.
        action, _ = sample_action(logits)
        for key, tensor in {
            "action": action,
            "player": state[:, PLAYER],
            "value": value.masked_fill(~valid, 0),
            "valid": valid,
            "phase": state[:, PHASE],
            "round": state[:, ROUND],
        }.items():
            records[key].append(tensor.clone())
        output = trainer.env.step(action, freeze_done=True)
        assert not (output[:, 0] == 99).any()
        records["result"].append(output)
    assert (trainer.env.state[:, PHASE] == 4).all(), "Incomplete diagnostic matches"
    records = {k: torch.stack(v) for k, v in records.items()}
    valid, player, value = (records[k] for k in ("valid", "player", "value"))
    result = records["result"]
    done = result[:, :, 2].bool() | ~valid
    reward = result[:, :, 0].float() * (1 - 2 * player)
    zero = torch.zeros(1024, device="cuda")
    last_player = trainer.env.state[:, PLAYER]
    mc = result[:, :, 0].sum(0)[None].float() * (1 - 2 * player)
    records["mc"] = mc
    for lam in (0.95, 1.0):
        _, full = native_gae(
            trainer.env.backend, reward, value, done, player, zero, last_player, lam
        )
        if lam == 1:
            torch.testing.assert_close(full[valid], mc[valid], rtol=0, atol=2e-6)
        else:
            records["gae95_full"] = full
        chunks = []
        for start in range(0, len(value), 128):
            end = min(start + 128, len(value))
            next_value = value[end] if end < len(value) else zero
            next_player = player[end] if end < len(value) else last_player
            _, estimate = native_gae(
                trainer.env.backend,
                reward[start:end],
                value[start:end],
                done[start:end],
                player[start:end],
                next_value,
                next_player,
                lam,
            )
            chunks.append(estimate)
        records["gae95_h128" if lam == 0.95 else "gae1_h128"] = torch.cat(chunks)
    assert not trainer.optimizer.state
    assert all(torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())
    arrays = {k: v.cpu().numpy() for k, v in records.items()}
    arrays["initial"] = initial
    arrays["final"] = trainer.env.state.cpu().numpy()
    arrays["state_digests"] = np.asarray(digests)
    counts = {
        "matches": int(result[:, :, 2].sum()),
        "rounds": int(result[:, :, 1].sum()),
        "active_decisions": int(valid.sum()),
        "executed_rows_including_frozen": int(valid.numel()),
        "steps": len(valid),
        "updates": 0,
    }
    assert counts["matches"] == 1024
    assert counts["rounds"] == int((trainer.env.state[:, ROUND] + 1).sum())
    return arrays, counts


def main():
    assert provenance()["source_sha256"] == SOURCE
    for _, path, digest, _ in PARENTS:
        assert sha(path) == digest
    BASE.mkdir(exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {
        "source_sha256": SOURCE,
        "script_sha256": sha(__file__),
        "protocol": {
            "parents": PARENTS,
            "batches": 4,
            "envs": 1024,
            "max_steps": 1024,
            "profile": "paper8",
            "endgame": True,
            "sampled_selfplay": True,
            "updates": 0,
            "horizon": 128,
            "seed_stride": 1024,
        },
        "cases": {},
    }
    (BASE / "protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for label, path, digest, first_seed in PARENTS:
        rows = []
        for batch in range(4):
            seed = first_seed + 1024 * batch
            arrays, counts = collect(path, seed)
            raw_path = BASE / f"{label}_{batch}.npz"
            np.savez_compressed(raw_path, **arrays)
            rows.append(
                {
                    "seed": seed,
                    "counts": counts,
                    "statistics": statistics(arrays),
                    "raw_sha256": sha(raw_path),
                }
            )
            print(
                json.dumps(
                    {
                        "case": label,
                        "batch": batch,
                        "counts": counts,
                        "all": rows[-1]["statistics"]["all"],
                    }
                ),
                flush=True,
            )
        report["cases"][label] = {"parent_sha256": digest, "batches": rows}
    report["status"] = "complete"
    report["limitations"] = (
        "Realized Monte Carlo outcomes are noisy, not conditional true values. "
        "Return RMSE/sign differences cannot establish biased gradients or improved strength. "
        "Windows aligned to match starts differ from steady-state training windows. "
        "Policies are sampled frozen self-play, not deterministic/public-opponent deployment."
    )
    (BASE / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
