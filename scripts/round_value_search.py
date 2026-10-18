"""Experimental history-conditioned one-round rollout with independent A/B tables.

Joint bootstrap includes shared table error and particle sampling. It remains an
approximate standard-error decision margin, not a safe-policy guarantee.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.search import sample_hidden_batch
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.history_conditioned_worlds import sample_joint_worlds
from scripts.round_transition_values import RoundWork, RoundWorkspace
from scripts.run_arbitration_study import PARENTS, sha


class BoundaryTables:
    """Signed parent values with explicit table/round/seat/model/points/dealer axes."""

    def __init__(self, values, bootstrap, checkpoint_sha256, identity):
        self.values = np.array(values, dtype=np.float64, copy=True)
        self.bootstrap = np.array(bootstrap, dtype=np.float64, copy=True)
        if self.values.shape != (2, 8, 2, 2, 59, 2):
            raise ValueError("Expected independent A/B public-context tables")
        if (
            self.bootstrap.ndim != 7
            or self.bootstrap.shape[0] != 2
            or self.bootstrap.shape[1] < 2
            or self.bootstrap.shape[2:] != self.values.shape[1:]
        ):
            raise ValueError("Expected A/B bootstrap values with the same public axes")
        for array in (self.values, self.bootstrap):
            if not np.isfinite(array).all() or (np.abs(array) > 1).any():
                raise ValueError("Boundary values must be finite signed utilities")
            array.setflags(write=False)
        self.checkpoint_sha256, self.identity = checkpoint_sha256, identity

    @classmethod
    def from_accuracy_study(cls, folder, parent):
        folder = Path(folder)
        if parent not in (1, 2):
            raise ValueError("Unknown retained parent")
        analysis = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
        if analysis["status"] != "passed":
            raise ValueError("Complete the estimator study before using its artifacts")
        assert all(sha(p) == digest for p, digest in analysis["inputs"].items())
        assert all(
            sha(folder / name) == digest
            for name, digest in analysis["artifacts"].items()
        )
        values = []
        for table in ("A", "B"):
            path = folder / f"parent{parent}_{table}"
            protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
            assert (
                protocol["parent"] == parent and protocol["samples_per_context"] == 128
            )
            assert protocol["seed"] == {"A": 42110001, "B": 52110001}[table]
            assert (
                protocol["inputs"][str(Path(PARENTS[parent - 1][0]))]
                == PARENTS[parent - 1][1]
            )
            with np.load(path / "kernels.npz", allow_pickle=False) as data:
                values.append(data["values"])
        with np.load(folder / "bootstrap_values.npz", allow_pickle=False) as data:
            bootstrap = data["values"][parent - 1]
        return cls(
            values, bootstrap, PARENTS[parent - 1][1], sha(folder / "analysis.json")
        )

    def lookup(self, rounds, points0, dealers, terminal, seat, labels, table):
        """Only public endpoint fields and sampled model labels enter the lookup."""
        rounds, points0, dealers, terminal, labels = map(
            np.asarray, (rounds, points0, dealers, terminal, labels)
        )
        if rounds.ndim != 1 or any(
            a.shape != rounds.shape for a in (points0, dealers, terminal, labels)
        ):
            raise ValueError("Expected one public endpoint/model label per lane")
        if (
            any(a.dtype.kind not in "iu" for a in (rounds, points0, dealers, labels))
            or terminal.dtype != bool
        ):
            raise ValueError("Invalid endpoint types")
        if (
            seat not in (0, 1)
            or table not in (0, 1)
            or not np.isin(labels, (0, 1)).all()
            or not np.isin(dealers, (0, 1)).all()
        ):
            raise ValueError("Unknown seat, table, dealer or model")
        if not ((rounds >= 0) & (rounds < 8)).all():
            raise ValueError("Invalid endpoint round")
        live = ~terminal
        if not (
            (points0[live] >= 1) & (points0[live] <= 59) & (rounds[live] >= 1)
        ).all():
            raise ValueError("Nonterminal endpoints must be later live rounds")
        value = (np.sign(points0.astype(np.int64) - 30) * (1 - 2 * seat)).astype(float)
        samples = np.broadcast_to(value, (self.bootstrap.shape[1], len(value))).copy()
        key = (rounds[live], seat, labels[live], points0[live] - 1, dealers[live])
        value[live] = self.values[table][key]
        samples[:, live] = self.bootstrap[table][(slice(None), *key)]
        return value, samples


class RoundValueSearch(HistoryConditionedSearch):
    """Keep strict parent/history behavior, replacing complete-match rollout only.

    Table A supplies proposals. Only the selected candidate can be confirmed,
    using table B and fresh particles/uncertainty resampling. No table selection.
    """

    def __init__(self, checkpoint, tables, **kwargs):
        if kwargs.get("confirmation_particles", 32) < 2:
            raise ValueError("Round-value search requires fresh confirmation")
        super().__init__(checkpoint, **kwargs)
        if tables.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError("Tables belong to a different continuation parent")
        self.tables = tables
        self.workspaces = {
            count: RoundWorkspace(2 * count, self.device) for count in self.workspaces
        }
        self.work, self.confirmation_work = RoundWork(), RoundWork()

    def reset_seed(self, seed):
        super().reset_seed(seed)
        self.bootstrap_rngs = [
            np.random.default_rng(np.random.SeedSequence([seed, i])) for i in (2, 3)
        ]

    def _joint_lower(self, mean, joint_samples):
        return float(mean - self.confidence * joint_samples.std(ddof=1))

    @torch.no_grad()
    def _compare_actions(
        self, reference, base, alternatives, count, rng, confirmation=False
    ):
        if self.mode == "history":
            if (
                self.support is None
                or self.support.model_names != self.continuation.names
            ):
                raise ValueError("Enumerate the matching model support first")
            worlds, labels = sample_joint_worlds(reference, self.support, rng, count)
        else:
            worlds = sample_hidden_batch(reference, rng, count)
            labels = rng.integers(0, 2, count, dtype=np.int32)
        table = int(confirmation)
        bootstrap_seed = int(self.bootstrap_rngs[table].integers(0, 2**31 - 1))
        weights = np.random.default_rng(bootstrap_seed).multinomial(
            count, np.full(count, 1 / count), size=self.tables.bootstrap.shape[1]
        )
        self.last_sampling["confirmation" if confirmation else "proposal"] = {
            "worlds_sha256": hashlib.sha256(worlds.tobytes()).hexdigest(),
            "model_indices": labels.tolist(),
            "bootstrap_seed": bootstrap_seed,
            "table": ("A", "B")[table],
            "tables_identity": self.tables.identity,
        }
        seat = int(reference.state[PLAYER])
        self.continuation.prepare(seat, labels)
        states = torch.from_numpy(worlds).to(self.device).repeat_interleave(2, dim=0)
        records, previous_base = [], None
        for alternative in alternatives:
            actions = torch.tensor([base, alternative], device=self.device).repeat(
                count
            )
            endpoints, used = self.workspaces[count].run(
                states, self.continuation, actions
            )
            for name, value in asdict(used).items():
                setattr(self.work, name, getattr(self.work, name) + value)
                if confirmation:
                    setattr(
                        self.confirmation_work,
                        name,
                        getattr(self.confirmation_work, name) + value,
                    )
            end = endpoints.cpu().numpy()
            fields = end[:, [ROUND, POINTS, DEALER, PHASE]].reshape(count, 2, 4)
            if previous_base is not None:
                np.testing.assert_array_equal(previous_base, fields[:, 0])
            previous_base = fields[:, 0].copy()
            value, boot = self.tables.lookup(
                end[:, ROUND],
                end[:, POINTS],
                end[:, DEALER],
                end[:, PHASE] == TERMINAL,
                seat,
                labels.repeat(2),
                table,
            )
            changes = np.diff(value.reshape(count, 2), axis=1).ravel() / 2
            bootstrap_changes = (
                np.diff(boot.reshape(len(boot), count, 2), axis=2)[:, :, 0] / 2
            )
            joint_samples = (
                np.einsum("bn,bn->b", bootstrap_changes, weights, optimize=True) / count
            )
            records.append(
                {
                    "alternative": int(alternative),
                    "win_change": float(changes.mean()),
                    "lower_margin": self._joint_lower(changes.mean(), joint_samples),
                    "particle_se": float(changes.std(ddof=1) / np.sqrt(count)),
                    "table_se": float(bootstrap_changes.mean(1).std(ddof=1)),
                    "joint_se": float(joint_samples.std(ddof=1)),
                    "joint_bootstrap_means": joint_samples.tolist(),
                    "paired_boundary_values": value.reshape(count, 2).tolist(),
                    "paired_boundary_fields": fields.tolist(),
                    "boundary_field_order": ["round", "points0", "dealer", "phase"],
                    "work": asdict(used),
                    "table": ("A", "B")[table],
                }
            )
        return records
