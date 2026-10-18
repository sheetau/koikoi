"""Experimental all-action terminal search for the retained Transformer actors.

Every alternative uses the same observation-consistent proposal worlds. A fresh
sample can confirm only the selected alternative. This is Monte Carlo rollout
policy improvement, not a new search algorithm or a strength guarantee.
"""

from dataclasses import asdict

import numpy as np
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import BANKRUPT, PHASE, ROUNDS, TERMINAL
from koikoi.search import sample_hidden_batch
from scripts.published_koi_search import PublishedKoiSearch


class PublishedActionSearch(PublishedKoiSearch):
    """Search every unforced action; retain strict CPU fallback and public guard."""

    @torch.no_grad()
    def _compare_actions(
        self, reference, base, alternatives, count, rng, confirmation=False
    ):
        worlds = sample_hidden_batch(reference, rng, count)
        states = torch.from_numpy(worlds).to(self.device)
        records = []
        previous_base = None
        for alternative in alternatives:
            proposals = torch.tensor([base, alternative], device=self.device).expand(
                count, 2
            )
            wins, points, work = self.workspaces[count](
                states, proposals, self.continuation
            )
            for key, value in asdict(work).items():
                setattr(self.work, key, getattr(self.work, key) + value)
                if confirmation:
                    setattr(
                        self.confirmation_work,
                        key,
                        getattr(self.confirmation_work, key) + value,
                    )
            point_rows = points.cpu().numpy().copy()
            if previous_base is not None:
                np.testing.assert_array_equal(previous_base, point_rows[:, 0])
            previous_base = point_rows[:, 0].copy()
            differences = (wins[:, 1] - wins[:, 0]).double().cpu().numpy() / 2
            records.append(
                {
                    "alternative": int(alternative),
                    "win_change": float(differences.mean()),
                    "lower_margin": self._lower(differences),
                    "point_change": float(np.diff(point_rows, axis=1).mean()),
                    "paired_points": point_rows.tolist(),
                    "work": asdict(work),
                }
            )
        return records

    def act(self, reference):
        if int(reference.state[ROUNDS]) != 8 or not reference.state[BANKRUPT]:
            raise ValueError("Search supports paper8 only")
        if reference.state[PHASE] == TERMINAL:
            raise ValueError("Search requires an active root")
        self.last_decision = None
        solved = solved_koi_action(reference)
        if solved is not None:
            self.solved_decisions += 1
            return solved
        base = int(self.parent.act(reference))
        legal = np.flatnonzero(reference.legal()).tolist()
        if base not in legal:
            raise ValueError("Fallback must select a legal action")
        if len(legal) == 1:
            return base
        alternatives = [action for action in legal if action != base]
        records = self._compare_actions(
            reference, base, alternatives, self.particles, self.rng
        )
        # Legal-action order breaks exact ties. Do not rank by confirmation.
        selected = max(records, key=lambda row: row["win_change"])
        alternative = selected["alternative"]
        self.searched_decisions += 1
        accept = selected["lower_margin"] > 0
        self.proposed_decisions += int(accept)
        confirmation = None
        if accept and self.confirmation_particles:
            confirmation = self._compare_actions(
                reference,
                base,
                [alternative],
                self.confirmation_particles,
                self.confirmation_rng,
                confirmation=True,
            )[0]
            self.confirmation_decisions += 1
            accept = confirmation["lower_margin"] > 0
        self.changed_decisions += int(accept)
        self.last_decision = {
            "base": base,
            "alternative": alternative,
            "alternatives": records,
            "proposal": selected,
            "confirmation": confirmation,
            "changed": bool(accept),
        }
        return alternative if accept else base
