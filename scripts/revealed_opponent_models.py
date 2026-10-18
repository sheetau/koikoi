"""Persistent finite-model elimination using delayed public hand revelations.

Assumptions: fixed deterministic stateless candidate policies and exact public
input reconstruction. A singleton identifies a member only within that declared
class. Empty support is explicit model mismatch, never a guessed identity.
"""

import hashlib

import numpy as np
import torch

from scripts.revealed_opponent_inputs import PublicActorSnapshot, reconstruct_revealed


class StrictPublishedPrediction:
    """Reuse the pinned upstream CPU model on one reconstructed input at a time."""

    def __init__(self, published_agent):
        self.models = published_agent.models
        if any(model.training for model in self.models.values()):
            raise ValueError("Predictions require deterministic evaluation mode")

    @torch.no_grad()
    def __call__(self, phase, features, legal):
        width = 50 if phase == 3 else 48
        tensor = torch.from_numpy(np.ascontiguousarray(features[:, :width])).unsqueeze(
            0
        )
        logits = self.models[phase](tensor).squeeze(0)
        mask = torch.from_numpy(legal[48:] if phase == 3 else legal[:48])
        index = int(logits.masked_fill(~mask, -torch.inf).argmax())
        return index + 48 if phase == 3 else index


class RevealedModelSet:
    def __init__(self, match_key, viewer, models):
        if not isinstance(match_key, str) or not match_key or viewer not in (0, 1):
            raise ValueError("Expected public match identity and viewer seat")
        if not models or any(not isinstance(name, str) or not name for name in models):
            raise ValueError("Supply named deterministic model predictors")
        self.match_key, self.viewer = match_key, viewer
        self.models = dict(models)
        self.candidates = set(models)
        self.round = -1
        self.last_decision = -1
        self.discards = set()
        self.buffer = []
        self.round_revealed = False
        self.first_identification = None
        self.events = []
        self.prediction_calls = {name: 0 for name in models}

    @property
    def status(self):
        return (
            "inconsistent"
            if not self.candidates
            else "identified"
            if len(self.candidates) == 1
            else "ambiguous"
        )

    def observe(self, snapshot, observed_action):
        """Call only AFTER this opponent action is publicly observed.

        The decision address is the pre-action global index. Snapshots of the
        learner's actions are not needed: their public effects are already in
        each stored public board/history. New rounds clear the pending history,
        while candidate exclusions persist for this match.
        """
        x = np.asarray(snapshot.features)
        if snapshot.match_key != self.match_key:
            raise ValueError("Snapshot belongs to a different match")
        if x.shape != (562,) or not np.isfinite(x).all() or x[561] != 1 - self.viewer:
            raise ValueError("Expected this opponent public snapshot")
        if np.any(x[:528].reshape(48, 11)[:, [0, 4]]):
            raise ValueError("Private hand channels must be erased")
        if snapshot.decision <= self.last_decision:
            raise ValueError("Observed decisions must increase")
        current_round = round(float(x[534]) * 12)
        if current_round < self.round:
            raise ValueError("Observed rounds must not go backwards")
        if type(observed_action) is not int or observed_action not in range(50):
            raise ValueError("Invalid observed action")
        if current_round != self.round:
            self.round = current_round
            self.discards = set()
            self.buffer = []
            self.round_revealed = False
        self.last_decision = snapshot.decision
        if self.round_revealed:
            return None
        self.buffer.append(
            (
                PublicActorSnapshot(x.copy(), snapshot.decision, snapshot.match_key),
                observed_action,
            )
        )
        phase = int(np.argmax(x[528:532]))
        if phase == 0:
            if observed_action >= 48 or observed_action in self.discards:
                raise ValueError("Invalid or repeated hand discard")
            self.discards.add(observed_action)
        if len(self.discards) < 8:
            return None
        reveal = {
            "round": self.round,
            "match_key": self.match_key,
            "actor_seat": 1 - self.viewer,
            "available_after_decision": snapshot.decision + 1,
            "initial_opponent_hand": sorted(self.discards),
        }
        before = sorted(self.candidates)
        checks = []
        # Earlier mismatches eliminate a deterministic model; later roots still
        # reconstruct for the evidence ledger even if all models are rejected.
        for old, observed in self.buffer:
            features, legal, assisted = reconstruct_revealed(
                old, reveal, snapshot.decision + 1
            )
            if not legal[observed]:
                raise ValueError("Observed history contradicts the revealed hand")
            phase = int(np.argmax(old.features[528:532]))
            predictions = {}
            for name in sorted(self.candidates):
                predicted = int(self.models[name](phase, features, assisted))
                if predicted not in range(50) or not assisted[predicted]:
                    raise ValueError("Candidate predictor returned an illegal action")
                self.prediction_calls[name] += 1
                predictions[name] = predicted
                if predicted != observed:
                    self.candidates.remove(name)
            checks.append(
                {
                    "decision": old.decision,
                    "phase": phase,
                    "observed": observed,
                    "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                    "assisted_mask_sha256": hashlib.sha256(
                        assisted.tobytes()
                    ).hexdigest(),
                    "predictions": predictions,
                }
            )
        event = {
            **reveal,
            "before": before,
            "after": sorted(self.candidates),
            "checks": checks,
        }
        self.events.append(event)
        self.round_revealed = True
        self.buffer = []
        if len(self.candidates) == 1 and self.first_identification is None:
            self.first_identification = {
                "round": self.round,
                "available_after_decision": snapshot.decision + 1,
                "model": next(iter(self.candidates)),
            }
        return event
