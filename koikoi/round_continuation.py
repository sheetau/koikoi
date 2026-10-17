"""Frozen public round-start continuation targets for explicit PPO experiments."""

import hashlib
import json
import math

import torch


def validate_spec(spec):
    """The checkpoint embeds coefficients and the exact calibration identities."""
    if not isinstance(spec, dict) or set(spec) != {"coefficients", "source"}:
        raise ValueError("Round continuation requires coefficients and source")
    weights = spec["coefficients"]
    if not isinstance(weights, list) or len(weights) != 8:
        raise ValueError("Round continuation requires eight coefficient pairs")
    for row in weights:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or any(
                type(x) not in (int, float)
                or not math.isfinite(x)
                or abs(x) > torch.finfo(torch.float32).max
                for x in row
            )
        ):
            raise ValueError("Round continuation coefficients must be finite pairs")
    source = spec["source"]
    if not isinstance(source, dict) or set(source) != {
        "parent_sha256",
        "report_sha256",
        "data_sha256",
        "audit_sha256",
    }:
        raise ValueError("Round continuation source identities missing")
    for digest in source.values():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Round continuation identities must be SHA256 digests")


def signature(spec):
    validate_spec(spec)
    return hashlib.sha256(
        json.dumps(
            spec, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class RoundContinuation:
    def __init__(self, spec, device):
        self.sha256 = signature(spec)
        self.parent_sha256 = spec["source"]["parent_sha256"]
        self._coefficients = torch.tensor(
            spec["coefficients"], dtype=torch.float32, device=device
        )

    def check(self, spec):
        if signature(spec) != self.sha256:
            raise ValueError("Round continuation specification changed")

    def value(self, round_index, points0, dealer):
        """Seat-zero expected outcome; only public scalar inputs are accepted.

        Terminal rows may carry round index8; their result is masked by the caller
        in favor of the actual outcome. No cards, RNG, or network values enter.
        """
        weights = self._coefficients[round_index.long().clamp(0, 7)]
        logit = weights[:, 0] * ((points0.float() - 30) / 30)
        logit += weights[:, 1] * (1 - 2 * dealer)
        return 2 * torch.sigmoid(logit) - 1
