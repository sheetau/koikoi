"""Checkpointable training-root selection with a fresh chance key per draw."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.continuation_chance import PublicContinuationContext, integer, renew_deal
from scripts.final_round_training import bank_identity
from scripts.public_memory_continuation import validate_continuation_bank
from scripts.public_memory_tail import memory_identity


class RenewedRootSampler:
    """Private simulator restart data; never pass this object to a policy.

    Original source multiplicities,learner seats,routing labels and public-memory
    codes are retained. Only deal/future chance is renewed. The caller must use
    an audited bank and allocate a disjoint chance-key range in its protocol.
    """

    def __init__(
        self, states, seats, opponents, memory, *, sampling_seed, chance_seed, max_draws
    ):
        sampling_seed = integer(sampling_seed, "sampling_seed")
        chance_seed = integer(chance_seed, "chance_seed")
        max_draws = integer(max_draws, "max_draws")
        if not 0 <= sampling_seed < 2**63:
            raise ValueError("Invalid sampling seed")
        if chance_seed < 0 or max_draws < 1 or chance_seed + max_draws > 2**31:
            raise ValueError("Invalid finite chance-key allocation")
        validate_continuation_bank(states, seats, opponents, memory)
        bank = tuple(t.detach().cpu().clone() for t in (states, seats, opponents))
        self.contexts = tuple(
            PublicContinuationContext.from_root(row) for row in bank[0].numpy()
        )
        self.seats, self.opponents = bank[1:]
        self.memory = memory.detach().cpu().clone()
        self.config = {
            "sampling_seed": sampling_seed,
            "chance_seed": chance_seed,
            "max_draws": max_draws,
        }
        self.identity = {
            "bank": bank_identity(*bank),
            "memory": memory_identity(self.memory),
            "sampler_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "deal_source": hashlib.sha256(
                Path("scripts/continuation_chance.py").read_bytes()
            ).hexdigest(),
        }
        self.generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
        self.cursor = 0
        self.batch_lengths = []

    def sample(self, count):
        count = integer(count, "count")
        if count < 1 or self.cursor + count > self.config["max_draws"]:
            raise ValueError("Chance-key allocation exhausted or invalid batch size")
        before = self.generator.get_state()
        try:
            indices = torch.randint(
                len(self.contexts), (count,), generator=self.generator
            )
            start = self.config["chance_seed"] + self.cursor
            seeds = torch.arange(start, start + count, dtype=torch.int64)
            states = torch.from_numpy(
                np.stack(
                    [
                        renew_deal(self.contexts[int(i)], int(seed))
                        for i, seed in zip(indices, seeds)
                    ]
                )
            )
        except BaseException:
            self.generator.set_state(before)
            raise
        self.cursor += count
        self.batch_lengths.append(count)
        return {
            "states": states,
            "seats": self.seats[indices].clone(),
            "opponents": self.opponents[indices].clone(),
            "memory": self.memory[indices].clone(),
            "indices": indices,
            "chance_seeds": seeds,
        }

    def state_dict(self):
        return {
            "format": "public_context_chance_sampler_v1",
            "config": dict(self.config),
            "identity": dict(self.identity),
            "cursor": self.cursor,
            "batch_lengths": list(self.batch_lengths),
            "rng": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state):
        if state.get("format") != "public_context_chance_sampler_v1":
            raise ValueError("Wrong chance-sampler format")
        if state["config"] != self.config or state["identity"] != self.identity:
            raise ValueError("Chance sampler source,bank or allocation mismatch")
        cursor = integer(state["cursor"], "cursor")
        lengths = [integer(n, "batch length") for n in state["batch_lengths"]]
        if (
            any(n < 1 for n in lengths)
            or sum(lengths) != cursor
            or not 0 <= cursor <= self.config["max_draws"]
        ):
            raise ValueError("Invalid chance-sampler accounting")
        # Replay only index draws,not deals/trajectories,to prevent a cursor/RNG
        # mismatch from silently reusing keys or changing subsequent root draws.
        restored = torch.Generator(device="cpu").manual_seed(
            self.config["sampling_seed"]
        )
        for count in lengths:
            torch.randint(len(self.contexts), (count,), generator=restored)
        if not torch.equal(restored.get_state(), state["rng"]):
            raise ValueError("Chance-sampler RNG disagrees with draw accounting")
        self.generator = restored
        self.cursor = cursor
        self.batch_lengths = copy.copy(lengths)
