"""Complete, identity-bound learned continuations for public frontier patches."""

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType

from koikoi.layout import PLAYER
from scripts.belief_tail import visible_keys
from scripts.diagnose_history_support import reference
from scripts.frontier_policy_patch import (
    FrontierExecutor,
    FrontierPatch,
    parent_frontier,
)
from scripts.incumbent_tail_controller import TailBinding
from scripts.interval_tail import (
    IntervalEvaluator,
    IntervalExecutor,
    PublicProgram,
    make_program,
    select_interval,
)
from scripts.verify_frontier_patch import verify_frontier_patch
from scripts.verify_interval_tail import verify_receipt

RULE = "legal-public-overrides-named-continuation-v1"


class DispatchRequired(dict):
    """Legacy consumers must not silently interpret a named default as parent."""

    def get(self, *args, **kwargs):
        raise ValueError("Named continuation requires explicit actor dispatch")

    def __contains__(self, key):
        raise ValueError("Named continuation requires explicit actor dispatch")


@dataclass(frozen=True)
class ContinuationProgram:
    binding: TailBinding
    parent_root_action: int
    overrides: tuple[tuple[bytes, int], ...]
    continuation_identity: str
    rule: str = RULE

    def validate(self):
        if (
            self.rule != RULE
            or not isinstance(self.continuation_identity, str)
            or not self.continuation_identity
            or self.continuation_identity == self.binding.parent_identity
        ):
            raise ValueError("Malformed named continuation")
        ordinary = PublicProgram(self.binding, self.parent_root_action, self.overrides)
        return DispatchRequired(ordinary.validate())

    @property
    def is_parent(self):
        return False


class ContinuationActors:
    """Fixed observation-only actors; callers authenticate their frozen weights."""

    def __init__(self, parent_identity, parent, alternatives):
        if (
            not isinstance(parent_identity, str)
            or not parent_identity
            or parent_identity in alternatives
            or any(not isinstance(k, str) or not k for k in alternatives)
        ):
            raise ValueError("Distinct nonempty actor identities required")
        self.parent_identity, self.parent = parent_identity, parent
        self.alternatives = MappingProxyType(dict(alternatives))
        self.cache, self.calls = {}, Counter()

    def actor_for(self, program):
        program.validate()
        if program.binding.parent_identity != self.parent_identity:
            raise ValueError("Reference actor identity differs")
        if type(program) is PublicProgram:
            return self.parent
        if type(program) is not ContinuationProgram:
            raise ValueError("Unsupported program semantics")
        if program.continuation_identity not in self.alternatives:
            raise ValueError("Unknown continuation actor identity")
        return self.alternatives[program.continuation_identity]

    def action(self, identity, state):
        if identity not in self.alternatives:
            raise ValueError("Unknown continuation actor identity")
        key = visible_keys(state[None], int(state[PLAYER]))[0]
        address = identity, key
        if address not in self.cache:
            env = reference(state)
            legal = tuple(int(i) for i in env.legal().nonzero()[0])
            action = (
                legal[0]
                if len(legal) == 1
                else int(self.alternatives[identity].act(env))
            )
            if action not in legal:
                raise ValueError("Continuation actor chose an illegal action")
            self.cache[address] = action
            self.calls[identity] += int(len(legal) > 1)
        return self.cache[address]


class ContinuationDynamics:
    def __init__(self, dynamics, program, actors):
        actors.actor_for(program)
        if dynamics.baseline is not actors.parent:
            raise ValueError("Dynamics reference is not the registered parent")
        self.reference, self.program, self.actors = dynamics, program, actors

    def __getattr__(self, name):
        return getattr(self.reference, name)

    def baseline_action(self, state):
        key = visible_keys(state[None], self.program.binding.viewer)[0]
        if key == self.program.binding.root_key:
            # Constructor invariant only: actual execution uses the mandatory
            # legal explicit root entry, including if this key is revisited.
            return self.reference.baseline_action(state)
        return self.actors.action(self.program.continuation_identity, state)


class ContinuationEvaluator(IntervalEvaluator):
    def __init__(self, worlds, dynamics, program, actors, **kwargs):
        actors.actor_for(program)
        adapted = (
            ContinuationDynamics(dynamics, program, actors)
            if type(program) is ContinuationProgram
            else dynamics
        )
        if dynamics.baseline is not actors.parent:
            raise ValueError("Dynamics reference is not the registered parent")
        super().__init__(worlds, adapted, program, **kwargs)
        # Identity and reference checks above authorize the named dispatch.
        self.entries = dict(self.entries)


def compare_continuations(
    worlds,
    dynamics,
    actors,
    opponent_ids,
    *,
    max_decisions=100000,
    max_nodes=25000,
    quantum=4096,
):
    if any(type(v) is not int or v < 1 for v in (max_decisions, max_nodes, quantum)):
        raise ValueError("Positive comparison budgets required")
    original = make_program(worlds, dynamics, actors.parent_identity, opponent_ids)
    programs = [original]
    programs.extend(
        make_program(
            worlds,
            dynamics,
            actors.parent_identity,
            opponent_ids,
            {original.binding.root_key: action},
        )
        for action in original.binding.legal
        if action != original.parent_root_action
    )
    for identity in actors.alternatives:
        program = ContinuationProgram(
            original.binding,
            original.parent_root_action,
            ((original.binding.root_key, actors.action(identity, worlds.states[0])),),
            identity,
        )
        program.validate()
        programs.append(program)
    evaluators = [ContinuationEvaluator(worlds, dynamics, p, actors) for p in programs]
    receipts = [e.receipt() for e in evaluators]
    used, blocked = 0, set()
    while used < max_decisions:
        parent = receipts[0]
        if select_interval(parent, receipts[1:]).candidate is not None:
            break
        active = [
            i for i in range(1, len(receipts)) if receipts[i].upper > parent.lower
        ]
        if not active:
            break
        options = [
            i
            for i in [0, *active]
            if i not in blocked and receipts[i].upper > receipts[i].lower
        ]
        if not options:
            break
        index = max(options, key=lambda i: receipts[i].upper - receipts[i].lower)
        evaluator = evaluators[index]
        previous = evaluator.work["decisions"]
        allowance = min(quantum, max_decisions - used)
        receipt = evaluator.run(previous + allowance, max_nodes)
        spent = evaluator.work["decisions"] - previous
        used += spent
        # Other evaluators have not changed. Reusing their checked immutable
        # receipts avoids rescanning their pending mass after every quantum.
        receipts[index] = receipt
        if receipt.refusal not in (None, "Physical transition cap reached"):
            blocked.add(index)
        elif not spent:
            if allowance == max_decisions - used:
                blocked.add(index)
            else:
                quantum *= 2
    if used != sum(r.work["decisions"] for r in receipts) or used > max_decisions:
        raise ValueError("Physical work differs from the shared comparison allowance")
    return {
        "selection": select_interval(receipts[0], receipts[1:]),
        "parent_receipt": receipts[0],
        "candidate_receipts": tuple(receipts[1:]),
        "decisions": used,
        "evaluation_work": [r.work for r in receipts],
    }


def solve_continuation_frontier(
    worlds,
    dynamics,
    actors,
    opponent_ids,
    *,
    draws=3,
    max_decisions=2000000,
    prefix_decisions=250000,
    local_decisions=100000,
    max_nodes_per_phase=25000,
    max_rows=65536,
):
    if dynamics.baseline is not actors.parent:
        raise ValueError("Dynamics reference is not the registered parent")
    frontier = parent_frontier(
        worlds,
        dynamics,
        actors.parent_identity,
        opponent_ids,
        draws=draws,
        max_decisions=min(max_decisions, prefix_decisions),
        max_rows=max_rows,
    )
    used, comparisons, gain = frontier.work["decisions"], [], Fraction(0)
    if frontier.complete:
        for key, node in frontier.nodes:
            allowance = min(local_decisions, max_decisions - used)
            if allowance < 1:
                break
            result = compare_continuations(
                node,
                dynamics,
                actors,
                opponent_ids,
                max_decisions=allowance,
                max_nodes=max_nodes_per_phase,
            )
            used += result["decisions"]
            comparisons.append((key, result))
            gain += (
                Fraction(sum(map(int, node.weights)), frontier.total_mass)
                * result["selection"].gain_lower
            )
    if used > max_decisions:
        raise ValueError("Global transition allowance exceeded")
    return FrontierPatch(frontier, tuple(comparisons), gain, used)


class ContinuationFrontierExecutor:
    def __init__(self, patch, actors):
        actors.actor_for(patch.frontier.parent_program)
        self.engine = FrontierExecutor(patch, actors.parent, actors.parent_identity)
        for key, result in patch.comparisons:
            selection = result["selection"]
            if selection.candidate is not None:
                fallback = actors.actor_for(selection.candidate.program)
                local = IntervalExecutor(selection, fallback, actors.parent_identity)
                local.entries = dict(local.entries)
                self.engine.patches[key] = local

    def act(self, env):
        return self.engine.act(env)


class ReplayContinuation:
    """Independent Python oracle actor; no planning action cache is consulted."""

    def __init__(self, program, actors):
        self.program, self.parent = program, actors.parent
        self.continuation = actors.actor_for(program)
        self.records = {}

    def act(self, env):
        binding = self.program.binding
        if int(env.state[PLAYER]) != binding.viewer:
            raise ValueError("Continuation replay received the wrong player's input")
        key = visible_keys(env.state[None], binding.viewer)[0]
        if key == binding.root_key:
            return int(self.parent.act(env))
        action = int(self.continuation.act(env))
        if key in self.records and self.records[key] != action:
            raise ValueError("Continuation is not deterministic on own/public history")
        self.records[key] = action
        return action


def verify_continuation_receipt(worlds, receipt, actors, opponents):
    actors.actor_for(receipt.program)
    program = receipt.program
    if type(program) is PublicProgram:
        report = verify_receipt(
            worlds, receipt, actors.parent, opponents, actors.parent_identity
        )
        return report, receipt
    replay = ReplayContinuation(program, actors)
    # Internal oracle projection only: the supplied actor implements the
    # declared fallback. It is never saved as a parent-reference certificate.
    projected = replace(
        receipt,
        program=PublicProgram(
            program.binding, program.parent_root_action, program.overrides
        ),
    )
    report = verify_receipt(
        worlds, projected, replay, opponents, actors.parent_identity
    )
    report["continuation_identity"] = program.continuation_identity
    report["actor_calls"]["continuation"] = report["actor_calls"].pop("parent", 0)
    entries = dict(program.overrides)
    # A reached illegal override falls back to the named actor. Preserve that
    # actual action rather than copying the illegal entry into a full map.
    entries.update(replay.records)
    explicit = replace(
        projected,
        program=PublicProgram(
            program.binding, program.parent_root_action, tuple(sorted(entries.items()))
        ),
    )
    return report, explicit


def verify_continuation_patch(worlds, patch, actors, opponents, *, full_values=False):
    ContinuationFrontierExecutor(patch, actors)
    prefix_only = FrontierPatch(
        patch.frontier, (), Fraction(0), patch.frontier.work["decisions"]
    )
    report = verify_frontier_patch(
        worlds, prefix_only, actors.parent, opponents, actors.parent_identity
    )
    nodes, exact, projected = dict(patch.frontier.nodes), Fraction(0), []
    for key, result in patch.comparisons:
        selection = result["selection"]
        if selection.candidate is None:
            projected.append((key, result))
            continue
        node = nodes[key]
        parent_value, _ = verify_continuation_receipt(
            node, selection.parent, actors, opponents
        )
        candidate_value, explicit = verify_continuation_receipt(
            node, selection.candidate, actors, opponents
        )
        gain = Fraction(candidate_value["value"]) - Fraction(parent_value["value"])
        if not gain >= selection.gain_lower > 0:
            raise ValueError("Independent named-continuation gain fails")
        reach = Fraction(sum(map(int, node.weights)), patch.frontier.total_mass)
        exact += reach * gain
        report["local_checks"].append(
            {
                "reach": str(reach),
                "gain": str(gain),
                "oracle": {"parent": parent_value, "candidate": candidate_value},
            }
        )
        projected.append(
            (key, {**result, "selection": replace(selection, candidate=explicit)})
        )
    if exact < patch.gain_lower:
        raise ValueError("Weighted independent gain is below the certificate")
    report.update(exact_increment=str(exact), gain_lower=str(patch.gain_lower))
    if full_values:
        explicit_patch = FrontierPatch(
            patch.frontier, tuple(projected), patch.gain_lower, patch.decisions
        )
        full = verify_frontier_patch(
            worlds,
            explicit_patch,
            actors.parent,
            opponents,
            actors.parent_identity,
            full_values=True,
        )
        if Fraction(full["exact_increment"]) != exact:
            raise ValueError("Explicit continuation projection changes the value")
        report["full_values"] = full["full_values"]
    return report
