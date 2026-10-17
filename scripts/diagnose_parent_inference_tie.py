"""Trace a completed development mismatch without changing models or rules."""

import argparse

import torch

from koikoi.agents import NeuralAgent, solved_koi_action
from koikoi.arbitration import ArbitrationAgent
from koikoi.evaluate import make_agent
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from scripts.run_arbitration_study import PARENTS, write


def diagnose(seed=21000447, seat=1):
    torch.set_num_threads(2)
    gate = ArbitrationAgent("runs/arbitration_study/parent1_phase/final.pt")
    parent = NeuralAgent(PARENTS[0][0], deterministic=True)
    opponent = make_agent("rl-point-assisted", 0, True)
    env = ReferenceEnv(seed, "paper8")
    decision = 0
    while env.state[PHASE] != TERMINAL:
        if env.state[PLAYER] != seat:
            action = opponent.act(env)
        else:
            controlled, original = gate.act(env), parent.act(env)
            if controlled != original:
                features, legal = gate.env.published_observe()
                packed, parent_legal = packed_observation(parent.published_env)
                frozen = gate.bank.experts[0]
                source = parent.model.actor
                report = {
                    "seed": seed,
                    "candidate_seat": seat,
                    "decision": decision,
                    "phase": int(env.state[PHASE]),
                    "round": int(env.state[ROUND]),
                    "points": env.state[POINTS : POINTS + 2].tolist(),
                    "gate_action": controlled,
                    "parent_action": original,
                    "states_equal": torch.equal(
                        gate.env.state, parent.published_env.state
                    ),
                    "features_equal": torch.equal(
                        features, packed[:, 561:].reshape(-1, 300, 50)
                    ),
                    "masks_equal": torch.equal(legal, parent_legal),
                    "weights_equal": all(
                        torch.equal(a, b)
                        for a, b in zip(
                            frozen.state_dict().values(),
                            source.state_dict().values(),
                            strict=True,
                        )
                    ),
                    "native_guard": gate.env.solved_actions().tolist(),
                    "reference_guard": solved_koi_action(env),
                    "controller_sha256": gate.checkpoint_sha256,
                    "parent_sha256": parent.checkpoint_sha256,
                    "raw_scores": {},
                }
                actions = torch.where(legal[0])[0]
                report["legal_actions"] = actions.tolist()
                # Factorial intervention is process-local: no checkpoint is saved.
                with torch.no_grad():
                    for name, actor in (("frozen", frozen), ("parent", source)):
                        initial = next(actor.parameters()).requires_grad
                        for gradient_flag in (False, True):
                            actor.requires_grad_(gradient_flag)
                            score = actor(features, gate.env.state[:, PHASE], legal)[
                                0, actions
                            ]
                            repeat = actor(features, gate.env.state[:, PHASE], legal)[
                                0, actions
                            ]
                            report["raw_scores"][
                                f"{name}_requires_grad_{gradient_flag}"
                            ] = {
                                "scores": score.tolist(),
                                "repeat_exact": torch.equal(score, repeat),
                                "greedy": int(actions[score.argmax()]),
                                "scaled_greedy": int(
                                    actions[(score / parent.model.temperature).argmax()]
                                ),
                            }
                        actor.requires_grad_(initial)
                report["scope"] = (
                    "One traced near-tie. Requires-grad interventions affect only diagnostic in-memory copies; training and deployment checkpoints are unchanged. Neither feature nor game-rule equivalence implies identical floating-point action selection for every inference path."
                )
                return report
            action = original
        env.step(action)
        decision += 1
    raise ValueError("No action mismatch on the requested reference trajectory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=21000447)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    report = diagnose(args.seed, args.seat)
    write(
        "runs/arbitration_diagnostic_parent1/parent_inference_path_difference.json",
        report,
    )
    print(report)


if __name__ == "__main__":
    main()
