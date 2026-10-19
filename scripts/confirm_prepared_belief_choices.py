"""Fresh confirmation of fixed positive-mean development candidates."""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.upstream import PublishedAgent
from scripts.prepared_belief_search import KINDS, PreparedBeliefSearch
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.run_prepared_belief_actions import complete, load_roots

SOURCE = Path("runs/prepared_belief_actions")


def run(output):
    if output.exists():
        raise ValueError("Preserve existing outputs")
    audit = read(SOURCE / "audit.json")
    assert audit["status"] == "passed"
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA execution required")
    inputs = dict(audit["inputs"])
    for path in (
        Path(__file__),
        Path("docs/PREPARED_BELIEF_CONFIRMATION_CHECK.rst"),
        SOURCE / "audit.json",
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    choices, prior_outcomes = (
        read(SOURCE / "choices.json"),
        read(SOURCE / "outcomes.json"),
    )
    assert len(choices) == len(prior_outcomes) == 25
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "root_seed": "103100001+10000*parent+case_index",
            "confirmation_particles": 512,
            "confidence": 2.5,
            "selection": "Previously fixed alternative; confirm iff proposal mean > 0",
            "scope": "Follow-up development diagnostic, no full-match strength criterion",
        },
    )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agents = {
        p: PreparedBeliefSearch(
            path, particles=2, confirmation_particles=512, device="cuda"
        )
        for p, (path, _) in enumerate(PARENTS, 1)
    }
    results = []
    try:
        for (row, env, batch), choice in zip(load_roots(), choices):
            agent = agents[row["parent"]]
            assert (row["parent"], row["case_index"]) == (
                choice["parent"],
                choice["case_index"],
            )
            base = int(agent.parent.act(env))
            assert base == choice["base"]
            conditions = {}
            for kind in KINDS:
                previous = choice["conditions"][kind]["decision"]
                alternative = previous["alternative"]
                proposed = previous["proposal"]["win_change"] > 0
                confirmation = None
                sampling = None
                start = time.perf_counter()
                if proposed:
                    agent.prepare(
                        env,
                        batch,
                        kind,
                        row["public_identification"],
                        row["decision"],
                        row["seat"],
                    )
                    confirmation = agent._compare_actions(
                        env,
                        base,
                        [alternative],
                        512,
                        np.random.default_rng(
                            103100001 + 10000 * row["parent"] + row["case_index"]
                        ),
                        confirmation=True,
                    )[0]
                    sampling = agent.last_sampling["confirmation"]
                accepted = proposed and confirmation["lower_margin"] > 0
                conditions[kind] = {
                    "action": alternative if accepted else base,
                    "confirmed": proposed,
                    "confirmation": confirmation,
                    "sampling": sampling,
                    "seconds": time.perf_counter() - start,
                }
            result = {
                key: choice[key]
                for key in (
                    "parent",
                    "case_index",
                    "opponent",
                    "seed",
                    "decision",
                    "seat",
                    "base",
                )
            }
            result["conditions"] = conditions
            results.append(result)
            write(output / f"choice_{row['parent']}_{row['case_index']}.json", result)
            write(
                output / "process.json",
                {"status": "confirming", "completed_roots": len(results)},
            )
            print(len(results), flush=True)
        write(output / "choices.json", results)
        frozen = sha(output / "choices.json")
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("point", "wp")
        }
        outcomes, reused = [], 0
        work = dict.fromkeys(
            ("decisions", "round_endings", "conditional_completions"), 0
        )
        for (row, env, _), result, prior in zip(load_roots(), results, prior_outcomes):
            selected = {result["base"]} | {
                c["action"] for c in result["conditions"].values()
            }
            completions, new_actions = {}, []
            for action in sorted(selected):
                if str(action) in prior["completions"]:
                    completion = prior["completions"][str(action)]
                    reused += 1
                else:
                    completion = complete(
                        env,
                        action,
                        agents[row["parent"]].parent,
                        opponents[row["opponent"]],
                        row["seat"],
                    )
                    new_actions.append(action)
                    for key, value in completion["work"].items():
                        work[key] += value
                completions[str(action)] = completion
            outcomes.append(
                {
                    "parent": row["parent"],
                    "case_index": row["case_index"],
                    "completions": completions,
                    "new_actions": new_actions,
                }
            )
        assert sha(output / "choices.json") == frozen
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(output / "outcomes.json", outcomes)
        write(
            output / "report.json",
            {
                "status": "complete",
                "inputs": inputs,
                "protocol_sha256": sha(output / "protocol.json"),
                "choices_sha256": frozen,
                "outcomes_sha256": sha(output / "outcomes.json"),
                "source_prefix_replay_decisions": 2
                * sum(c["decision"] for c in results),
                "search_work": {str(p): asdict(a.work) for p, a in agents.items()},
                "new_actual_completion_work": work,
                "reused_completions": reused,
                "new_historical_neural_calls": 0,
                "new_evaluation_deals": 0,
                "full_evaluation_matches": 0,
            },
        )
        write(output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
