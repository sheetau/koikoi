"""CPU-only identity and numerical audit of all frozen rule adaptations."""

import argparse
from pathlib import Path

import torch

from scripts.export_public_stock_results import read, sha, write


def identical(actual, expected):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            raise ValueError("Frozen parent tensor differs")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Frozen parent fields differ")
        for key in expected:
            identical(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError("Frozen parent sequence differs")
        for left, right in zip(actual, expected, strict=True):
            identical(left, right)
    elif actual != expected:
        raise ValueError("Frozen parent metadata differs")


def audit(base):
    torch.set_num_threads(1)
    protocol = read(base / "protocol.json")
    bindings = {str(base / "protocol.json"): sha(base / "protocol.json")}
    parents = []
    for path, digest in protocol["parents"]:
        if sha(path) != digest:
            raise ValueError("Original parent file changed")
        bindings[path] = digest
        parents.append(torch.load(path, map_location="cpu", weights_only=False))
    results = {}
    for profile in protocol["profiles"]:
        for parent_id, parent in enumerate(parents):
            label = f"{profile}_parent{parent_id}"
            folder = base / label
            checkpoint = folder / f"policy_{protocol['updates']:05d}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = saved["config"]
            if (
                saved["format"] != "public_stock_residual_v1"
                or saved["parent_sha256"] != protocol["parents"][parent_id][1]
                or config["profile"] != profile
                or config["parent"] != parent_id
                or config["seed"] != protocol["training_seeds"][parent_id]
                or config["deal_seed"] != protocol["training_deal_seeds"][parent_id]
                or saved["counters"]["decisions"] != protocol["decisions_per_model"]
                or saved["counters"]["updates"] != protocol["updates"]
            ):
                raise ValueError(f"Wrong adaptation identity or budget: {label}")
            identical(config, read(folder / "config.json")["config"])
            for field in ("actor", "value", "config"):
                identical(saved["parent"][field], parent[field])
            for path, digest in saved["bindings"].items():
                current = bindings.get(path)
                if current is None:
                    current = sha(path)
                    bindings[path] = current
                if current != digest:
                    raise ValueError(f"Training input changed: {path}")
            width = config["width"]
            shapes = {
                "body.0.weight": (width, 660),
                "body.0.bias": (width,),
                "body.2.weight": (width, width),
                "body.2.bias": (width,),
                "actor.weight": (50, width),
                "actor.bias": (50,),
                "critic.weight": (1, width),
                "critic.bias": (1,),
            }
            state = saved["model"]
            if state.keys() != shapes.keys():
                raise ValueError(f"Correction parameters differ: {label}")
            for key, shape in shapes.items():
                if (
                    tuple(state[key].shape) != shape
                    or not torch.isfinite(state[key]).all()
                ):
                    raise ValueError(f"Invalid correction tensor: {label}/{key}")
            actor_norm = float(state["actor.weight"].norm())
            public_norm = float(state["body.0.weight"][:, 560:656].norm())
            if actor_norm == 0:
                raise ValueError(f"Actor correction remains untrained: {label}")
            if ("preview1" in profile or "choice1" in profile) and public_norm == 0:
                raise ValueError(f"Public-card input weights remain zero: {label}")
            bindings[str(checkpoint)] = sha(checkpoint)
            results[label] = {
                "parent_actor_value_config_identical": True,
                "finite_correction_tensors": len(state),
                "actor_weight_norm": actor_norm,
                "public_card_input_weight_norm": public_norm,
                "counters": saved["counters"],
            }
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    return {
        "status": "passed",
        "models": results,
        "bindings": bindings,
        "scope": "Saved identity, finite tensors and nonzero adaptation weights; not strength evidence or proof of useful use of public features",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    result = audit(Path(args.study))
    write(destination, result)
    print({"status": result["status"], "models": len(result["models"])})


if __name__ == "__main__":
    main()
