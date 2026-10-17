"""Explicit deterministic published training opponent, with pinned restart data."""

from copy import copy

import torch

from .upstream_batch import BatchPublishedAgent


def runtime_identity(device):
    device = torch.device(device)
    return {
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "mkldnn": torch.backends.mkldnn.enabled,
    }


class PublishedOpponent:
    def __init__(self, variant, device, endgame, snapshot=None):
        # This explicitly identified training mode fixes the teacher's arithmetic.
        # Ordinary self-play and historical-opponent construction do not set flags.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.agent = (
            BatchPublishedAgent(variant, device, endgame)
            if snapshot is None
            else BatchPublishedAgent.from_checkpoint(snapshot["agent"], device)
        )
        if (self.agent.variant, self.agent.endgame) != (variant, endgame):
            raise ValueError("Published opponent configuration mismatch")
        self.runtime = runtime_identity(self.agent.device)
        if snapshot is not None and snapshot["runtime"] != self.runtime:
            raise ValueError(
                "Published opponent runtime mismatch; exact resume rejected"
            )
        payload = self.agent.checkpoint()
        self.metadata = {
            "kind": "published-batch-v1",
            "selection": "deterministic_argmax",
            "grouping": "all_current_opponent_rows_by_phase",
            **{k: v for k, v in payload.items() if k != "weights"},
            "runtime": self.runtime.copy(),
        }

    def check_runtime(self):
        if runtime_identity(self.agent.device) != self.runtime:
            raise ValueError("Published opponent runtime changed during training")

    @torch.no_grad()
    def actions(self, env, learner):
        """Infer only actual opponent turns; never sample the teacher's policy."""
        indices = torch.where(~learner)[0]
        result = torch.zeros(env.count, dtype=torch.int64, device=env.state.device)
        if indices.numel():
            view = copy(env)
            view.state = env.state[indices].contiguous()
            view.count = len(indices)
            result[indices] = self.agent.act(view).long()
        return result

    def checkpoint(self):
        self.check_runtime()
        payload = self.agent.checkpoint()
        if any(
            payload[key] != self.metadata[key] for key in payload if key != "weights"
        ):
            raise ValueError("Published opponent identity changed during training")
        return {"agent": payload, "runtime": self.runtime.copy()}
