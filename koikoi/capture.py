"""CUDA-graph minibatch execution with warmup updates completely rolled back."""

import torch


class CapturedRollout:
    """Capture a tensor-only collector; warmup must not advance games or RNG."""

    @torch.no_grad()
    def __init__(self, collect, state):
        device = state.device
        saved_state = state.clone()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    collect()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.batch, self.totals = collect()
            state.copy_(saved_state)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)

    def __call__(self):
        self.graph.replay()
        # PPO replaces its advantage entry after normalization. Preserve the
        # original graph output references for the next collection.
        return self.batch.copy(), self.totals


class CapturedMinibatch:
    """Fixed-shape step; model/Adam tensors retain their addresses after capture."""

    def __init__(self, step, model, optimizer, example):
        self.inputs = {key: value.detach().clone() for key, value in example.items()}
        device = next(model.parameters()).device
        saved_model = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        saved_optimizer = {
            parameter: {key: value.detach().clone() for key, value in state.items()}
            for parameter, state in optimizer.state.items()
        }
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step(self.inputs)
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph):
                self.output = step(self.inputs)
            # Warmup and recording must not add optimizer updates or change the
            # experiment's weights, moments, step counters, or random streams.
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    value.copy_(saved_model[key])
                for parameter, state in optimizer.state.items():
                    saved = saved_optimizer.get(parameter, {})
                    for key, value in state.items():
                        if key in saved:
                            value.copy_(saved[key])
                        else:
                            value.zero_()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)

    def __call__(self, inputs):
        for key, value in inputs.items():
            if (
                value.shape != self.inputs[key].shape
                or value.dtype != self.inputs[key].dtype
            ):
                raise ValueError("Captured minibatch shape/dtype changed")
            self.inputs[key].copy_(value)
        self.graph.replay()
        return self.output
