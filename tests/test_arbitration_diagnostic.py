import pytest
import torch

from koikoi.arbitration import ArbitrationConfig, ArbitrationPolicyValue
from scripts.diagnose_arbitration import context_grid, grouped, proposal_mask


@pytest.mark.parametrize("context", ["full", "phase", "no_match"])
def test_controller_intervention_isolates_explicit_public_context(context):
    model = ArbitrationPolicyValue(ArbitrationConfig(context=context, width=1))
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()
        if context != "phase":
            model.actor[0].weight[0, 536] = 2
            model.actor[0].bias[0] = -1
            model.actor[2].weight[0, 0] = 1
            model.actor[4].weight[1, 0] = 1
        else:
            model.actor.bias[0] = 1
    public = torch.zeros(4, 562)
    public[:, 528] = 1
    public[:, 536:538] = 30 / 64
    public[:, 560] = 5
    public[:, 561] = 20
    original = public.clone()
    grid = context_grid(model, public)
    assert torch.equal(original, public)
    if context == "full":
        assert (grid[:, 0] == 5).all()
        assert (grid[:, -1] == 20).all()
    else:
        assert (grid == 5).all()
    chosen = model(public, proposal_mask(public))[0].argmax(-1)
    summary = grouped(public, chosen)
    assert summary["all"]["disagreements"] == 4
    assert summary["tied"]["states"] == 4
    assert summary["phase1"]["point_fraction_at_disagreement"] is None
