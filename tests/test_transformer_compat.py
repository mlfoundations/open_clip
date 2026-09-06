"""CustomTransformer compatibility and checkpoint recomputation for both block signatures."""
import pickle
from copy import deepcopy

import pytest
import torch
from torch import nn

from open_clip.transformer import CustomTransformer, LayerNorm, Transformer


@pytest.mark.parametrize("block_types", [
    "CustomResidualAttentionBlock", ["CustomResidualAttentionBlock"] * 2,
])
def test_custom_transformer_constructor_and_pickle(block_types):
    torch.manual_seed(19)
    model = CustomTransformer(16, 2, 2, 2.5, 0.1, nn.SiLU, LayerNorm, block_types)
    torch.manual_seed(19)
    reference = Transformer(16, 2, 2, 2.5, 0.1, nn.SiLU, LayerNorm, block_type="custom")
    reference.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 4, 16)
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
    restored = pickle.loads(pickle.dumps(model))
    torch.testing.assert_close(restored(x), reference(x))


@pytest.mark.parametrize("kind", ["default", "custom", "custom_features"])
@pytest.mark.parametrize("impl", ["inline", "composable"])
@pytest.mark.parametrize("intermediates", [False, True])
def test_transformer_checkpoint_outputs_and_gradients(kind, impl, intermediates):
    options = dict(qk_norm=True, scale_attn=True, scale_fc=True) if kind == "custom_features" else {}
    model = Transformer(16, 2, 2, block_type="custom" if options else kind, **options)
    checkpointed = deepcopy(model)
    checkpointed.set_grad_checkpointing(impl=impl)
    x = torch.randn(2, 4, 16)
    mask = torch.full((4, 4), float("-inf")).triu(1)

    def run(m):
        inputs = x.clone().requires_grad_()
        if intermediates:
            output, selected = m.forward_intermediates(inputs, attn_mask=mask, indices=[0, 1])
            loss = sum(value.square().mean() for value in selected)
        else:
            output = m(inputs, attn_mask=mask)
            loss = output.square().mean()
        loss.backward()
        assert all(p.grad is not None for p in m.parameters())
        return output, inputs.grad, {name: p.grad for name, p in m.named_parameters()}

    torch.testing.assert_close(run(checkpointed), run(model))
