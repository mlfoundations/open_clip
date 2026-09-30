import torch

from open_clip.modified_resnet import ModifiedResNet


def test_modified_resnet_forward_intermediates_early_indices():
    torch.manual_seed(0)
    model = ModifiedResNet([1, 1, 1, 1], output_dim=16, heads=2, image_size=64, width=8).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        # indices that stop before the last stage must still run the full trunk for the pooled features
        output = model.forward_intermediates(x, indices=[0, 1])
        early = model.forward_intermediates(x, indices=[0, 1], stop_early=True, intermediates_only=True)
        expected = model(x)

    assert torch.allclose(output['image_features'], expected)
    assert [t.shape[1] for t in output['image_intermediates']] == [8, 32]
    for a, b in zip(output['image_intermediates'], early['image_intermediates']):
        assert torch.equal(a, b)
