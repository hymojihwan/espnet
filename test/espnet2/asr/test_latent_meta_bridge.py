import torch
import torch.nn as nn

from espnet2.asr.latent_meta_bridge import LatentMetaBridge


class SecondOrderSafePredictor(nn.Module):
    """Small predictor avoiding attention kernels without double backward."""

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, features, src_key_padding_mask=None):
        if src_key_padding_mask is None:
            context = features.mean(dim=1, keepdim=True)
        else:
            valid = (~src_key_padding_mask).unsqueeze(-1).to(features.dtype)
            context = (features * valid).sum(dim=1, keepdim=True)
            context = context / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.net(features + context)


def make_bridge(**overrides):
    config = dict(
        input_dim=8,
        latent_dim=8,
        adapter_dim=4,
        adapter_scale=1.0,
        predictor_layers=1,
        predictor_heads=2,
        predictor_ff_dim=16,
        dropout_rate=0.0,
        mask_ratio=0.5,
        mask_span=1,
        inner_lr=0.05,
        inner_steps=1,
        first_order=False,
        gradient_clip=1.0,
        adapt_during_training=True,
        adapt_at_inference=True,
        train_per_sample=False,
        ema_decay=0.75,
        inference_seed=17,
        loss_type="smooth_l1",
    )
    config.update(overrides)
    return LatentMetaBridge(**config)


def test_exact_maml_backward_through_frozen_asr_tail(monkeypatch):
    torch.manual_seed(0)
    bridge = make_bridge(first_order=False)
    # Some CPU scaled-dot-product-attention kernels do not implement double
    # backward.  The Meta-BRIDGE graph contract is independent of that kernel,
    # so use an equivalent differentiable sequence predictor in this test.
    bridge.predictor = SecondOrderSafePredictor(bridge.latent_dim)
    frozen_tail = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 5))
    for parameter in frozen_tail.parameters():
        parameter.requires_grad = False

    observed_create_graph = []
    original_grad = torch.autograd.grad

    def record_grad(*args, **kwargs):
        observed_create_graph.append(kwargs.get("create_graph", False))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    features = torch.randn(2, 9, 8)
    lengths = torch.tensor([9, 7], dtype=torch.long)

    bridge.train()
    aligned, output_lengths = bridge(features, lengths)
    frozen_tail(aligned).square().mean().backward()

    assert observed_create_graph == [True]
    assert aligned.shape == features.shape
    assert torch.equal(output_lengths, lengths)
    stats = bridge.get_stats()
    for name in (
        "latent_meta_support_pre",
        "latent_meta_support_post",
        "latent_meta_adapter_delta_norm",
        "latent_meta_feature_delta_rms",
    ):
        assert torch.isfinite(stats[name])
    assert stats["latent_meta_adapter_delta_norm"] > 0.0

    adapter_gradient = bridge.adapter.up.weight.grad
    assert adapter_gradient is not None
    assert torch.isfinite(adapter_gradient).all()
    assert adapter_gradient.abs().sum() > 0.0
    predictor_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in bridge.predictor.parameters()
        if parameter.grad is not None
    )
    assert predictor_gradient > 0.0
    assert all(parameter.grad is None for parameter in frozen_tail.parameters())
    assert all(
        parameter.grad is None for parameter in bridge.target_encoder.parameters()
    )


def test_eval_adapts_inside_no_grad_deterministically_without_mutation():
    torch.manual_seed(1)
    bridge = make_bridge(train_per_sample=True, inference_seed=23)
    features = torch.randn(2, 10, 8)
    lengths = torch.tensor([10, 6], dtype=torch.long)
    state_before = {
        name: value.detach().clone() for name, value in bridge.state_dict().items()
    }

    bridge.eval()
    with torch.no_grad():
        first, first_lengths = bridge(features, lengths)
        first_stats = bridge.get_stats()
        second, second_lengths = bridge(features, lengths)

    assert torch.equal(first, second)
    assert torch.equal(first_lengths, lengths)
    assert torch.equal(second_lengths, lengths)
    assert first.grad_fn is None
    assert not first.requires_grad
    assert first_stats["latent_meta_adapter_delta_norm"] > 0.0
    assert torch.isfinite(first_stats["latent_meta_support_post"])
    for name, value in bridge.state_dict().items():
        assert torch.equal(value, state_before[name]), name


def test_k0_with_zero_adapter_scale_is_exact_identity():
    torch.manual_seed(2)
    bridge = make_bridge(
        adapter_scale=0.0,
        adapt_during_training=False,
        adapt_at_inference=False,
    )
    with torch.no_grad():
        for parameter in bridge.adapter.parameters():
            parameter.normal_()
    features = torch.randn(2, 8, 8)
    lengths = torch.tensor([8, 5], dtype=torch.long)

    for training in (True, False):
        bridge.train(training)
        with torch.no_grad():
            aligned, output_lengths = bridge(features, lengths)
        stats = bridge.get_stats()

        assert torch.equal(aligned, features)
        assert torch.equal(output_lengths, lengths)
        assert stats["latent_meta_adapter_delta_norm"].item() == 0.0
        assert stats["latent_meta_feature_delta_rms"].item() == 0.0
        assert stats["latent_meta_feature_delta_relative"].item() == 0.0


def test_ema_target_is_frozen_eval_and_updates_by_decay():
    bridge = make_bridge(ema_decay=0.75)
    bridge.train()
    assert not bridge.target_encoder.training
    assert all(
        not parameter.requires_grad
        for parameter in bridge.target_encoder.parameters()
    )

    with torch.no_grad():
        for parameter in bridge.online_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in bridge.target_encoder.parameters():
            parameter.zero_()
    adapter_before = {
        name: value.detach().clone()
        for name, value in bridge.adapter.state_dict().items()
    }
    predictor_before = {
        name: value.detach().clone()
        for name, value in bridge.predictor.state_dict().items()
    }
    mask_token_before = bridge.mask_token.detach().clone()

    bridge.update_target_encoder()

    for parameter in bridge.target_encoder.parameters():
        assert torch.equal(parameter, torch.full_like(parameter, 0.5))
        assert parameter.grad is None
    for name, value in bridge.adapter.state_dict().items():
        assert torch.equal(value, adapter_before[name])
    for name, value in bridge.predictor.state_dict().items():
        assert torch.equal(value, predictor_before[name])
    assert torch.equal(bridge.mask_token, mask_token_before)
    assert not bridge.target_encoder.training
