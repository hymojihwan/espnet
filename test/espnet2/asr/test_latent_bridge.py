import torch

from espnet2.asr.latent_bridge import LatentBridge


def _make_bridge(**overrides):
    config = dict(
        input_dim=8,
        context_layers=1,
        context_heads=2,
        context_ff_dim=16,
        predictor_layers=1,
        predictor_heads=2,
        predictor_ff_dim=16,
        dropout_rate=0.0,
        mask_ratio=0.5,
        mask_span=1,
        ema_decay=0.75,
        loss_type="cosine",
        loss_weight=0.08,
    )
    config.update(overrides)
    return LatentBridge(**config)


def test_train_hybrid_backpropagates_without_target_gradient():
    torch.manual_seed(0)
    bridge = _make_bridge().train()
    features = torch.randn(2, 8, 8)
    lengths = torch.tensor([8, 5])

    output, output_lengths = bridge(features, lengths)
    loss = output.square().mean() + bridge.loss_weight * bridge.compute_loss()
    loss.backward()

    assert output.shape == features.shape
    assert torch.equal(output_lengths, lengths)
    assert torch.equal(output[1, 5:], features[1, 5:])
    assert torch.isfinite(bridge.compute_loss())
    assert bridge.mask_token.grad is not None
    assert bridge.mask_token.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None
        for parameter in bridge.online_encoder.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in bridge.predictor.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in bridge.target_encoder.parameters()
    )


def test_eval_is_unmasked_deterministic_and_identity_at_initialization():
    torch.manual_seed(1)
    bridge = _make_bridge().eval()
    features = torch.randn(2, 7, 8)
    lengths = torch.tensor([7, 4])

    with torch.no_grad():
        first, first_lengths = bridge(features, lengths)
        second, second_lengths = bridge(features, lengths)

    assert torch.equal(first, features)
    assert torch.equal(second, features)
    assert torch.equal(first_lengths, lengths)
    assert torch.equal(second_lengths, lengths)
    assert bridge.compute_loss() is None
    assert bridge.get_stats()["latent_bridge_mask_ratio"].item() == 0.0
    assert not bridge.target_encoder.training


def test_ema_updates_only_target_encoder():
    bridge = _make_bridge(ema_decay=0.75)
    with torch.no_grad():
        for parameter in bridge.online_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in bridge.target_encoder.parameters():
            parameter.zero_()
        predictor_before = {
            name: value.clone()
            for name, value in bridge.predictor.state_dict().items()
        }

    bridge.update_target_encoder()

    for parameter in bridge.target_encoder.parameters():
        assert torch.equal(parameter, torch.full_like(parameter, 0.5))
        assert not parameter.requires_grad
    for name, value in bridge.predictor.state_dict().items():
        assert torch.equal(value, predictor_before[name])
    assert not bridge.target_encoder.training
