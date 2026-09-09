import torch

from espnet2.asr.ajepa_spectrogram_mask import AJEPASpectrogramMask
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


def test_ajepa_mask_is_exact_patchwise_and_uses_private_rng():
    masker = AJEPASpectrogramMask(
        input_dim=80,
        patch_size=(16, 16),
        mask_ratio=0.75,
        strategy="random",
        seed=7,
    )
    features = torch.randn(2, 70, 80)
    lengths = torch.tensor([64, 64])

    rng_before = torch.random.get_rng_state().clone()
    masked, patch_mask = masker(features, lengths)
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert patch_mask.shape == (2, 4, 5)
    assert torch.equal(
        patch_mask.sum(dim=(1, 2)),
        torch.tensor([15, 15]),
    )
    assert masker.get_stats()["ajepa_target_patch_ratio"].item() == 0.75
    cell_mask = patch_mask.repeat_interleave(16, 1).repeat_interleave(16, 2)
    assert torch.equal(masked[:, :64][cell_mask], torch.zeros(30 * 256))
    assert torch.equal(masked[:, 64:], features[:, 64:])


def test_ajepa_spectrogram_ssl_keeps_asr_view_unmasked():
    torch.manual_seed(2)
    bridge = _make_bridge(
        spectrogram_masking=True,
        spectrogram_input_dim=80,
        spectrogram_patch_size=(16, 16),
        spectrogram_mask_ratio=0.75,
        spectrogram_mask_strategy="random",
    ).train()
    base_features = torch.randn(2, 12, 8)
    ssl_features = base_features + 0.5 * torch.randn_like(base_features)
    lengths = torch.tensor([12, 9])
    target_weights = torch.full((2, 12), 0.75)
    target_weights[1, 9:] = 0.0

    output, output_lengths = bridge(
        base_features,
        lengths,
        ssl_features=ssl_features,
        ssl_target_weights=target_weights,
    )
    loss = output.square().mean() + bridge.loss_weight * bridge.compute_loss()
    loss.backward()

    # Identity initialization proves the ASR branch used base_features rather
    # than the corrupted SSL view or a predicted hybrid.
    assert torch.equal(output, base_features)
    assert torch.equal(output_lengths, lengths)
    assert not bridge.mask_token.requires_grad
    assert bridge.mask_token.grad is None
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


def test_ajepa_curriculum_advances_with_ema_update():
    bridge = _make_bridge(
        spectrogram_masking=True,
        spectrogram_mask_strategy="curriculum",
        spectrogram_curriculum_steps=10,
    )
    assert bridge.spectrogram_masker.curriculum_step.item() == 0
    assert bridge.spectrogram_masker.curriculum_probability() == 0.01

    bridge.update_target_encoder()

    assert bridge.spectrogram_masker.curriculum_step.item() == 1
    assert bridge.spectrogram_masker.curriculum_probability() > 0.01


def test_patch_mask_projects_to_conv2d_center_aligned_weights():
    masker = AJEPASpectrogramMask(
        input_dim=80,
        patch_size=(16, 16),
        mask_ratio=0.75,
    )
    patch_mask = torch.zeros(1, 3, 5, dtype=torch.bool)
    patch_mask[:, 1, :4] = True
    weights = masker.project_to_latent(
        patch_mask,
        feature_time_steps=48,
        latent_time_steps=11,
        latent_lengths=torch.tensor([11]),
    )

    # Centers 19, 23, 27, and 31 correspond to time patch 1.
    assert torch.equal(weights[0, 4:8], torch.full((4,), 0.8))
    assert torch.equal(weights[0, :4], torch.zeros(4))
    assert torch.equal(weights[0, 8:], torch.zeros(3))
