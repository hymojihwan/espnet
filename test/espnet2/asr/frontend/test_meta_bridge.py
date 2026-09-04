import pytest
import torch

from espnet2.asr.frontend.meta_bridge import MetaBridgeFrontend
from espnet2.asr.frontend.se_meta_bridge import SEMetaBridgeFrontend
from espnet2.tasks.asr_jepa import frontend_choices


def make_frontend(**overrides):
    config = dict(
        output_dim=8,
        n_mels=8,
        n_fft=32,
        win_length=32,
        hop_length=16,
        patch_size=[4, 8],
        mask_ratio=0.5,
        embedding_dim=8,
        context_encoder_dim=8,
        target_encoder_dim=8,
        predictor_dim=8,
        decoder_dim=8,
        num_context_encoder_layers=1,
        num_target_encoder_layers=1,
        num_predictor_layers=1,
        num_decoder_layers=1,
        encoder_type="mlp",
        predictor_type="mlp",
        decoder_type="mlp",
        dropout_rate=0.0,
        embedding_loss_weight=0.08,
        meta_adapter_dim=4,
        meta_inner_lr=1.0e-3,
        meta_inner_steps=1,
    )
    config.update(overrides)
    return MetaBridgeFrontend(**config)


def test_meta_bridge_train_backward():
    frontend = make_frontend()
    waveform = torch.randn(2, 640)
    lengths = torch.tensor([640, 560], dtype=torch.long)

    frontend.train()
    features, feature_lengths = frontend(waveform, lengths)
    meta_loss = frontend.compute_jepa_loss()
    assert features.shape == (2, 41, 8)
    assert feature_lengths.tolist() == [41, 36]
    assert meta_loss is not None
    assert torch.isfinite(meta_loss)

    (features.square().mean() + meta_loss).backward()
    assert frontend.meta_adapter.up.weight.grad is not None
    assert next(frontend.context_encoder.parameters()).grad is not None


def test_meta_bridge_adapts_inside_no_grad():
    frontend = make_frontend()
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)

    frontend.eval()
    with torch.no_grad():
        features, feature_lengths = frontend(waveform, lengths)

    assert features.shape == (1, 41, 8)
    assert feature_lengths.tolist() == [41]
    assert torch.isfinite(features).all()
    assert frontend.compute_jepa_loss() is None


def test_meta_bridge_rejects_reconstructed_asr_input():
    with pytest.raises(ValueError, match="asr_input_mode"):
        make_frontend(asr_input_mode="masked")


def test_se_meta_bridge_registered():
    assert frontend_choices.get_class("se_meta_bridge") is SEMetaBridgeFrontend
