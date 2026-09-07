import pytest
import torch
import torch.nn as nn

from espnet2.asr.espnet_jepa_model import ESPnetJEPAASRModel
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
    stats = frontend.get_jepa_loss_stats()
    assert features.shape == (2, 41, 8)
    assert feature_lengths.tolist() == [41, 36]
    assert meta_loss is not None
    assert torch.isfinite(meta_loss)
    assert torch.isfinite(stats["loss_meta_support_pre"])
    assert torch.isfinite(stats["loss_meta_support_post"])
    assert torch.isfinite(stats["meta_feature_delta_rms"])
    assert torch.isfinite(stats["meta_feature_delta_relative"])

    (features.square().mean() + meta_loss).backward()
    assert frontend.meta_adapter.up.weight.grad is not None
    assert next(frontend.context_encoder.parameters()).grad is not None


def test_meta_bridge_can_disable_outer_support_loss():
    frontend = make_frontend(
        meta_outer_support_loss=False,
        encoder_type="transformer",
        predictor_type="transformer",
        meta_inner_lr=1.0e-2,
    )
    waveform = torch.randn(2, 640)
    lengths = torch.tensor([640, 560], dtype=torch.long)

    frontend.train()
    features, _ = frontend(waveform, lengths)
    stats = frontend.get_jepa_loss_stats()

    assert frontend.compute_jepa_loss() is None
    assert "loss_jepa_mel_reconstruction" not in stats
    assert stats["meta_adapter_delta_norm"] > 0.0
    assert stats["meta_feature_delta_rms"] > 0.0
    assert stats["loss_meta_support_post"] <= (
        stats["loss_meta_support_pre"] + 1.0e-6
    )

    features.square().mean().backward()
    assert frontend.meta_adapter.up.weight.grad is not None
    assert next(frontend.context_encoder.parameters()).grad is None


def test_meta_bridge_training_adaptation_can_be_per_sample(monkeypatch):
    frontend = make_frontend(
        meta_train_per_sample=True,
        meta_outer_support_loss=False,
    )
    waveform = torch.randn(2, 640)
    lengths = torch.tensor([640, 560], dtype=torch.long)
    observed_batch_sizes = []
    original_inner_adapt = frontend._inner_adapt

    def record_batch_size(base_features, feature_lengths):
        observed_batch_sizes.append(base_features.size(0))
        return original_inner_adapt(base_features, feature_lengths)

    monkeypatch.setattr(frontend, "_inner_adapt", record_batch_size)
    frontend.train()
    features, _ = frontend(waveform, lengths)

    assert features.shape == (2, 41, 8)
    assert observed_batch_sizes == [1, 1]
    assert torch.isfinite(frontend.get_jepa_loss_stats()[
        "loss_meta_support_post"
    ])


def test_meta_bridge_validation_adaptation_is_per_sample(monkeypatch):
    frontend = make_frontend(
        meta_train_per_sample=True,
        meta_outer_support_loss=False,
    )
    waveform = torch.randn(2, 640)
    lengths = torch.tensor([640, 560], dtype=torch.long)
    observed_batch_sizes = []
    original_inner_adapt = frontend._inner_adapt

    def record_batch_size(base_features, feature_lengths):
        observed_batch_sizes.append(base_features.size(0))
        return original_inner_adapt(base_features, feature_lengths)

    monkeypatch.setattr(frontend, "_inner_adapt", record_batch_size)
    frontend.eval()
    with torch.no_grad():
        features, feature_lengths = frontend(waveform, lengths)

    assert features.shape == (2, 41, 8)
    assert feature_lengths.tolist() == [41, 36]
    assert observed_batch_sizes == [1, 1]
    assert frontend.compute_jepa_loss() is None


def test_meta_bridge_adapter_query_is_unmasked_target_free_and_episodic(
    monkeypatch,
):
    frontend = make_frontend(
        meta_query_input_mode="adapter",
        meta_train_per_sample=True,
        meta_outer_support_loss=False,
        meta_inner_eval_mode=True,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)
    clean_a = torch.randn_like(waveform)
    clean_b = torch.randn_like(waveform)
    before = {
        name: parameter.detach().clone()
        for name, parameter in frontend.meta_adapter.named_parameters()
    }

    def reject_masked_hybrid(*args, **kwargs):
        raise AssertionError("adapter query must not reconstruct query patches")

    monkeypatch.setattr(
        frontend,
        "_build_bridge_query",
        reject_masked_hybrid,
    )
    frontend.eval()
    with torch.no_grad():
        first, first_lengths = frontend(
            waveform,
            lengths,
            clean_input=clean_a,
            clean_input_lengths=lengths,
        )
        second, second_lengths = frontend(
            waveform,
            lengths,
            clean_input=clean_b,
            clean_input_lengths=lengths,
        )

    # Clean speech is ignored, the ASR query has the original frame layout,
    # and each call starts from the same persistent adapter initialization.
    assert torch.equal(first, second)
    assert torch.equal(first_lengths, second_lengths)
    assert first.shape == (1, 41, 8)
    assert frontend.get_jepa_loss_stats()["meta_feature_delta_rms"] > 0.0
    assert frontend._last_meta_support_mask is not None
    assert frontend._last_meta_query_mask is None
    for name, parameter in frontend.meta_adapter.named_parameters():
        assert torch.equal(before[name], parameter)


def test_meta_bridge_inner_eval_mode_restores_training_state(monkeypatch):
    frontend = make_frontend(
        dropout_rate=0.5,
        meta_inner_eval_mode=True,
        meta_outer_support_loss=False,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)
    observed_modes = []
    original_loss = frontend._masked_reconstruction_loss

    def record_modes(*args, **kwargs):
        observed_modes.append(
            (
                frontend.context_encoder.training,
                frontend.predictor.training,
                frontend.decoder.training,
            )
        )
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(frontend, "_masked_reconstruction_loss", record_modes)
    frontend.train()
    features, _ = frontend(waveform, lengths)
    features.square().mean().backward()

    assert observed_modes
    assert all(mode == (False, False, False) for mode in observed_modes)
    assert frontend.context_encoder.training
    assert frontend.predictor.training
    assert frontend.decoder.training
    assert frontend.meta_adapter.up.weight.grad is not None


def test_meta_bridge_query_matches_inner_eval_mode(monkeypatch):
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_inner_eval_mode=True,
        meta_outer_support_loss=False,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)
    observed_modes = []
    original_query = frontend._build_bridge_query

    def record_modes(*args, **kwargs):
        observed_modes.append(
            (
                frontend.context_encoder.training,
                frontend.predictor.training,
                frontend.decoder.training,
            )
        )
        return original_query(*args, **kwargs)

    monkeypatch.setattr(frontend, "_build_bridge_query", record_modes)
    frontend.train()
    frontend(waveform, lengths)

    assert observed_modes == [(False, False, False)]
    assert frontend.context_encoder.training
    assert frontend.predictor.training
    assert frontend.decoder.training


@pytest.mark.parametrize("enabled", [False, True])
def test_meta_freeze_backbone_eval_is_opt_in(enabled):
    meta_frontend = make_frontend(meta_freeze_backbone_eval=enabled)
    wrapper = nn.Module()
    wrapper.jepa_frontend = meta_frontend
    wrapper.se_model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    wrapper.se_no_grad = enabled

    model = ESPnetJEPAASRModel.__new__(ESPnetJEPAASRModel)
    nn.Module.__init__(model)
    model.frontend = wrapper
    model.encoder = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    model.ctc = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    model.decoder = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    if enabled:
        frozen_modules = (
            model.encoder,
            model.ctc,
            model.decoder,
            wrapper.se_model,
            meta_frontend.context_patch_embed,
            meta_frontend.context_encoder,
            meta_frontend.predictor,
            meta_frontend.decoder,
        )
        for module in frozen_modules:
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False
    model.train()

    model._enforce_meta_frozen_backbone_eval()

    expected_training = not enabled
    assert model.encoder.training is expected_training
    assert model.ctc.training is expected_training
    assert model.decoder.training is expected_training
    assert wrapper.se_model.training is expected_training
    assert meta_frontend.context_encoder.training is expected_training
    assert meta_frontend.predictor.training is expected_training
    assert meta_frontend.decoder.training is expected_training
    # The episode controller and adapter remain in training mode so the ASR
    # outer objective can still differentiate through the frozen backbone.
    assert meta_frontend.training
    assert meta_frontend.meta_adapter.training

    if enabled:
        adapter_input = torch.randn(1, 3, 8, requires_grad=True)
        model.encoder(adapter_input).sum().backward()
        assert adapter_input.grad is not None


def test_meta_frozen_eval_keeps_outer_trainable_asr_in_train_mode():
    meta_frontend = make_frontend(meta_freeze_backbone_eval=True)
    wrapper = nn.Module()
    wrapper.jepa_frontend = meta_frontend
    wrapper.se_model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    wrapper.se_no_grad = True

    model = ESPnetJEPAASRModel.__new__(ESPnetJEPAASRModel)
    nn.Module.__init__(model)
    model.frontend = wrapper
    model.encoder = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    model.ctc = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    model.decoder = None

    # SE and the SSL reconstruction head are frozen; the ASR is updated only
    # by the supervised outer objective during offline meta-training.
    for module in (
        wrapper.se_model,
        meta_frontend.context_patch_embed,
        meta_frontend.context_encoder,
        meta_frontend.predictor,
        meta_frontend.decoder,
    ):
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False

    model.train()
    model._enforce_meta_frozen_backbone_eval()

    assert model.encoder.training
    assert model.ctc.training
    assert not wrapper.se_model.training
    assert not meta_frontend.context_encoder.training
    assert not meta_frontend.predictor.training
    assert not meta_frontend.decoder.training
    assert meta_frontend.training
    assert meta_frontend.meta_adapter.training

    adapter_input = torch.randn(1, 3, 8, requires_grad=True)
    outer_loss = model.ctc(model.encoder(adapter_input)).sum()
    outer_loss.backward()
    assert adapter_input.grad is not None
    assert next(model.encoder.parameters()).grad is not None
    assert next(model.ctc.parameters()).grad is not None


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


def test_exact_maml_eval_does_not_build_higher_order_graph(monkeypatch):
    frontend = make_frontend(
        meta_first_order=False,
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)
    observed_create_graph = []
    original_grad = torch.autograd.grad

    def record_grad(*args, **kwargs):
        observed_create_graph.append(kwargs.get("create_graph", False))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    frontend.eval()
    with torch.no_grad():
        features, _ = frontend(waveform, lengths)

    assert observed_create_graph == [False]
    assert features.grad_fn is None


def test_meta_bridge_support_and_query_masks_are_disjoint():
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_disjoint_support_query=True,
    )
    features = torch.randn(2, 12, 8)
    lengths = torch.tensor([12, 8], dtype=torch.long)

    frontend.eval()
    support_mask = frontend._make_support_mask(features, lengths, step=0)
    query_mask = frontend._make_query_mask(
        features,
        lengths,
        support_mask,
        step=0,
    )
    _, patch_mask, _ = frontend._create_patches(features, lengths)

    assert not (support_mask & ~patch_mask).any()
    assert not (query_mask & ~patch_mask).any()
    assert not (support_mask & query_mask).any()
    assert support_mask.sum(dim=1).tolist() == [1, 1]
    assert query_mask.sum(dim=1).tolist() == [1, 1]


def test_meta_bridge_hybrid_preserves_unmasked_asr_patches():
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_query_residual_weight=1.0,
    )
    features = torch.randn(1, 12, 8)
    lengths = torch.tensor([12], dtype=torch.long)
    parameters = dict(frontend.meta_adapter.named_parameters())

    frontend.eval()
    support_mask = frontend._make_support_mask(features, lengths, step=0)
    query_mask = frontend._make_query_mask(
        features,
        lengths,
        support_mask,
        step=0,
    )
    output_full = frontend._build_bridge_query(
        features,
        lengths,
        parameters,
        query_mask,
    )
    base_patches, patch_mask, _ = frontend._create_patches(features, lengths)
    output_patches, _, _ = frontend._create_patches(output_full, lengths)
    unmasked = patch_mask & ~query_mask
    assert torch.equal(output_patches[unmasked], base_patches[unmasked])

    frontend.meta_query_residual_weight = 0.0
    output_zero = frontend._build_bridge_query(
        features,
        lengths,
        parameters,
        query_mask,
    )
    frontend.meta_query_residual_weight = 0.5
    output_half = frontend._build_bridge_query(
        features,
        lengths,
        parameters,
        query_mask,
    )
    assert torch.allclose(output_zero, features)
    assert torch.allclose(
        output_half,
        output_zero + 0.5 * (output_full - output_zero),
        atol=1.0e-6,
        rtol=1.0e-5,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"meta_inner_steps": 0},
        {"meta_adapt_at_inference": False},
    ],
)
def test_meta_bridge_k0_still_builds_bridge_query(monkeypatch, overrides):
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        **overrides,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)
    num_query_calls = 0
    original_build_query = frontend._build_bridge_query

    def record_query(*args, **kwargs):
        nonlocal num_query_calls
        num_query_calls += 1
        return original_build_query(*args, **kwargs)

    monkeypatch.setattr(frontend, "_build_bridge_query", record_query)
    frontend.eval()
    with torch.no_grad():
        frontend(waveform, lengths)

    assert num_query_calls == 1
    assert frontend._last_meta_query_mask is not None
    assert frontend._last_meta_query_mask.any()


@pytest.mark.parametrize(
    "overrides",
    [
        {"meta_inner_steps": 1},
        {"meta_inner_steps": 0},
        {"meta_inner_steps": 1, "meta_adapt_at_inference": False},
    ],
)
def test_meta_bridge_forward_returns_bridge_query(monkeypatch, overrides):
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        **overrides,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)

    def sentinel_query(asr_features, *args, **kwargs):
        return torch.full_like(asr_features, 7.25)

    monkeypatch.setattr(frontend, "_build_bridge_query", sentinel_query)
    frontend.eval()
    with torch.no_grad():
        features, _ = frontend(waveform, lengths)

    assert torch.equal(features, torch.full_like(features, 7.25))


def test_meta_bridge_k0_k1_use_identical_eval_query_mask():
    common = dict(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_train_per_sample=True,
    )
    k1_frontend = make_frontend(**common)
    k0_frontend = make_frontend(
        **common,
        meta_adapt_at_inference=False,
    )
    k0_frontend.load_state_dict(k1_frontend.state_dict())
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)

    k0_frontend.eval()
    k1_frontend.eval()
    with torch.no_grad():
        k0_frontend(waveform, lengths)
        k1_frontend(waveform, lengths)

    assert torch.equal(
        k0_frontend._last_meta_support_mask,
        k1_frontend._last_meta_support_mask,
    )
    assert torch.equal(
        k0_frontend._last_meta_query_mask,
        k1_frontend._last_meta_query_mask,
    )


def test_meta_bridge_one_patch_query_falls_back_to_base():
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_disjoint_support_query=True,
    )
    features = torch.randn(1, 4, 8)
    lengths = torch.tensor([4], dtype=torch.long)

    frontend.eval()
    output, auxiliary_loss, _ = (
        frontend._query_episode_without_adaptation(
            features,
            lengths,
            features,
            lengths,
        )
    )

    assert frontend._last_meta_support_mask.sum() == 1
    assert frontend._last_meta_query_mask.sum() == 0
    assert torch.equal(output, features)
    assert auxiliary_loss is None


def test_meta_bridge_eval_query_is_deterministic_and_episode_local():
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_train_per_sample=True,
    )
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)
    before = {
        name: parameter.detach().clone()
        for name, parameter in frontend.meta_adapter.named_parameters()
    }

    frontend.eval()
    with torch.no_grad():
        first, _ = frontend(waveform, lengths)
        first_support = frontend._last_meta_support_mask.clone()
        first_query = frontend._last_meta_query_mask.clone()
        second, _ = frontend(waveform, lengths)

    assert torch.equal(first, second)
    assert torch.equal(first_support, frontend._last_meta_support_mask)
    assert torch.equal(first_query, frontend._last_meta_query_mask)
    for name, parameter in frontend.meta_adapter.named_parameters():
        assert torch.equal(before[name], parameter)


@pytest.mark.parametrize("first_order", [True, False])
def test_outer_gradient_flows_through_frozen_bridge_query(first_order):
    frontend = make_frontend(
        meta_query_input_mode="bridge_hybrid",
        meta_query_mask_ratio=0.5,
        meta_train_per_sample=True,
        meta_outer_support_loss=False,
        meta_first_order=first_order,
        meta_inner_lr=1.0e-2,
    )
    for name, parameter in frontend.named_parameters():
        if not name.startswith("meta_adapter."):
            parameter.requires_grad = False
    waveform = torch.randn(1, 640)
    lengths = torch.tensor([640], dtype=torch.long)

    frontend.train()
    features, _ = frontend(waveform, lengths)
    features.square().mean().backward()

    adapter_gradient = frontend.meta_adapter.up.weight.grad
    assert adapter_gradient is not None
    assert torch.isfinite(adapter_gradient).all()
    assert adapter_gradient.abs().sum() > 0.0
    for name, parameter in frontend.named_parameters():
        if not name.startswith("meta_adapter."):
            assert parameter.grad is None


def test_meta_bridge_rejects_reconstructed_asr_input():
    with pytest.raises(ValueError, match="asr_input_mode"):
        make_frontend(asr_input_mode="masked")


def test_se_meta_bridge_registered():
    assert frontend_choices.get_class("se_meta_bridge") is SEMetaBridgeFrontend
