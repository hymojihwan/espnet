"""Feature-domain JEPA masked-patch frontend for ASR.

This frontend consumes extracted noisy features directly, while clean targets
can still be provided as raw waveform and converted to clean log-mel inside the
frontend.
"""

from typing import Optional, Tuple

import torch

from espnet2.asr.frontend.jepa_masked import JEPA_MaskedPatchFrontend


class JEPA_MaskedPatchFeatureFrontend(JEPA_MaskedPatchFrontend):
    """JEPA masked-patch frontend operating on feature sequences directly."""

    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 80,
        **kwargs,
    ):
        super().__init__(
            output_dim=output_dim,
            n_mels=input_dim,
            **kwargs,
        )
        self.input_dim = input_dim

    def _extract_logmel(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle both extracted features and raw waveform clean targets."""
        if input.dim() == 3:
            if input.size(-1) != self.input_dim:
                raise ValueError(
                    f"Expected feature dim {self.input_dim}, but got {input.size(-1)}"
                )
            input = input[:, : input_lengths.max()]
            return input, input_lengths
        if input.dim() == 2:
            return super()._extract_logmel(input, input_lengths)
        raise ValueError(
            f"{self.__class__.__name__} expects (B, T, F) features or (B, T) waveform, got {tuple(input.shape)}"
        )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ):
        x_n, feats_lens = self._extract_logmel(input, input_lengths)
        batch_size, time_steps, feat_dim = x_n.shape
        original_shape = (time_steps, feat_dim)

        noisy_patches, patch_mask, patch_grid = self._create_patches(x_n, feats_lens)
        num_patches = noisy_patches.shape[1]

        if self.training:
            mask = self._random_mask_patches(num_patches, batch_size, x_n.device)
            mask = mask & patch_mask
        else:
            mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=x_n.device)

        if self.encoder_type == "transformer":
            noisy_patches_embedded = self.context_patch_embed(noisy_patches)
            if mask.any():
                mask_token_expanded = self.mask_token.expand(batch_size, num_patches, -1)
                noisy_patches_embedded = torch.where(
                    mask.unsqueeze(-1),
                    mask_token_expanded,
                    noisy_patches_embedded,
                )
            context_features = noisy_patches_embedded
            if self.context_pos_enc is not None:
                context_features = self.context_pos_enc(context_features)
            valid_mask = patch_mask.unsqueeze(1)
            for encoder_layer in self.context_encoder["encoders"]:
                context_features, valid_mask = encoder_layer(context_features, valid_mask)
            context_features = self.context_encoder["after_norm"](context_features)
        else:
            noisy_patches_masked = noisy_patches.clone()
            if mask.any():
                mask_token_expanded = self.mask_token.expand(batch_size, num_patches, -1)
                noisy_patches_masked = torch.where(
                    mask.unsqueeze(-1),
                    mask_token_expanded,
                    noisy_patches_masked,
                )
            context_features = self.context_encoder(noisy_patches_masked)

        if mask.any() and self.training and clean_input is not None:
            lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            x_c, clean_feats_lens = self._extract_logmel(clean_input, lens)
            if x_c.size(1) < time_steps:
                x_c = torch.nn.functional.pad(x_c, (0, 0, 0, time_steps - x_c.size(1)))
            elif x_c.size(1) > time_steps:
                x_c = x_c[:, :time_steps]
            clean_feats_lens = torch.minimum(clean_feats_lens.to(feats_lens.device), feats_lens)
            clean_patches, _, _ = self._create_patches(x_c, clean_feats_lens)

            with torch.no_grad():
                if self.encoder_type == "transformer":
                    target_latents = self.target_patch_embed(clean_patches)
                    if self.target_pos_enc is not None:
                        target_latents = self.target_pos_enc(target_latents)
                    valid_mask = patch_mask.unsqueeze(1)
                    for encoder_layer in self.target_encoder["encoders"]:
                        target_latents, valid_mask = encoder_layer(target_latents, valid_mask)
                    target_latents = self.target_encoder["after_norm"](target_latents)
                else:
                    target_latents = self.target_encoder(clean_patches)

            if self.predictor_type == "transformer":
                pred_features = context_features
                if self.predictor_pos_enc is not None:
                    pred_features = self.predictor_pos_enc(pred_features)
                valid_mask = patch_mask.unsqueeze(1)
                for encoder_layer in self.predictor["encoders"]:
                    pred_features, valid_mask = encoder_layer(pred_features, valid_mask)
                predicted_latents = self.predictor["after_norm"](pred_features)
            else:
                predicted_latents = self.predictor(context_features)

            masked_predicted_latents = predicted_latents[
                mask.unsqueeze(-1).expand_as(predicted_latents)
            ].view(-1, self.embedding_dim)
            reconstructed_patches = self.decoder(masked_predicted_latents)

            full_reconstructed_patches = noisy_patches.clone()
            full_reconstructed_patches[mask] = reconstructed_patches.to(full_reconstructed_patches.dtype)
            x_hat = self._unpatch(full_reconstructed_patches, original_shape, patch_grid)

            self._last_x_n = x_n
            self._last_x_c = x_c
            self._last_x_hat = x_hat
            self._last_mask = mask
            self._last_patch_grid = patch_grid
            self._last_noisy_patches = noisy_patches
            self._last_clean_patches = clean_patches
            self._last_reconstructed_patches = reconstructed_patches
        else:
            if mask.any():
                if self.predictor_type == "transformer":
                    pred_features = context_features
                    if self.predictor_pos_enc is not None:
                        pred_features = self.predictor_pos_enc(pred_features)
                    valid_mask = patch_mask.unsqueeze(1)
                    for encoder_layer in self.predictor["encoders"]:
                        pred_features, valid_mask = encoder_layer(pred_features, valid_mask)
                    predicted_latents = self.predictor["after_norm"](pred_features)
                    masked_predicted_latents = predicted_latents[
                        mask.unsqueeze(-1).expand_as(predicted_latents)
                    ].view(-1, self.embedding_dim)
                else:
                    masked_context_features = context_features[
                        mask.unsqueeze(-1).expand_as(context_features)
                    ].view(-1, self.embedding_dim)
                    masked_predicted_latents = self.predictor(masked_context_features)

                reconstructed_patches = self.decoder(masked_predicted_latents)
                full_reconstructed_patches = noisy_patches.clone()
                full_reconstructed_patches[mask] = reconstructed_patches.to(full_reconstructed_patches.dtype)
                x_hat = self._unpatch(full_reconstructed_patches, original_shape, patch_grid)
            else:
                x_hat = x_n

            self._last_x_n = None
            self._last_x_c = None
            self._last_x_hat = None
            self._last_mask = None
            self._last_patch_grid = None
            self._last_noisy_patches = None
            self._last_clean_patches = None
            self._last_reconstructed_patches = None

        return x_hat, feats_lens
