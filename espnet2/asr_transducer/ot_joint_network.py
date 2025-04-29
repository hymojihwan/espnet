import torch
import torch.nn as nn
import torch.nn.functional as F

class OTGuidedJointNetwork(torch.nn.Module):
    def __init__(self, d_enc: int, d_dec: int, d_fusion: int, vocab_size: int):
        """
        Args:
            d_enc: Encoder feature 차원.
            d_dec: Decoder feature 차원.
            d_fusion: 융합 후 내부 표현 차원.
            vocab_size: 전체 vocabulary 크기 (blank 포함).
        """
        super().__init__()
        # Encoder와 Decoder의 feature 차원을 맞추기 위한 projection
        self.lin_enc = nn.Linear(d_enc, d_fusion)
        self.ot_dec_proj = nn.Linear(d_dec, d_fusion)
        self.lin_dec = nn.Linear(d_dec, d_fusion)      # 원본 decoder feature용 (별도)

        # 최종 joint representation을 vocabulary로 매핑
        self.out_proj = nn.Linear(d_fusion, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,       # [B, T, d_enc]
        decoder_out: torch.Tensor,       # [B, U, d_dec]
        ot_decoder_feature: torch.Tensor     # [B, T, d_dec]
    ):
        """
        Args:
            encoder_out: [B, T, d_enc] – Encoder 출력.
            decoder_out: [B, U, d_dec] – Decoder 출력.
            transport_plan: [B, T, U] – OT 정렬 정보.
        Returns:
            logits: [B, T, U, vocab_size] – Joint network의 최종 출력.
        """
        # OT-guided decoder feature 생성: [B, T, d_dec]

        # Feature projection 수행
        proj_enc = self.lin_enc(encoder_out)               # [B, T, d_fusion]
        proj_ot_dec = self.ot_dec_proj(ot_decoder_feature) # [B, T, d_fusion]

        # 🚩 Residual 방식으로 결합 (gate 제거)
        fused_feature = proj_enc + proj_ot_dec             # [B, T, d_fusion]

        # Decoder feature도 같은 공간으로 투영
        proj_dec = self.lin_dec(decoder_out)                  # [B, U, d_fusion]

        # Broadcast 방식으로 두 feature를 결합
        fused_feature_exp = fused_feature.unsqueeze(2)     # [B, T, 1, d_fusion]
        proj_dec_exp = proj_dec.unsqueeze(1)               # [B, 1, U, d_fusion]

        joint_input = fused_feature_exp + proj_dec_exp     # [B, T, U, d_fusion]

        # non-linear activation
        joint_out = torch.tanh(joint_input)

        logits = self.out_proj(joint_out)                  # [B, T, U, vocab_size]

        return logits