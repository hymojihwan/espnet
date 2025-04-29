import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentPredictor(nn.Module):
    """
    Alignment Predictor는 encoder와 decoder의 feature를 받아
    두 모달리티 간의 정렬 정보를 예측하는 auxiliary 네트워크입니다.
    """
    def __init__(self, d_enc: int, d_dec: int, hidden_dim: int):
        """
        Args:
            d_enc: encoder feature 차원.
            d_dec: decoder feature 차원.
            hidden_dim: 내부 결합(feature combination)을 위한 hidden 차원.
        """
        super().__init__()
        self.linear_enc = nn.Linear(d_enc, hidden_dim)
        self.linear_dec = nn.Linear(d_dec, hidden_dim)
        self.linear_score = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        """
        Args:
            encoder_out: [B, T, d_enc] – 음성 encoder 출력.
            decoder_out: [B, U, d_dec] – 텍스트 decoder 출력.
        Returns:
            alignment_pred: [B, T, U] – 각 encoder frame와 decoder token 간의 예측 정렬 확률.
        """
        B, T, _ = encoder_out.size()
        _, U, _ = decoder_out.size()

        # 각각의 feature를 hidden 차원으로 투영합니다.
        enc_proj = self.linear_enc(encoder_out)   # [B, T, hidden_dim]
        dec_proj = self.linear_dec(decoder_out)     # [B, U, hidden_dim]

        # 두 모달리티의 모든 조합에 대해 결합합니다.
        # encoder feature는 (B, T, 1, hidden_dim)으로, decoder feature는 (B, 1, U, hidden_dim)으로 확장한 뒤 더합니다.
        enc_expanded = enc_proj.unsqueeze(2).expand(B, T, U, -1)
        dec_expanded = dec_proj.unsqueeze(1).expand(B, T, U, -1)

        combined = torch.tanh(enc_expanded + dec_expanded)  # [B, T, U, hidden_dim]
        # 각 (T, U) 쌍에 대해 스칼라 score를 계산합니다.
        score = self.linear_score(combined).squeeze(-1)  # [B, T, U]
        # softmax를 통해 decoder token 차원(U)에서 정렬 확률로 변환합니다.
        alignment_pred = F.softmax(score, dim=-1)
        return alignment_pred