"""ESPnet2 ASR Transducer model."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked
import random

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.asr_transducer.activation import get_activation


if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:

    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRROTTransducerModel(AbsESPnetModel):
    """ESPnet2ASRTransducerModel module definition.

    Args:
        vocab_size: Size of complete vocabulary (w/ SOS/EOS and blank included).
        token_list: List of tokens in vocabulary (minus reserved tokens).
        frontend: Frontend module.
        specaug: SpecAugment module.
        normalize: Normalization module.
        encoder: Encoder module.
        decoder: Decoder module.
        joint_network: Joint Network module.
        transducer_weight: Weight of the Transducer loss.
        use_k2_pruned_loss: Whether to use k2 pruned Transducer loss.
        k2_pruned_loss_args: Arguments of the k2 loss pruned Transducer loss.
        warmup_steps: Number of steps in warmup, used for pruned loss scaling.
        validation_nstep: Maximum number of symbol expansions at each time step
                          when reporting CER or/and WER using mAES.
        fastemit_lambda: FastEmit lambda value.
        auxiliary_ctc_weight: Weight of auxiliary CTC loss.
        auxiliary_ctc_dropout_rate: Dropout rate for auxiliary CTC loss inputs.
        auxiliary_lm_loss_weight: Weight of auxiliary LM loss.
        auxiliary_lm_loss_smoothing: Smoothing rate for LM loss' label smoothing.
        ignore_id: Initial padding ID.
        sym_space: Space symbol.
        sym_blank: Blank Symbol.
        report_cer: Whether to report Character Error Rate during validation.
        report_wer: Whether to report Word Error Rate during validation.
        extract_feats_in_collect_stats: Whether to use extract_feats stats collection.

    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Encoder,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        window_size: int = 5,
        ot_weight: float = 0.5,
        prob_alignment_masking: float = 0.2,
        transducer_weight: float = 1.0,
        use_k2_pruned_loss: bool = False,
        k2_pruned_loss_args: Dict = {},
        warmup_steps: int = 25000,
        validation_nstep: int = 2,
        fastemit_lambda: float = 0.0,
        auxiliary_ctc_weight: float = 0.0,
        auxiliary_ctc_dropout_rate: float = 0.0,
        auxiliary_lm_loss_weight: float = 0.0,
        auxiliary_lm_loss_smoothing: float = 0.05,
        ignore_id: int = -1,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        report_cer: bool = False,
        report_wer: bool = False,
        extract_feats_in_collect_stats: bool = True,
    ) -> None:
        """Construct an ESPnetASRTransducerModel object."""
        super().__init__()

        # The following labels ID are reserved:
        #    - 0: Blank symbol.
        #    - 1: Unknown symbol.
        #    - vocab_size - 1: SOS/EOS symbol.
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.sym_space = sym_space
        self.sym_blank = sym_blank

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        self.encoder = encoder
        self.decoder = decoder
        self.joint_network = joint_network

        self.criterion_transducer = None
        self.error_calculator = None

        self.alignment_gate = torch.nn.Parameter(torch.tensor(0.0))  # 초기값 0으로 설정
        self.training_step = 0

        self.use_auxiliary_ctc = auxiliary_ctc_weight > 0
        self.use_auxiliary_lm_loss = auxiliary_lm_loss_weight > 0

        self.ot_weight = ot_weight
        self.window_size = window_size
        self.prob_alignment_masking = prob_alignment_masking

        if use_k2_pruned_loss:
            self.am_proj = torch.nn.Linear(
                encoder.output_size,
                vocab_size,
            )

            self.lm_proj = torch.nn.Linear(
                decoder.output_size,
                vocab_size,
            )

            self.warmup_steps = warmup_steps
            self.steps_num = -1

            self.k2_pruned_loss_args = k2_pruned_loss_args
            self.k2_loss_type = k2_pruned_loss_args.get("loss_type", "regular")

        self.use_k2_pruned_loss = use_k2_pruned_loss

        if self.use_auxiliary_ctc:
            self.ctc_lin = torch.nn.Linear(encoder.output_size, vocab_size)
            self.ctc_dropout_rate = auxiliary_ctc_dropout_rate

        if self.use_auxiliary_lm_loss:
            self.lm_lin = torch.nn.Linear(decoder.output_size, vocab_size)

            eps = auxiliary_lm_loss_smoothing / (vocab_size - 1)

            self.lm_loss_smooth_neg = eps
            self.lm_loss_smooth_pos = (1 - auxiliary_lm_loss_smoothing) + eps

        self.transducer_weight = transducer_weight
        self.fastemit_lambda = fastemit_lambda

        self.auxiliary_ctc_weight = auxiliary_ctc_weight
        self.auxiliary_lm_loss_weight = auxiliary_lm_loss_weight

        self.report_cer = report_cer
        self.report_wer = report_wer
        self.validation_nstep = validation_nstep

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        epoch=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward architecture and compute loss(es).

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)
            text: Label ID sequences. (B, L)
            text_lengths: Label ID sequences lengths. (B,)
            kwargs: Contains "utts_id".

        Return:
            loss: Main loss value.
            stats: Task statistics.
            weight: Task weights.

        """
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        batch_size = speech.shape[0]
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2. Transducer-related I/O preparation
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )

        # 3. Decoder
        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        # 4. Joint Network and RNNT loss computation
        if self.use_k2_pruned_loss:
            loss_trans = self._calc_k2_transducer_pruned_loss(
                encoder_out, decoder_out, text, t_len, u_len, **self.k2_pruned_loss_args
            )
        else:
            joint_out = self.joint_network(
                encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
            )
            
            if self.training:
                loss_ot = self._calc_wasserstein_loss(
                    encoder_out,
                    decoder_out,
                )
                loss_trans = self._calc_transducer_loss(
                    encoder_out,
                    joint_out,
                    target,
                    t_len,
                    u_len,
                )

                # if epoch is not None and epoch > 10:
                #     loss_trans = self._calc_rott_loss(
                #         encoder_out,
                #         decoder_out,
                #         joint_out,
                #         target,
                #         t_len,
                #         u_len,
                #     )
                # else:
                #     loss_trans = self._calc_transducer_loss(
                #     encoder_out,
                #     joint_out,
                #     target,
                #     t_len,
                #     u_len,
                # )
            else:
                loss_trans = self._calc_transducer_loss(
                    encoder_out,
                    joint_out,
                    target,
                    t_len,
                    u_len,
                )

        # 5. Auxiliary losses
        loss_ctc, loss_lm = 0.0, 0.0

        if self.use_auxiliary_ctc:
            loss_ctc = self._calc_ctc_loss(
                encoder_out,
                target,
                t_len,
                u_len,
            )

        if self.use_auxiliary_lm_loss:
            loss_lm = self._calc_lm_loss(decoder_out, target)


        if self.training:
            loss = ((1 - self.ot_weight) * (
                self.transducer_weight * loss_trans
                + self.auxiliary_ctc_weight * loss_ctc
                + self.auxiliary_lm_loss_weight * loss_lm
            )) + (self.ot_weight * loss_ot)
        else:
            loss = (self.transducer_weight * loss_trans
                + self.auxiliary_ctc_weight * loss_ctc
                + self.auxiliary_lm_loss_weight * loss_lm)
            
        # 6. CER/WER computation.
        if not self.training and (self.report_cer or self.report_wer):
            if self.error_calculator is None:
                from espnet2.asr_transducer.error_calculator import ErrorCalculator

                if self.use_k2_pruned_loss and self.k2_loss_type == "modified":
                    self.validation_nstep = 1

                self.error_calculator = ErrorCalculator(
                    self.decoder,
                    self.joint_network,
                    self.token_list,
                    self.sym_space,
                    self.sym_blank,
                    nstep=self.validation_nstep,
                    report_cer=self.report_cer,
                    report_wer=self.report_wer,
                )

            cer_transducer, wer_transducer = self.error_calculator(
                encoder_out, target, t_len
            )
        else:
            cer_transducer, wer_transducer = None, None

        stats = dict(
            loss=loss.detach(),
            loss_transducer=loss_trans.detach(),
            loss_ot=loss_ot.detach() if self.training else None,
            loss_aux_ctc=loss_ctc.detach() if loss_ctc > 0.0 else None,
            loss_aux_lm=loss_lm.detach() if loss_lm > 0.0 else None,
            cer_transducer=cer_transducer,
            wer_transducer=wer_transducer,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features sequences and features lengths sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)
            text: Label ID sequences. (B, L)
            text_lengths: Label ID sequences lengths. (B,)
            kwargs: Contains "utts_id".

        Return:
            {}: "feats": Features sequences. (B, T, D_feats),
                "feats_lengths": Features sequences lengths. (B,)

        """
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )

            feats, feats_lengths = speech, speech_lengths

        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder speech sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            encoder_out: Encoder outputs. (B, T, D_enc)
            encoder_out_lens: Encoder outputs lengths. (B,)

        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        encoder_out, encoder_out_lens = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features sequences and features sequences lengths.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            feats: Features sequences. (B, T, D_feats)
            feats_lengths: Features sequences lengths. (B,)

        """
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths

        return feats, feats_lengths


    def sinkhorn_knopp(self, cost_matrix, epsilon=1.0, max_iter=3):
        n, m = cost_matrix.shape
        u = torch.ones(n, device=cost_matrix.device) 
        v = torch.ones(m, device=cost_matrix.device) 

        K = torch.exp(-cost_matrix / epsilon)

        prev_u, prev_v = None, None
        for _ in range(max_iter):
            prev_u, prev_v = u.clone(), v.clone()
            u = 1.0 / (torch.matmul(K, v) + 1e-9)
            v = 1.0 / (torch.matmul(K.T, u) + 1e-9)
            
            # 변화량이 작으면 조기 종료
            if torch.norm(u - prev_u) < 1e-4 and torch.norm(v - prev_v) < 1e-4:
                break

        transport_plan = torch.matmul(torch.diag(u), torch.matmul(K, torch.diag(v)))
        return transport_plan

    def compute_cosine_cost_matrix(self, audio_features, text_features):
        audio_norm = F.normalize(audio_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        cosine_similarity = torch.matmul(audio_norm, text_norm.T)
        cost_matrix = 1 - cosine_similarity
        return cost_matrix

           
    def _calc_wasserstein_loss(self, audio_features, text_features, epsilon=1.0, max_iter=3):
        B, T, D = audio_features.size()
        _, U, _ = text_features.size()
        
        total_loss = 0.0
        plans = []

        mu = torch.full((T,), 1.0 / T, device=audio_features.device)

        for b in range(B):
            # 1) cost matrix
            cost = self.compute_cosine_cost_matrix(
                audio_features[b], text_features[b]
            )  # [T, U]

            # 2) K = exp(-cost/epsilon)
            K = torch.exp(-cost / epsilon)  # [T, U]

            # 3) enforce only source marginal:
            #    find u s.t. diag(u) K 1 = mu  => u = mu / (K @ 1)
            #    one‐shot or iterated
            u = torch.ones_like(mu)
            one_vec = torch.ones((U,), device=audio_features.device)
            for _ in range(max_iter):
                u = mu / (K @ one_vec + 1e-9)

            # 4) build relaxed plan
            plan = torch.diag(u) @ K  # [T, U]
            plans.append(plan.unsqueeze(0))

            # 5) compute loss = <plan, cost>
            total_loss = total_loss + torch.sum(plan * cost)

        plans = torch.cat(plans, dim=0)  # [B, T, U]
        loss_relaxed = total_loss / B
        
        return loss_relaxed
     
    # def create_causal_mask_window(self, audio_len, text_len, window, device):
    #     """
    #     audio_len: int, audio sequence 길이
    #     text_len: int, text sequence 길이
    #     window: int, 고정 window 크기 (허용할 과거 토큰의 최대 개수)
    #     device: torch.device, mask가 생성될 디바이스
        
    #     각 audio time step t에 대해,
    #     - 우선 linear mapping으로 현재 허용되는 텍스트 인덱스: j_allowed = floor((t+1)*text_len/audio_len)
    #     - 이후, [max(0, j_allowed - window), j_allowed) 범위의 토큰만 허용합니다.
    #     """
    #     mask = torch.zeros(audio_len, text_len, dtype=torch.bool, device=device)
    #     for t in range(audio_len):
    #         j_allowed = int((t + 1) * text_len / audio_len)  # 현재까지 허용되는 인덱스
    #         start_idx = max(0, j_allowed - window)
    #         mask[t, start_idx:j_allowed] = True
    #     return mask


    # def _calc_wasserstein_loss(self, audio_features, text_features, epsilon=1.0, max_iter=3, window=10):
    #     """
    #     audio_features: Tensor, shape = (batch_size, audio_len, feature_dim)
    #     text_features:  Tensor, shape = (batch_size, text_len, feature_dim)
    #     window:         int or None, 만약 정수면 각 audio time step에서 최근 window 크기의 text token만 고려
        
    #     반환:
    #     total_wasserstein_loss: 평균 Wasserstein 손실
    #     aligned_features: 정렬된 feature (batch_size, audio_len, text_len, feature_dim)
    #     """
    #     batch_size, audio_len, feature_dim = audio_features.size()
    #     _, text_len, _ = text_features.size()
        
    #     total_wasserstein_loss = 0.0
    #     aligned_features = []

    #     for i in range(batch_size):
    #         # cosine 기반 cost matrix 계산 (audio_len x text_len)
    #         cost_matrix = self.compute_cosine_cost_matrix(audio_features[i], text_features[i])
            
    #         # causal mask 적용: 현재 audio frame t에서는 미래 토큰을 배제하기 위해
    #         if window is not None:
    #             causal_mask = self.create_causal_mask_window(audio_len, text_len, window, audio_features.device)
    #             high_cost = 1e6  # 허용되지 않는 위치에 부여할 높은 비용
    #             cost_matrix = torch.where(causal_mask, cost_matrix, high_cost * torch.ones_like(cost_matrix))
            
    #         transport_plan = self.sinkhorn_knopp(cost_matrix, epsilon, max_iter)  # (audio_len, text_len)
            
    #         # 예시로, audio와 text feature를 결합한 aligned feature 계산 (원하는 방식으로 수정 가능)
    #         # aligned_feature = torch.einsum('tu,td,ud -> tud', transport_plan, audio_features[i], text_features[i])
    #         # aligned_features.append(aligned_feature)
            
    #         # Entropy regularization을 포함한 Wasserstein 손실 계산
    #         entropy_term = torch.sum(transport_plan * torch.log(transport_plan + 1e-9))
    #         wasserstein_loss = torch.sum(transport_plan * cost_matrix) - (epsilon * entropy_term)
    #         total_wasserstein_loss += wasserstein_loss

    #     total_wasserstein_loss /= batch_size
    #     # aligned_features = torch.stack(aligned_features, dim=0)  # (batch_size, audio_len, text_len, feature_dim)

    #     return total_wasserstein_loss
        
    def _calc_wasserstein_loss(
        self,
        audio_features: torch.Tensor,  # [B, T, D]
        text_features: torch.Tensor,   # [B, U, D]
        epsilon: float = 1.0,
        max_iter: int = 50,
        delta: float = 5.0,
        penalty: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conditional‐Monotonic (Relaxed) OT loss:
        - source marginal (rows) is uniform
        - target marginal (cols) is unconstrained (semi‐relaxed)
        - monotonicity is softly enforced via a penalty mask
        - entropy regularization with eps

        Args:
        audio_features: [B, T, D]
        text_features:  [B, U, D]
        epsilon:        entropic regularization coefficient
        max_iter:       Sinkhorn iterations to enforce row marginal
        delta:          monotonic allowance margin
        penalty:        cost penalty outside monotonic band

        Returns:
        loss_cmot: scalar relaxed CMOT loss
        plans:     [B, T, U] transport plans
        """
        B, T, _ = audio_features.size()
        _, U, _ = text_features.size()

        # uniform source marginal
        mu = torch.full((T,), 1.0 / T, device=audio_features.device)
        one_vec = torch.ones((U,), device=audio_features.device)

        total_loss = 0.0
        plans = []

        for b in range(B):
            # 1) raw cost: cosine distance
            C = self.compute_cosine_cost_matrix(
                audio_features[b], text_features[b]
            )  # [T, U]

            # 2) monotonic penalty mask
            mask = self.compute_monotonic_mask(T, U, delta, device=C.device)
            # mask has 0 where allowed, large penalty (1) where disallowed
            C_relaxed = C + penalty * mask

            # 3) entropic kernel
            K = torch.exp(-C_relaxed / epsilon)

            # 4) enforce only source marginal: diag(u) K 1 = mu
            u = torch.ones_like(mu)
            for _ in range(max_iter):
                u = mu / (K @ one_vec + 1e-9)

            # 5) build semi‐relaxed monotonic plan
            plan = torch.diag(u) @ K  # [T, U]
            plans.append(plan.unsqueeze(0))

            # 6) compute loss = <plan, cost>
            total_loss += torch.sum(plan * C_relaxed)

        plans = torch.cat(plans, dim=0)  # [B, T, U]
        loss_cmot = total_loss / B
        return loss_cmot, plans

    # def batch_relaxed_ot(self, audio_features, text_features):
    #     """
    #     Batch-wise relaxed OT from audio to text.
    #     Args:
    #         audio_features: Tensor of shape (B, T, D)
    #         text_features: Tensor of shape (B, U, D)
    #     Returns:
    #         transport_plan: Tensor of shape (B, T, U)
    #         alignment: Tensor of shape (B, T)
    #     """

    #     B, T, D = audio_features.size()
    #     _, U, D = text_features.size()

    #     # Cost matrix (B, T, U) using cosine distance
    #     cost_matrix = 1 - F.cosine_similarity(
    #         audio_features.unsqueeze(2),  # (B, T, 1, D)
    #         text_features.unsqueeze(1),   # (B, 1, U, D)
    #         dim=-1
    #     )

    #     # 각 audio 토큰에 대해 가장 가까운 text 토큰을 찾기
    #     nearest_text_indices = torch.argmin(cost_matrix, dim=-1)  # (B, T)

    #     # audio feature의 norm을 기준으로 질량(ms) 설정
    #     ms = torch.norm(audio_features, p=2, dim=-1)  # (B, T)

    #     # Transport plan 초기화
    #     transport_plan = torch.zeros(B, T, U, device=audio_features.device)

    #     # transport_plan에 질량 할당
    #     transport_plan.scatter_(-1, nearest_text_indices.unsqueeze(-1), ms.unsqueeze(-1))

    #     return transport_plan, nearest_text_indices

    # def _calc_rott_loss(
    #     self,
    #     encoder_out,
    #     decoder_out,
    #     joint_out,
    #     target,
    #     t_len,
    #     u_len
    # ):
    #     B, T, U, V = joint_out.size()
    #     # transport_plan : [B, T, U], alignment : [B, T]
    #     transport_plan, alignment = self.batch_relaxed_ot(encoder_out, decoder_out)
    #     alignment_mask = torch.zeros((B, T, U), dtype=torch.bool, device=joint_out.device)
        
    #     for b in range(B):
    #         for t in range(T):
    #             aligned_idx = alignment[b, t]
    #             left = max(aligned_idx - self.window_size, 0)
    #             right = min(aligned_idx + self.window_size - 1, U-1)
                
    #             alignment_range = torch.arange(left, right+1, device=joint_out.device)
    #             alignment_mask[b, t, alignment_range] = 1
        

    #     loss_mask = alignment_mask.unsqueeze(-1).expand(-1, -1, -1, V)

    #     # 각 audio frame별로 확률적 masking 여부 결정 (논문 방식)
    #     prob_mask = torch.rand((B, T, 1, 1), device=joint_out.device) < self.prob_alignment_masking
    #     final_mask = torch.where(prob_mask, loss_mask, torch.ones_like(loss_mask))

    #     # joint_out logits 단계에서 바로 masked_fill 적용 (ESPnet 방식)
    #     masked_joint_out = joint_out.float().masked_fill(~final_mask, -30.0)

    #     if self.criterion_transducer is None:
    #         try:
    #             from warprnnt_pytorch import RNNTLoss

    #             self.criterion_transducer = RNNTLoss(
    #                 reduction="mean",
    #                 fastemit_lambda=self.fastemit_lambda,
    #             )
    #         except ImportError:
    #             logging.error(
    #                 "warp-transducer was not installed. "
    #                 "Please consult the installation documentation."
    #             )
    #             exit(1)

    #     with autocast(False):
    #         loss_transducer = self.criterion_transducer(
    #             masked_joint_out,
    #             target,
    #             t_len,
    #             u_len,
    #         )
        

        # return loss_transducer

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        joint_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            joint_out: Joint Network output sequences (B, T, U, D_joint)
            target: Target label ID sequences. (B, L)
            t_len: Encoder output sequences lengths. (B,)
            u_len: Target label ID sequences lengths. (B,)

        Return:
            loss_transducer: Transducer loss value.

        """
        if self.criterion_transducer is None:
            try:
                from warprnnt_pytorch import RNNTLoss

                self.criterion_transducer = RNNTLoss(
                    reduction="mean",
                    fastemit_lambda=self.fastemit_lambda,
                )
            except ImportError:
                logging.error(
                    "warp-transducer was not installed. "
                    "Please consult the installation documentation."
                )
                exit(1)

        with autocast(False):
            loss_transducer = self.criterion_transducer(
                joint_out.float(),
                target,
                t_len,
                u_len,
            )

        return loss_transducer

    def _calc_k2_transducer_pruned_loss(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        labels: torch.Tensor,
        encoder_out_len: torch.Tensor,
        decoder_out_len: torch.Tensor,
        prune_range: int = 5,
        simple_loss_scaling: float = 0.5,
        lm_scale: float = 0.0,
        am_scale: float = 0.0,
        loss_type: str = "regular",
        reduction: str = "mean",
        padding_idx: int = 0,
    ) -> torch.Tensor:
        """Compute k2 pruned Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            decoder_out: Decoder output sequences. (B, T, D_dec)
            labels: Label ID sequences. (B, L)
            encoder_out_len: Encoder output sequences lengths. (B,)
            decoder_out_len: Target label ID sequences lengths. (B,)
            prune_range: How many tokens by frame are used compute the pruned loss.
            simple_loss_scaling: The weight to scale the simple loss after warm-up.
            lm_scale: The scale factor to smooth the LM part.
            am_scale: The scale factor to smooth the AM part.
            loss_type: Define the type of path to take for loss computation.
                         (Either 'regular', 'smoothed' or 'constrained')
            padding_idx: SOS/EOS + Padding index.

        Return:
            loss_transducer: Transducer loss value.

        """
        try:
            import k2

            if self.fastemit_lambda > 0.0:
                logging.info(
                    "Disabling FastEmit, it is not available with k2 Transducer loss. "
                    "Please see delay_penalty option instead."
                )
        except ImportError:
            logging.error(
                "k2 was not installed. Please consult the installation documentation."
            )
            exit(1)

        # Note (b-flo): We use a dummy scaling scheme until the training parts are
        # revised (in a short future).
        self.steps_num += 1

        if self.steps_num < self.warmup_steps:
            pruned_loss_scaling = 0.1 + 0.9 * (self.steps_num / self.warmup_steps)
            simple_loss_scaling = 1.0 - (
                (self.steps_num / self.warmup_steps) * (1.0 - simple_loss_scaling)
            )
        else:
            pruned_loss_scaling = 1.0

        labels_unpad = [y[y != self.ignore_id].tolist() for y in labels]

        target = k2.RaggedTensor(labels_unpad).to(decoder_out.device)
        target_padded = target.pad(mode="constant", padding_value=padding_idx)
        target_padded = target_padded.to(torch.int64)

        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = decoder_out_len
        boundary[:, 3] = encoder_out_len

        lm = self.lm_proj(decoder_out)
        am = self.am_proj(encoder_out)

        with autocast(False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm.float(),
                am.float(),
                target_padded,
                padding_idx,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                rnnt_type=loss_type,
                reduction=reduction,
                return_grad=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad,
            py_grad,
            boundary,
            prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            self.joint_network.lin_enc(encoder_out),
            self.joint_network.lin_dec(decoder_out),
            ranges,
        )

        joint_out = self.joint_network(am_pruned, lm_pruned, no_projection=True)

        with autocast(False):
            pruned_loss = k2.rnnt_loss_pruned(
                joint_out.float(),
                target_padded,
                ranges,
                padding_idx,
                boundary,
                rnnt_type=loss_type,
                reduction=reduction,
            )

        loss_transducer = (
            simple_loss_scaling * simple_loss + pruned_loss_scaling * pruned_loss
        )

        return loss_transducer

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)
            t_len: Encoder output sequences lengths. (B,)
            u_len: Target label ID sequences lengths. (B,)

        Return:
            loss_ctc: CTC loss value.

        """
        ctc_in = self.ctc_lin(
            torch.nn.functional.dropout(encoder_out, p=self.ctc_dropout_rate)
        )
        ctc_in = torch.log_softmax(ctc_in.transpose(0, 1), dim=-1)

        target_mask = target != 0
        ctc_target = target[target_mask].cpu()

        with torch.backends.cudnn.flags(deterministic=True):
            loss_ctc = torch.nn.functional.ctc_loss(
                ctc_in,
                ctc_target,
                t_len,
                u_len,
                zero_infinity=True,
                reduction="sum",
            )
        loss_ctc /= target.size(0)

        return loss_ctc

    def _calc_lm_loss(
        self,
        decoder_out: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LM loss (i.e.: Cross-entropy with smoothing).

        Args:
            decoder_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)

        Return:
            loss_lm: LM loss value.

        """
        batch_size = decoder_out.size(0)

        logp = torch.log_softmax(
            self.lm_lin(decoder_out[:, :-1, :]).view(-1, self.vocab_size),
            dim=1,
        )
        target = target.view(-1).type(torch.int64)
        ignore = (target == 0).unsqueeze(1)

        with torch.no_grad():
            true_dist = logp.clone().fill_(self.lm_loss_smooth_neg)

            true_dist.scatter_(1, target.unsqueeze(1), self.lm_loss_smooth_pos)

        loss_lm = torch.nn.functional.kl_div(logp, true_dist, reduction="none")
        loss_lm = loss_lm.masked_fill(ignore, 0).sum() / batch_size

        return loss_lm
