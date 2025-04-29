"""Search algorithms for Transducer models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.ot_joint_network import OTGuidedJointNetwork


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms.

    Args:
        score: Total log-probability.
        yseq: Label sequence as integer ID sequence.
        dec_state: RNN/MEGA Decoder state (None if Stateless).
        lm_state: RNNLM state. ((N, D_lm), (N, D_lm)) or None

    """

    score: float
    yseq: List[int]
    dec_state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES.

    Args:
        : Hypothesis dataclass arguments.
        dec_out: Decoder output sequence. (B, D_dec)
        lm_score: Log-probabilities of the LM for given label. (vocab_size)

    """

    dec_out: torch.Tensor = None
    lm_score: torch.Tensor = None


class BeamSearchTransducer:
    """Beam search implementation for Transducer.

    Args:
        decoder: Decoder module.
        joint_network: Joint network module.
        beam_size: Size of the beam.
        lm: LM module.
        lm_weight: LM weight for soft fusion.
        search_type: Search algorithm to use during inference.
        max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
        u_max: Maximum expected target sequence length. (ALSD)
        nstep: Number of maximum expansion steps at each time step. (mAES)
        expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
        expansion_beta:
             Number of additional candidates for expanded hypotheses selection. (mAES)
        score_norm: Normalize final scores by length.
        nbest: Number of final hypothesis.
        streaming: Whether to perform chunk-by-chunk beam search.

    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: OTGuidedJointNetwork,
        beam_size: int,
        lm: Optional[torch.nn.Module] = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 3,
        u_max: int = 50,
        nstep: int = 2,
        expansion_gamma: float = 2.3,
        expansion_beta: int = 2,
        score_norm: bool = False,
        nbest: int = 1,
        streaming: bool = False,
    ) -> None:
        """Construct a BeamSearchTransducer object."""
        super().__init__()

        self.decoder = decoder
        self.joint_network = joint_network

        self.vocab_size = decoder.vocab_size

        assert beam_size <= self.vocab_size, (
            "beam_size (%d) should be smaller than or equal to vocabulary size (%d)."
            % (
                beam_size,
                self.vocab_size,
            )
        )
        self.beam_size = beam_size

        if search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            assert max_sym_exp > 1, "max_sym_exp (%d) should be greater than one." % (
                max_sym_exp
            )
            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            assert not streaming, "ALSD is not available in streaming mode."

            assert u_max >= 0, "u_max should be a positive integer, a portion of max_T."
            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "maes":
            assert self.vocab_size >= beam_size + expansion_beta, (
                "beam_size (%d) + expansion_beta (%d) "
                " should be smaller than or equal to vocab size (%d)."
                % (beam_size, expansion_beta, self.vocab_size)
            )
            self.max_candidates = beam_size + expansion_beta

            self.nstep = nstep
            self.expansion_gamma = expansion_gamma

            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError(
                "Specified search type (%s) is not supported." % search_type
            )

        self.use_lm = lm is not None

        if self.use_lm:
            assert hasattr(lm, "rnn_type"), "Transformer LM is currently not supported."

            self.sos = self.vocab_size - 1

            self.lm = lm
            self.lm_weight = lm_weight

        self.score_norm = score_norm
        self.nbest = nbest

        self.reset_cache()

    def __call__(
        self,
        enc_out: torch.Tensor,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)
            is_final: Whether enc_out is the final chunk of data.

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        hyps = self.search_algorithm(enc_out)

        if is_final:
            self.reset_cache()

            return self.sort_nbest(hyps)

        self.search_cache = hyps

        return hyps

    def reset_cache(self) -> None:
        """Reset cache for streaming decoding."""
        self.decoder.score_cache = {}
        self.search_cache = None

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort in-place hypotheses by score or score given sequence length.

        Args:
            hyps: Hypothesis.

        Return:
            hyps: Sorted hypothesis.

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def recombine_hyps(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Recombine hypotheses with same label ID sequence.

        Args:
            hyps: Hypotheses.

        Returns:
            final: Recombined hypotheses.

        """
        final = {}

        for hyp in hyps:
            str_yseq = "_".join(map(str, hyp.yseq))

            if str_yseq in final:
                final[str_yseq].score = np.logaddexp(final[str_yseq].score, hyp.score)
            else:
                final[str_yseq] = hyp

        return [*final.values()]

    def select_k_expansions(
        self,
        hyps: List[ExtendedHypothesis],
        topk_idx: torch.Tensor,
        topk_logp: torch.Tensor,
    ) -> List[ExtendedHypothesis]:
        """Return K hypotheses candidates for expansion from a list of hypothesis.

        K candidates are selected according to the extended hypotheses probabilities
        and a prune-by-value method. Where K is equal to beam_size + beta.

        Args:
            hyps: Hypotheses.
            topk_idx: Indices of candidates hypothesis.
            topk_logp: Log-probabilities of candidates hypothesis.

        Returns:
            k_expansions: Best K expansion hypotheses candidates.

        """
        k_expansions = []

        for i, hyp in enumerate(hyps):
            hyp_i = [
                (int(k), hyp.score + float(v))
                for k, v in zip(topk_idx[i], topk_logp[i])
            ]
            k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

            k_expansions.append(
                sorted(
                    filter(
                        lambda x: (k_best_exp - self.expansion_gamma) <= x[1], hyp_i
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

        return k_expansions

    def create_lm_batch_inputs(self, hyps_seq: List[List[int]]) -> torch.Tensor:
        """Make batch of inputs with left padding for LM scoring.

        Args:
            hyps_seq: Hypothesis sequences.

        Returns:
            : Padded batch of sequences.

        """
        max_len = max([len(h) for h in hyps_seq])

        return torch.LongTensor(
            [[self.sos] + ([0] * (max_len - len(h))) + h[1:] for h in hyps_seq],
            device=self.decoder.device,
        )
        
        
    def compute_euclidean_cost_matrix(self, audio_features, text_features):
        return torch.cdist(audio_features, text_features, p=2)
    
    def compute_cosine_cost_matrix(self, audio_features, text_features):
        audio_norm = F.normalize(audio_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        cosine_similarity = torch.matmul(audio_norm, text_norm.T)
        cost_matrix = 1 - cosine_similarity
        return cost_matrix
    

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
        transport_plan = transport_plan / (transport_plan.sum(dim=-1, keepdim=True) + 1e-9)

        return transport_plan
    
    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation with OT-guided joint network.
        
        이 예시는 각 encoder timestep에서, OT-guided joint network를 통해 
        OT 정렬 정보를 반영한 logits를 산출하여 beam search를 수행합니다.
        
        Args:
            enc_out: Encoder output sequence. (T, D_enc)
        
        Returns:
            nbest_hyps: N-best hypothesis.
        """
        beam_k = min(self.beam_size, (self.vocab_size - 1))
        max_t = len(enc_out)  # encoder time steps

        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            kept_hyps = [
                Hypothesis(
                    score=0.0,
                    yseq=[0],
                    dec_state=self.decoder.init_state(1),
                    lm_state=None,
                )
            ]

        # Beam search loop over encoder time steps
        for t in range(max_t):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                # 선택된 가장 높은 score의 hypothesis
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                # 현재 partial hypothesis로 decoder score 계산
                dec_out, state = self.decoder.score(
                    max_hyp.yseq,
                    max_hyp.dec_state,
                )  
                # dec_out: [1, U, D_dec], U: 현재까지 생성된 token 수

                # --- OT-guided Joint Network 적용 ---
                # encoder frame: enc_out[t : t+1, :] -> shape: [1, D_enc]
                # Expand to [1, 1, D_enc] to represent T=1.
                enc_frame = enc_out[t : t + 1, :].unsqueeze(0)  # [1, 1, D_enc]
                # dec_out: [1, U, D_dec] 그대로 사용.
                # Compute cost matrix between enc_frame and dec_out.
                # 먼저 squeeze encoder frame to remove batch dimension if needed.
                # Here, for a single instance, we compute cost as:
                #   cost_matrix: [1, U] = 1 - cosine_similarity(enc_frame, each token in dec_out)
                # We'll compute it manually:
                enc_norm = F.normalize(enc_frame, p=2, dim=-1)   # [1, 1, D_enc]
                dec_norm = F.normalize(dec_out, p=2, dim=-1)       # [1, U, D_dec]
                # Since T=1, we compute cosine similarity along D dimension.
                # Result shape: [1, 1, U]
                cosine_sim = torch.bmm(enc_norm.view(1,1,-1), dec_norm.view(1, -1, dec_norm.size(-1)).transpose(1,2))
                # But easier: for T=1, flatten:
                cosine_sim = torch.matmul(enc_norm.squeeze(1), dec_norm.squeeze(0).T)  # [1, U]
                cost_matrix = 1 - cosine_sim  # [1, U]
                # Compute transport plan with Sinkhorn; note that our sinkhorn_knopp expects 2D input.
                transport_plan = self.sinkhorn_knopp(cost_matrix.squeeze(0), epsilon=self.epsilon, max_iter=self.max_iter)
                # transport_plan: [1, U]; add batch and T dimension: [1, 1, U]
                transport_plan = transport_plan.unsqueeze(0)

                # OT-guided Joint Network를 통해 logits 산출
                # self.ot_guided_joint는 미리 정의된 OTGuidedJointNetwork 모듈 (위 예시 코드 참고)
                logits = self.joint_network(enc_frame, dec_out, transport_plan)
                # logits: [1, 1, U, vocab_size]
                # For current time step, we use the joint output corresponding to T=1
                logit = logits.squeeze(1).squeeze(0)  # [U, vocab_size]

                # log softmax로 확률 산출
                logp = torch.log_softmax(logit, dim=-1)  # [U, vocab_size]

                # 첫 번째 후보는 blank token (index 0) -> no symbol expansion
                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0])),
                        yseq=max_hyp.yseq,
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    lm_scores, lm_state = self.lm.score(
                        torch.LongTensor([self.sos] + max_hyp.yseq[1:]).to(self.decoder.device),
                        max_hyp.lm_state,
                        None,
                    )
                else:
                    lm_state = max_hyp.lm_state

                # Expand non-blank token candidates
                top_k = logp[1:].topk(beam_k, dim=-1)
                for token_logp, k in zip(*top_k):
                    score = max_hyp.score + float(token_logp)
                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]
                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )

                if len(kept_most_prob) >= self.beam_size:
                    kept_hyps = kept_most_prob
                    break

            return kept_hyps
    
    def align_length_sync_decoding(
        self,
        enc_out: torch.Tensor,
    ) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoder output sequences. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        t_max = int(enc_out.size(0))
        u_max = min(self.u_max, (t_max - 1))

        B = [Hypothesis(yseq=[0], score=0.0, dec_state=self.decoder.init_state(1))]
        final = []

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for i in range(t_max + u_max):
            A = []

            B_ = []
            B_enc_out = []

            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u

                if t > (t_max - 1):
                    continue

                B_.append(hyp)
                B_enc_out.append((t, enc_out[t]))

            if B_:
                beam_enc_out = torch.stack([b[1] for b in B_enc_out])
                beam_dec_out, beam_state = self.decoder.batch_score(B_)

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(self.beam_size, dim=-1)

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        self.create_lm_batch_inputs([b.yseq for b in B_]),
                        [b.lm_state for b in B_],
                        None,
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if B_enc_out[i][0] == (t_max - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                            new_hyp.lm_state = beam_lm_states[i]

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam_size]
                B = self.recombine_hyps(B)

        if final:
            return final

        return B

    def time_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        if self.search_cache is not None:
            B = self.search_cache
        else:
            B = [
                Hypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.init_state(1),
                )
            ]

            if self.use_lm:
                B[0].lm_state = self.lm.zero_state()

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = enc_out_t.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_dec_out, beam_state = self.decoder.batch_score(C)

                beam_logp = torch.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(self.beam_size, dim=-1)

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                        )

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            self.create_lm_batch_inputs([c.yseq for c in C]),
                            [c.lm_state for c in C],
                            None,
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                                new_hyp.lm_state = beam_lm_states[i]

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[: self.beam_size]

            B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam_size]

        return B
        
    def modified_adaptive_expansion_search(self, enc_out: torch.Tensor) -> List[ExtendedHypothesis]:
        """Modified version of Adaptive Expansion Search (mAES) with OT-guided joint network.
        
        Args:
            enc_out: Encoder output sequence. (T, D_enc)
        
        Returns:
            nbest_hyps: N-best hypothesis.
        """
        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            init_tokens = [
                ExtendedHypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.init_state(1),
                )
            ]
            beam_dec_out, beam_state = self.decoder.batch_score(init_tokens)
            if self.use_lm:
                beam_lm_scores, beam_lm_states = self.lm.batch_score(
                    self.create_lm_batch_inputs([h.yseq for h in init_tokens]),
                    [h.lm_state for h in init_tokens],
                    None,
                )
                lm_state = beam_lm_states[0]
                lm_score = beam_lm_scores[0]
            else:
                lm_state = None
                lm_score = None
            kept_hyps = [
                ExtendedHypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.select_state(beam_state, 0),
                    dec_out=beam_dec_out[0],
                    lm_state=lm_state,
                    lm_score=lm_score,
                )
            ]
                
        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            enc_frame = enc_out_t.unsqueeze(0)  # [1, D_enc]

            list_b = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([h.dec_out for h in hyps])  # [num_hyps, D_dec]

                # --- 올바르게 수정된 OT-guided Joint Network 적용 ---
                # Cost matrix를 [num_hyps, 1]로 구성 (각 hypothesis별로 따로 처리)
                cost_matrix = self.compute_cosine_cost_matrix(enc_frame, beam_dec_out)  # (1, num_hyps)

                transport_plan = self.sinkhorn_knopp(cost_matrix, epsilon=1.0, max_iter=3)  # (1, num_hyps)
                aligned_dec_out = torch.matmul(transport_plan, beam_dec_out)

                # joint_network 호출 시 차원을 명확히 맞추기
                logits = self.joint_network(
                    enc_frame.unsqueeze(0),               # [1, D_enc]
                    beam_dec_out.unsqueeze(0),            # [num_hyps, D_dec]
                    aligned_dec_out.unsqueeze(0)          # [1, D_dec]
                )  # 결과: [1, 1, num_hyps, vocab_size]

                beam_logp, beam_idx = torch.log_softmax(logits.squeeze(0), dim=-1).topk(self.max_candidates, dim=-1)

                # 이후 k_expansion을 위해 차원 명확히 맞추기 위해서는
                beam_logp = beam_logp.squeeze(0)  # [num_hyps, max_candidates]
                beam_idx = beam_idx.squeeze(0)    # [num_hyps, max_candidates]
                k_expansions = self.select_k_expansions(hyps, beam_idx, beam_logp)
        
                list_exp = []
                for i, hyp in enumerate(hyps):
                    for k, new_score in k_expansions[i]:
                        new_hyp = ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=new_score,
                            dec_out=hyp.dec_out,
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_score=hyp.lm_score,
                        )
                        if k == 0:
                            list_b.append(new_hyp)
                        else:
                            new_hyp.yseq.append(int(k))
                            if self.use_lm:
                                new_hyp.score += self.lm_weight * float(hyp.lm_score[k])
                            list_exp.append(new_hyp)
        
                if not list_exp:
                    kept_hyps = sorted(self.recombine_hyps(list_b), key=lambda x: x.score, reverse=True)[: self.beam_size]
                    break
                else:
                    beam_dec_out, beam_state = self.decoder.batch_score(list_exp)
                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            self.create_lm_batch_inputs([h.yseq for h in list_exp]),
                            [h.lm_state for h in list_exp],
                            None,
                        )
                    if n < (self.nstep - 1):
                        for i, hyp in enumerate(list_exp):
                            hyp.dec_out = beam_dec_out[i]
                            hyp.dec_state = self.decoder.select_state(beam_state, i)
                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_score = beam_lm_scores[i]
                        hyps = list_exp[:]
                    else:
                        cost_matrix = self.compute_cosine_cost_matrix(enc_frame, beam_dec_out)  # (1, num_hyps)

                        transport_plan = self.sinkhorn_knopp(cost_matrix, epsilon=1.0, max_iter=3)  # (1, num_hyps)
                        aligned_dec_out = torch.matmul(transport_plan, beam_dec_out)

                        # joint_network 호출 시 차원을 명확히 맞추기
                        logits = self.joint_network(
                            enc_frame,               # [1, D_enc]
                            beam_dec_out,            # [num_hyps, D_dec]
                            aligned_dec_out          # [1, D_dec]
                        )  # 결과: [1, num_hyps, vocab_size]

                        beam_logp = torch.log_softmax(logits, dim=-1)

                        for i, hyp in enumerate(list_exp):
                            hyp.score += float(beam_logp[i, 0])
                            hyp.dec_out = beam_dec_out[i]
                            hyp.dec_state = self.decoder.select_state(beam_state, i)
                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_score = beam_lm_scores[i]

                        kept_hyps = sorted(
                            self.recombine_hyps(list_b + list_exp), 
                            key=lambda x: x.score, 
                            reverse=True
                        )[: self.beam_size]
            return kept_hyps