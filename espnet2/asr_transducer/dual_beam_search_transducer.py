"""Beam search for dual-transducer models with prefix-conditioned lower refinement."""

from typing import Any, Dict, List, Optional

import torch

from espnet2.asr_transducer.beam_search_transducer import (
    BeamSearchTransducer,
    Hypothesis,
)
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork


class DualBeamSearchTransducer(BeamSearchTransducer):
    """Dual-transducer beam search with hypothesis-prefix acoustic refinement.

    This keeps the original BeamSearchTransducer untouched and only extends the
    default search path used by the dual model. Each hypothesis prefix can be
    mapped to a refined encoder output through a prefix refiner callable.
    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        beam_size: int,
        prefix_refiner,
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
        super().__init__(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=beam_size,
            lm=lm,
            lm_weight=lm_weight,
            search_type=search_type,
            max_sym_exp=max_sym_exp,
            u_max=u_max,
            nstep=nstep,
            expansion_gamma=expansion_gamma,
            expansion_beta=expansion_beta,
            score_norm=score_norm,
            nbest=nbest,
            streaming=streaming,
        )
        self.prefix_refiner = prefix_refiner
        self.prefix_refiner_cache: Dict[str, torch.Tensor] = {}

        if search_type != "default":
            raise NotImplementedError(
                "DualBeamSearchTransducer currently supports only default beam search."
            )

    def reset_cache(self) -> None:
        super().reset_cache()
        self.prefix_refiner_cache = {}

    def _prefix_key(self, yseq: List[int]) -> str:
        return "_".join(map(str, yseq))

    def _get_prefix_enc_out(
        self,
        base_enc_out: torch.Tensor,
        yseq: List[int],
    ) -> torch.Tensor:
        key = self._prefix_key(yseq)
        if key not in self.prefix_refiner_cache:
            self.prefix_refiner_cache[key] = self.prefix_refiner(yseq)
        return self.prefix_refiner_cache[key]

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        beam_k = min(self.beam_size, (self.vocab_size - 1))
        max_t = len(enc_out)

        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            kept_hyps = [
                Hypothesis(
                    score=0.0,
                    yseq=[0],
                    dec_state=self.decoder.init_state(1),
                )
            ]

        for t in range(max_t):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state = self.decoder.score(
                    max_hyp.yseq,
                    max_hyp.dec_state,
                )

                hyp_enc_out = self._get_prefix_enc_out(enc_out, max_hyp.yseq)

                logp = torch.log_softmax(
                    self.joint_network(hyp_enc_out[t : t + 1, :], dec_out),
                    dim=-1,
                ).squeeze(0)
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq,
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    lm_scores, lm_state = self.lm.score(
                        torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:], device=self.decoder.device
                        ),
                        max_hyp.lm_state,
                        None,
                    )
                else:
                    lm_state = max_hyp.lm_state

                for candidate_logp, k in zip(*top_k):
                    score = max_hyp.score + float(candidate_logp)

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
