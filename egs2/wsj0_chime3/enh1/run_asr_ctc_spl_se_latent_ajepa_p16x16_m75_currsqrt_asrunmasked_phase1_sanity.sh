#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ESPNET_CUDA_VISIBLE_DEVICES="${ESPNET_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export PYTHONPATH="/home/user/Workspace/espnet_bridge_meta_tta:${PYTHONPATH:-}"

../../TEMPLATE/asr1/asr.sh \
    --expdir exp/-5to15 \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --token_type bpe \
    --nbpe 2048 \
    --bpemode unigram \
    --feats_type raw \
    --feats_normalize utterance_mvn \
    --audio_format flac \
    --use_lm false \
    --use_clean_speech false \
    --nlsyms_txt data/nlsyms.txt \
    --asr_task asr_jepa \
    --asr_config conf/new_SPL/CTC/train_asr_ctc_spl_se_latent_ajepa_p16x16_m75_currsqrt_asrunmasked_phase1_wsj0_sanity.yaml \
    --inference_config conf/tuning/decode_ctc_bs1.yaml \
    --asr_tag train_asr_ctc_spl_se_latent_ajepa_p16x16_m75_currsqrt_asrunmasked_phase1_wsj0_chime3_sanity_s0 \
    --speed_perturb_factors "" \
    --stage 11 \
    --stop_stage 11 \
    --train_set train \
    --valid_set valid \
    --test_sets "test_0" \
    "$@"
