#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ESPNET_CUDA_VISIBLE_DEVICES="${ESPNET_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export PYTHONPATH="/home/user/Workspace/espnet_bridge_shared_tta:${PYTHONPATH:-}"

./asr.sh \
    --stage 11 \
    --stop_stage 11 \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --asr_task asr_jepa \
    --feats_type raw \
    --feats_normalize null \
    --use_lm false \
    --use_clean_speech false \
    --asr_config conf/tuning/SPL/train_asr_ctc_spl_se960_latent_meta_bridge_exactmaml_a16_k1_ilr01_sanity.yaml \
    --train_set train_clean_100_noisy_randm5to15 \
    --valid_set dev_noisy_randm5to15 \
    --test_sets "test_clean_noisy_snr0" \
    --lm_train_text data/train_clean_100_noisy_randm5to15/text \
    --bpe_train_text data/train_clean_100_noisy_randm5to15/text \
    --asr_tag asr_ctc_spl_se960_latent_meta_bridge_exactmaml_a16_k1_ilr01_sanity \
    "$@"
