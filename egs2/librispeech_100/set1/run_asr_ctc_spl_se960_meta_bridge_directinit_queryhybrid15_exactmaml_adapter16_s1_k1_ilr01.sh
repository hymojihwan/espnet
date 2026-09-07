#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,2,3
export ESPNET_CUDA_VISIBLE_DEVICES=0,2,3

./asr.sh \
    --stage 11 \
    --stop_stage 13 \
    --lang en \
    --ngpu 3 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 3 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --asr_task asr_jepa \
    --feats_type raw \
    --feats_normalize null \
    --use_lm false \
    --use_clean_speech false \
    --asr_config conf/tuning/SPL/train_asr_ctc_spl_se960_meta_bridge_directinit_queryhybrid15_exactmaml_adapter16_s1_k1_ilr01.yaml \
    --inference_config conf/decode_asr_ctc_only.yaml \
    --inference_asr_model valid.cer_ctc.ave_10best.pth \
    --train_set train_clean_100_noisy_randm5to15 \
    --valid_set dev_noisy_randm5to15 \
    --test_sets "test_clean_noisy_snrm5 test_clean_noisy_snr0 test_clean_noisy_snr5 test_clean_noisy_snr10 test_clean_noisy_snr15 test_other_noisy_snrm5 test_other_noisy_snr0 test_other_noisy_snr5 test_other_noisy_snr10 test_other_noisy_snr15" \
    --lm_train_text data/train_clean_100_noisy_randm5to15/text \
    --bpe_train_text data/train_clean_100_noisy_randm5to15/text \
    --asr_tag asr_ctc_spl_se960_meta_bridge_directinit_queryhybrid15_exactmaml_adapter16_s1_k1_ilr01_main \
    "$@"
