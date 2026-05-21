#!/usr/bin/env bash
# SE-JEPA ASR: Conv-TasNet (SI-SNR) + JEPA predictive + Conformer CTC
# Flow: Noisy wav → SE → enhanced wav → log-mel → JEPA → Conformer → CTC

set -e
set -u
set -o pipefail

train_set="train"
# train_set="train_clean_100_noisy"
valid_set="dev"
test_sets="test_clean test_clean_noisy test_other test_other_noisy"

asr_task="asr_se_jepa"
asr_config=conf/tuning/train_se_jepa_scratch.yaml
inference_config=conf/tuning/decode_jepa_ctc.yaml
inference_asr_model=valid.wer_ctc.ave_10best.pth

./asr.sh \
    --asr_task "${asr_task}" \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 8 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --asr_args "--unused_parameters true --use_amp true" \
    "$@"
