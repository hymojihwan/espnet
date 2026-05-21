#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
# train_set="train_clean_100"
# train_set="train_clean_100_noisy"
valid_set="dev"
test_sets="test_clean test_clean_noisy test_other test_other_noisy"

# Data directories
train_data_dir="data"
valid_data_dir="data"

# Use JEPA task
asr_task="asr_jepa"
asr_config=conf/tuning/train_jepa_balanced_scratch_no_specaug.yaml
# asr_config=conf/tuning/train_jepa_audio_scratch_no_specaug.yaml
# asr_config=conf/tuning/train_jepa_hybrid_scratch_no_specaug.yaml
# asr_config=conf/tuning/train_jepa_vit_scratch_no_specaug.yaml
# asr_config=conf/tuning/train_jepa_masked_scratch_no_specaug.yaml
# asr_config=conf/tuning/train_jepa_masked_scratch.yaml
inference_config=conf/tuning/decode_jepa_ctc.yaml
inference_asr_model=valid.wer_ctc.ave_10best.pth

# --speed_perturb_factors "0.9 1.0 1.1" \

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
    "$@"

