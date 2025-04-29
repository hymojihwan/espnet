#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

forced_alignments_path="data/train_clean_100/forced_alignments.npz"
asr_task=asr_ar_transducer

asr_config=conf/tuning/train_asr_transducer_conformer.yaml
inference_config=conf/decode_asr.yaml


./ar_rnnt_asr.sh \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --asr_task "${asr_task}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --forced_alignments "${forced_alignments_path}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
