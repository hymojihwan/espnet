#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"

enh_asr_task="enh_asr_transducer"
enh_asr_config=conf/tuning/enh_asr_transducer/conv_tasnet_se_only.yaml
inference_config=conf/tuning/enh_asr_transducer/decode_enh_asr_transducer.yaml
inference_enh_asr_model=valid.loss_enh.ave_10best.pth
pretrained_asr_model=../asr1/exp/asr_conformer-rnnt_raw_en_bpe2048_sp/valid.loss.ave_10best.pth

echo "=== Starting SE-Only Training (ASR Frozen) ==="
echo "Config: ${enh_asr_config}"
echo "ASR Model: ${pretrained_asr_model} (will be frozen)"
echo "Training Set: ${train_set}"
echo "Validation Set: ${valid_set}"

./enh_asr.sh \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 16 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --inference_enh_asr_model "${inference_enh_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --pretrained_asr_model "${pretrained_asr_model}" "$@"

echo "=== SE-Only Training Completed ==="
echo "Model saved as: ${inference_enh_asr_model}" 