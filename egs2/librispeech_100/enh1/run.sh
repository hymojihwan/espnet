#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

# Enhancement-only training for LibriSpeech-100 noisy data
train_set=train_clean_100_noisy
valid_set=dev_clean_noisy
test_sets="test_clean_noisy test_other_noisy"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 1 \
    --ref_channel 0 \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"

