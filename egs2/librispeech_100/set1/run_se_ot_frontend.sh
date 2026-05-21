#!/usr/bin/env bash
set -e
set -u
set -o pipefail

sample_rate=16k
train_set="train_clean_100_noisy_randm5to15"
valid_set="dev_noisy_randm5to15"
test_sets="test_clean_noisy_snrm5 test_clean_noisy_snr0 test_clean_noisy_snr5 test_clean_noisy_snr10 test_clean_noisy_snr15 test_other_noisy_snrm5 test_other_noisy_snr0 test_other_noisy_snr5 test_other_noisy_snr10 test_other_noisy_snr15"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
    --gpu_inference true \
    --ref_num 1 \
    --ref_channel 0 \
    --enh_config conf/tuning/train_enh_se_ot_frontend.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
