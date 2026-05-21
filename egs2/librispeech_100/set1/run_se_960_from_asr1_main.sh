#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Reuse set1 enhancement pipeline with 960h noisy sets prepared in egs2/librispeech/asr1/data
SRC_DATA=/home/user/Workspace/espnet/egs2/librispeech/asr1/data
DST_DATA=./data

sets=(
  train_960_noisy_randm5to15
  dev_noisy_randm5to15
  test_clean_noisy_snrm5
  test_clean_noisy_snr0
  test_clean_noisy_snr5
  test_clean_noisy_snr10
  test_clean_noisy_snr15
  test_other_noisy_snrm5
  test_other_noisy_snr0
  test_other_noisy_snr5
  test_other_noisy_snr10
  test_other_noisy_snr15
)

mkdir -p "${DST_DATA}"
for s in "${sets[@]}"; do
  if [ ! -d "${SRC_DATA}/${s}" ]; then
    echo "[ERROR] Missing source set: ${SRC_DATA}/${s}" >&2
    exit 1
  fi
  ln -sfn "${SRC_DATA}/${s}" "${DST_DATA}/${s}"
done

train_set="train_960_noisy_randm5to15"
valid_set="dev_noisy_randm5to15"
test_sets="test_clean_noisy_snrm5 test_clean_noisy_snr0 test_clean_noisy_snr5 test_clean_noisy_snr10 test_clean_noisy_snr15 test_other_noisy_snrm5 test_other_noisy_snr0 test_other_noisy_snr5 test_other_noisy_snr10 test_other_noisy_snr15"

./enh.sh \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --fs 16k \
  --ngpu 4 \
  --nj 16 \
  --inference_nj 16 \
  --gpu_inference true \
  --ref_num 1 \
  --ref_channel 0 \
  --audio_format flac.ark \
  --enh_config conf/tuning/train_enh_convtasnet_small_librispeech_960_stable.yaml \
  --enh_tag train_enh_convtasnet_small_librispeech960_raw \
  --use_dereverb_ref false \
  --use_noise_ref false \
  --inference_model "valid.loss.best.pth" \
  "$@"
