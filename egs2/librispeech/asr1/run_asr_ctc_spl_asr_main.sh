#!/usr/bin/env bash
set -e
set -u
set -o pipefail

train_set="train_960_noisy_randm5to15"
valid_set="dev_noisy_randm5to15"
test_sets="test_clean_noisy_snrm5 test_clean_noisy_snr0 test_clean_noisy_snr5 test_clean_noisy_snr10 test_clean_noisy_snr15 test_other_noisy_snrm5 test_other_noisy_snr0 test_other_noisy_snr5 test_other_noisy_snr10 test_other_noisy_snr15"

asr_task="asr"
asr_config="conf/tuning/SPL/train_asr_ctc_spl_asr.yaml"
inference_config="conf/tuning/decode_ctc_bs1.yaml"
inference_asr_model="valid.cer_ctc.ave_10best.pth"
speed_perturb_factors=""

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 16 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_normalize utterance_mvn \
    --asr_task "${asr_task}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
