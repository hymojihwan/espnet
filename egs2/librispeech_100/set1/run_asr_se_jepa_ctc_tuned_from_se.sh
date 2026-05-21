#!/usr/bin/env bash
set -e
set -u
set -o pipefail

train_set="train_clean_100_noisy_randm5to15"
valid_set="dev_noisy_randm5to15"
test_sets="test_clean_noisy_snrm5 test_clean_noisy_snr0 test_clean_noisy_snr5 test_clean_noisy_snr10 test_clean_noisy_snr15 test_other_noisy_snrm5 test_other_noisy_snr0 test_other_noisy_snr5 test_other_noisy_snr10 test_other_noisy_snr15"

asr_task="asr_jepa"
asr_config="conf/tuning/train_asr_se_jepa_ctc_tuned_from_se.yaml"
inference_config="conf/tuning/decode_ctc_bs1.yaml"
inference_asr_model="valid.cer_ctc.ave_10best.pth"
pretrained_model="exp/asr_asr_ctc_frozen_se_fast70_main/valid.cer_ctc.ave_10best.pth"
decode_args=""
speed_perturb_factors=""

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --decode_args)
            decode_args="$2"
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 16 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn \
    --asr_task "${asr_task}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --pretrained_model "${pretrained_model}" \
    --ignore_init_mismatch true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    ${decode_args:+--inference_args "${decode_args}"} \
    "${args[@]}"
