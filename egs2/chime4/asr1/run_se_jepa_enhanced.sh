#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

stage="${STAGE:-1}"
stop_stage="${STOP_STAGE:-13}"
ngpu="${NGPU:-4}"
token_type="${TOKEN_TYPE:-bpe}"
nbpe="${NBPE:-500}"
bpemode="${BPEMODE:-unigram}"
suffix="${SUFFIX:-enhsmall}"
speed_perturb_factors=""
asr_tag="${ASR_TAG:-train_asr_se_jepa_transducer_enhanced_en_bpe500}"
asr_config="${ASR_CONFIG:-conf/tuning/train_asr_se_jepa_transducer_enhanced.yaml}"
inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_transducer.yaml}"
inference_asr_model="${INFERENCE_ASR_MODEL:-valid.loss.ave_10best.pth}"
asr_args="${ASR_ARGS:-}"

. utils/parse_options.sh

train_set="tr05_simu_isolated_1ch_track_${suffix}"
valid_set="dt05_simu_isolated_1ch_track_${suffix}"
test_sets="et05_multi_isolated_1ch_track_${suffix} et05_simu_snr-5_isolated_1ch_track_${suffix} et05_simu_snr0_isolated_1ch_track_${suffix} et05_simu_snr5_isolated_1ch_track_${suffix} et05_simu_snr10_isolated_1ch_track_${suffix} et05_simu_snr15_isolated_1ch_track_${suffix}"

ensure_enhanced_data() {
    if [ ! -f "data/${train_set}/wav.scp" ] || [ ! -f "data/${valid_set}/wav.scp" ]; then
        ./local/data_from_enh1_enhanced.sh --suffix "${suffix}"
    fi
}

ensure_enhanced_data

./asr.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --lang en \
    --ngpu "${ngpu}" \
    --nj 8 \
    --gpu_inference false \
    --inference_nj 4 \
    --format_wav_scp_skip_bad_files true \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --token_type "${token_type}" \
    --nbpe "${nbpe}" \
    --bpemode "${bpemode}" \
    --max_wav_duration 30 \
    --audio_format wav \
    --feats_type raw \
    --use_lm false \
    --asr_task asr_transducer \
    --asr_config "${asr_config}" \
    --asr_args "${asr_args}" \
    --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
