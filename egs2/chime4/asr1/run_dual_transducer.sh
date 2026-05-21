#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

channel_mode="${CHANNEL_MODE:-ch1}"
dual_stage="${DUAL_STAGE:-joint}"
extra_annotations="${EXTRA_ANNOTATIONS:-/DB/CHiME4/data/annotations}"
simulate_stage="${SIMULATE_STAGE:-1}"
simulate_stop_stage="${SIMULATE_STOP_STAGE:-2}"
stage="${STAGE:-1}"
stop_stage="${STOP_STAGE:-}"

. utils/parse_options.sh

if [ "${channel_mode}" = allch ]; then
    default_train_set=tr05_simu_allch_track
    default_valid_set=dt05_simu_allch_track
    default_test_sets="et05_simu_allch_track et05_simu_snr-5_allch_track et05_simu_snr0_allch_track et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    if [ "${dual_stage}" = upper ]; then
        default_train_set=tr05_multi_noisy
        default_valid_set=dt05_multi_isolated_1ch_track
    else
        default_train_set=tr05_simu_isolated_1ch_track
        default_valid_set=dt05_simu_isolated_1ch_track
    fi
    default_test_sets="dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

train_set="${TRAIN_SET:-${default_train_set}}"
valid_set="${VALID_SET:-${default_valid_set}}"
test_sets="${TEST_SETS:-${default_test_sets}}"

asr_task=asr_transducer
token_type="${TOKEN_TYPE:-char}"
nbpe="${NBPE:-500}"
bpemode="${BPEMODE:-unigram}"
speed_perturb_factors="${SPEED_PERTURB_FACTORS:-}"
ngpu="${NGPU:-1}"
if [ -n "${ASR_CONFIG:-}" ]; then
    asr_config="${ASR_CONFIG}"
else
    case "${dual_stage}" in
        lower)
            asr_config="conf/tuning/train_asr_dual_transducer_stage1_lower.yaml"
            ;;
        upper)
            asr_config="conf/tuning/train_asr_dual_transducer_stage2_upper.yaml"
            ;;
        joint)
            asr_config="conf/tuning/train_asr_dual_transducer_conformer_raw.yaml"
            ;;
        *)
            echo "[run_dual_transducer.sh] unsupported DUAL_STAGE=${dual_stage}" >&2
            exit 1
            ;;
    esac
fi
inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_transducer.yaml}"
inference_asr_model="${INFERENCE_ASR_MODEL:-valid.loss.ave_10best.pth}"
if [ -n "${ASR_TAG:-}" ]; then
    asr_tag="${ASR_TAG}"
else
    case "${dual_stage}" in
        lower)
            asr_tag="train_asr_dual_transducer_stage1_lower_raw"
            ;;
        upper)
            asr_tag="train_asr_dual_transducer_stage2_upper_raw"
            ;;
        joint)
            asr_tag="train_asr_dual_transducer_conformer_raw"
            ;;
    esac
fi
asr_args="${ASR_ARGS:-}"
stage1_asr_tag="${STAGE1_ASR_TAG:-train_asr_dual_transducer_stage1_lower_raw}"
stage1_asr_model="${STAGE1_ASR_MODEL:-valid.loss.ave_10best.pth}"
init_from_stage1="${INIT_FROM_STAGE1:-true}"
lm_train_text="data/${train_set}/text"
if [ -f "data/local/other_text/text" ]; then
    lm_train_text="${lm_train_text} data/local/other_text/text"
fi

ensure_data() {
    local enh_train_src enh_valid_src
    if [ "${channel_mode}" = allch ]; then
        enh_train_src="../enh_asr1/data/tr05_simu_allch_track"
        enh_valid_src="../enh_asr1/data/dt05_simu_allch_track"
    else
        enh_train_src="../enh_asr1/data/tr05_simu_isolated_1ch_track"
        enh_valid_src="../enh_asr1/data/dt05_simu_isolated_1ch_track"
    fi

    if [ ! -f "${enh_train_src}/wav.scp" ] || [ ! -f "${enh_train_src}/spk1.scp" ] || \
       [ ! -f "${enh_valid_src}/wav.scp" ] || [ ! -f "${enh_valid_src}/spk1.scp" ]; then
        echo "[run_dual_transducer.sh] prepare enh_asr1 data first (${extra_annotations})"
        (
            cd ../enh_asr1
            ./local/data.sh \
                --extra-annotations "${extra_annotations}" \
                --stage "${simulate_stage}" \
                --stop_stage "${simulate_stop_stage}"
        )
    fi

    if [ ! -f "data/${train_set}/wav.scp" ] || [ ! -f "data/${valid_set}/wav.scp" ]; then
        echo "[run_dual_transducer.sh] prepare asr1 data with clean references"
        ./local/data.sh --stage 0 --stop_stage 2
    elif [ "${dual_stage}" != "upper" ] && \
         { [ ! -f "data/${train_set}/clean_speech.scp" ] || [ ! -f "data/${valid_set}/clean_speech.scp" ]; }; then
        echo "[run_dual_transducer.sh] prepare paired clean references for ${dual_stage} stage"
        ./local/data.sh --stage 0 --stop_stage 2
    fi
}

ensure_data

if [ -z "${stop_stage}" ]; then
    if [ "${dual_stage}" = "lower" ]; then
        stop_stage=11
    else
        stop_stage=13
    fi
fi

if [ "${dual_stage}" = "upper" ] && [ "${init_from_stage1}" = "true" ]; then
    stage1_checkpoint="exp/asr_${stage1_asr_tag}/${stage1_asr_model}"
    if [ -f "${stage1_checkpoint}" ]; then
        asr_args="${asr_args} --init_param ${stage1_checkpoint}:lower_transducer:lower_transducer"
    else
        echo "[run_dual_transducer.sh] stage1 checkpoint not found: ${stage1_checkpoint}" >&2
        echo "[run_dual_transducer.sh] continuing upper-stage training without lower init" >&2
    fi
fi

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
    --audio_format flac.ark \
    --feats_type raw \
    --use_lm false \
    --asr_task "${asr_task}" \
    --asr_config "${asr_config}" \
    --asr_args "${asr_args}" \
    --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "${lm_train_text}" "$@"
