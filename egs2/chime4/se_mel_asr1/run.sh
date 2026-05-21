#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

SECONDS=0

stage=1
stop_stage=13
channel_mode="${CHANNEL_MODE:-allch}"
speed_perturb_factors="${SPEED_PERTURB_FACTORS:-0.9 1.0 1.1}"
mapper_config="${MAPPER_CONFIG:-conf/train_logmel_mapper.yaml}"
enh_tag="${ENH_TAG:-enh_train_cleanmel_mapper80}"
feature_suffix="${FEATURE_SUFFIX:-_mel80cleanmel}"
mapper_checkpoint="${MAPPER_CHECKPOINT:-}"
token_type="${TOKEN_TYPE:-char}"
use_lm="${USE_LM:-false}"
asr_config="${ASR_CONFIG:-conf/tuning/train_asr_dual_transducer_conformer_cleanmel.yaml}"
inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_transducer.yaml}"
inference_asr_model="${INFERENCE_ASR_MODEL:-valid.loss.ave_10best.pth}"
asr_pretrained="${ASR_PRETRAINED:-}"
ignore_init_mismatch="${IGNORE_INIT_MISMATCH:-false}"
asr_tag="${ASR_TAG:-train_dual_transducer_mel80cleanmel_char_sp}"
asr_task="${ASR_TASK:-asr_transducer}"
asr_stage="${ASR_STAGE:-3}"
asr_stop_stage="${ASR_STOP_STAGE:-13}"
ngpu="${NGPU:-4}"

. ./path.sh
. ./cmd.sh
. ../asr1/utils/parse_options.sh

log "$0 $*"

if [ "${channel_mode}" = allch ]; then
    base_train_set=tr05_multi_mixed_track
    valid_set=dt05_multi_mixed_track
    test_sets="et05_multi_mixed_track et05_simu_snr-5_allch_track et05_simu_snr0_allch_track et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    base_train_set=tr05_multi_isolated_1ch_track
    valid_set=dt05_multi_isolated_1ch_track
    test_sets="et05_multi_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

if [ -n "${speed_perturb_factors}" ]; then
    feature_train_set="${base_train_set}_sp${feature_suffix}"
else
    feature_train_set="${base_train_set}${feature_suffix}"
fi
feature_valid_set="${valid_set}${feature_suffix}"
feature_test_sets=""
for dset in ${test_sets}; do
    feature_test_sets+=" ${dset}${feature_suffix}"
done

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 1 ]; then
    enh_stage="${stage}"
    enh_stop_stage="$(min "${stop_stage}" 3)"
    ./enh.sh \
        --stage "${enh_stage}" \
        --stop_stage "${enh_stop_stage}" \
        --channel_mode "${channel_mode}" \
        --speed_perturb_factors "${speed_perturb_factors}" \
        --mapper_config "${mapper_config}" \
        --enh_tag "${enh_tag}" \
        --feature_suffix "${feature_suffix}" \
        --mapper_checkpoint "${mapper_checkpoint}" \
        --ngpu "${ngpu}"
fi

if [ "${stop_stage}" -ge 4 ]; then
    [ -f "../asr1/data/${feature_train_set}/feats.scp" ] || { log "Missing extracted features: ../asr1/data/${feature_train_set}/feats.scp"; exit 1; }
    if [ -n "${asr_pretrained}" ]; then
        [ -f "${asr_pretrained}" ] || { log "Missing ASR pretrained model: ${asr_pretrained}"; exit 1; }
    fi

    log "Stage 4: ASR training on clean-mel features (asr.sh stages ${asr_stage}-${asr_stop_stage})"
    cd ../asr1
    ./asr.sh \
        --asr_task "${asr_task}" \
        --ngpu "${ngpu}" \
        --stage "${asr_stage}" \
        --stop_stage "${asr_stop_stage}" \
        --lang en \
        --feats_type extracted \
        --token_type "${token_type}" \
        --use_lm "${use_lm}" \
        --inference_asr_model "${inference_asr_model}" \
        --pretrained_model "${asr_pretrained}" \
        --ignore_init_mismatch "${ignore_init_mismatch}" \
        --train_set "${feature_train_set}" \
        --valid_set "${feature_valid_set}" \
        --test_sets "${feature_test_sets}" \
        --bpe_train_text "data/${feature_train_set}/text" \
        --lm_train_text "data/${feature_train_set}/text" \
        --nlsyms_txt data/nlsyms.txt \
        --asr_config "${asr_config}" \
        --inference_config "${inference_config}" \
        --asr_tag "${asr_tag}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
