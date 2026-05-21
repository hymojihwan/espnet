#!/usr/bin/env bash
# Frozen SE frontend + Transducer ASR (non-k2) for LibriSpeech-100 set1.
set -e
set -u
set -o pipefail

_script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${_script_dir}"
. ./path.sh
. ./cmd.sh

stage="${STAGE:-1}"
stop_stage="${STOP_STAGE:-2}"
ngpu="${NGPU:-4}"
nj="${NJ:-16}"
python=python3

train_set="train_clean_100_noisy_randm5to15"
valid_set="dev_noisy_randm5to15"
token_type="${TOKEN_TYPE:-bpe}"
nbpe="${NBPE:-2048}"
bpemode="${BPEMODE:-unigram}"

config="${CONFIG:-conf/tuning/train_enh_s2t_frozen_enh_transducer_nonk2.yaml}"
stats_dir="${STATS_DIR:-exp/enh_s2t_frozen_enh_transducer_stats_bpe${nbpe}}"
expdir="${EXPDIR:-exp/enh_s2t_frozen_enh_transducer_nonk2_bpe${nbpe}}"
enh_pretrained="${ENH_PRETRAINED:-exp/enh_train_enh_convtasnet_small_librispeech_raw/valid.loss.best.pth}"

. utils/parse_options.sh

log() {
    echo "[$(date '+%F %T')] $*"
}

min() {
    local a b
    a=$1
    shift
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

token_dir="data/en_token_list/bpe_${bpemode}${nbpe}"
token_list="${token_dir}/tokens.txt"
bpemodel="${token_dir}/bpe.model"

if [ ! -f "${enh_pretrained}" ]; then
    log "ERROR: enhancement checkpoint not found: ${enh_pretrained}"
    exit 1
fi
if [ ! -f "${token_list}" ] || [ ! -f "${bpemodel}" ]; then
    log "ERROR: token files not found: ${token_list} / ${bpemodel}"
    exit 1
fi

if [ -f "data/nlsyms.txt" ]; then
    nlsyms_opts=(--non_linguistic_symbols data/nlsyms.txt)
else
    nlsyms_opts=()
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: collect stats -> ${stats_dir}"
    _logdir="${stats_dir}/logdir"
    mkdir -p "${_logdir}"
    _nj="$(min "${nj}" "$(<data/${train_set}/wav.scp wc -l)" "$(<data/${valid_set}/wav.scp wc -l)")"

    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "data/${train_set}/wav.scp" ${split_scps}

    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "data/${valid_set}/wav.scp" ${split_scps}

    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.enh_s2t_train \
        --collect_stats true \
        --use_preprocessor true \
        --config "${config}" \
        --token_type "${token_type}" \
        --bpemodel "${bpemodel}" \
        --token_list "${token_list}" \
        "${nlsyms_opts[@]}" \
        --cleaner none \
        --g2p none \
        --text_name text \
        --train_data_path_and_name_and_type "data/${train_set}/wav.scp,speech,sound" \
        --train_data_path_and_name_and_type "data/${train_set}/spk1.scp,speech_ref1,sound" \
        --train_data_path_and_name_and_type "data/${train_set}/text,text,text" \
        --valid_data_path_and_name_and_type "data/${valid_set}/wav.scp,speech,sound" \
        --valid_data_path_and_name_and_type "data/${valid_set}/spk1.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/${valid_set}/text,text,text" \
        --train_shape_file "${_logdir}/train.JOB.scp" \
        --valid_shape_file "${_logdir}/valid.JOB.scp" \
        --output_dir "${_logdir}/stats.JOB"

    _agg=""
    for i in $(seq "${_nj}"); do
        _agg+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_agg} --output_dir "${stats_dir}"

    _ntok="$(<"${token_list}" wc -l)"
    for x in train valid; do
        awk -v N="${_ntok}" '{ print $0 "," N }' "${stats_dir}/${x}/text_shape" \
            > "${stats_dir}/${x}/text_shape.${token_type}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: train -> ${expdir}"
    _fold=80000
    _train_args=(
        --use_preprocessor true
        --config "${config}"
        --token_type "${token_type}"
        --bpemodel "${bpemodel}"
        --token_list "${token_list}"
        --cleaner none
        --g2p none
        --text_name text
        --train_data_path_and_name_and_type "data/${train_set}/wav.scp,speech,sound"
        --train_data_path_and_name_and_type "data/${train_set}/spk1.scp,speech_ref1,sound"
        --train_data_path_and_name_and_type "data/${train_set}/text,text,text"
        --valid_data_path_and_name_and_type "data/${valid_set}/wav.scp,speech,sound"
        --valid_data_path_and_name_and_type "data/${valid_set}/spk1.scp,speech_ref1,sound"
        --valid_data_path_and_name_and_type "data/${valid_set}/text,text,text"
        --train_shape_file "${stats_dir}/train/speech_shape"
        --train_shape_file "${stats_dir}/train/speech_ref1_shape"
        --train_shape_file "${stats_dir}/train/text_shape.${token_type}"
        --valid_shape_file "${stats_dir}/valid/speech_shape"
        --valid_shape_file "${stats_dir}/valid/speech_ref1_shape"
        --valid_shape_file "${stats_dir}/valid/text_shape.${token_type}"
        --fold_length "${_fold}"
        --fold_length "${_fold}"
        --fold_length 400
        --output_dir "${expdir}"
        --init_param "${enh_pretrained}::enh_model"
        --freeze_param enh_model
        --ignore_init_mismatch true
    )
    if [ "${#nlsyms_opts[@]}" -gt 0 ]; then
        _train_args+=("${nlsyms_opts[@]}")
    fi

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${expdir}/train.log" \
        --log "${expdir}/train.log" \
        --ngpu "${ngpu}" \
        --num_nodes 1 \
        --init_file_prefix "${expdir}/.dist_init_" \
        --multiprocessing_distributed true \
        -- \
        ${python} -m espnet2.bin.enh_s2t_train "${_train_args[@]}"
fi

log "Done."
