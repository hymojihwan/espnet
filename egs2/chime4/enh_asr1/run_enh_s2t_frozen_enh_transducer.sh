#!/usr/bin/env bash
# Frozen Conv-TasNet frontend + Conformer Transducer ASR for CHiME4.
#
# Stages:
#   0: prepare enh_asr1 data + BPE500 assets
#   1: collect stats
#   2: train
#
# Note:
#   Decoding is intentionally not wired here yet because
#   espnet2/bin/asr_transducer_inference.py does not support EnhS2TTask directly.
set -e
set -u
set -o pipefail

_script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${_script_dir}"
. ./path.sh
. ./cmd.sh

stage="${STAGE:-0}"
stop_stage="${STOP_STAGE:-2}"
ngpu="${NGPU:-4}"
nj="${NJ:-32}"
python=python3
enh_model_size="${ENH_MODEL_SIZE:-small}"
token_type="${TOKEN_TYPE:-bpe}"
nbpe="${NBPE:-500}"
bpemode="${BPEMODE:-unigram}"
channel_mode="${CHANNEL_MODE:-ch1}"
speed_perturb_factors="${SPEED_PERTURB_FACTORS:-0.9 1.0 1.1}"

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

if [ "${stop_stage}" -gt 2 ]; then
    log "ERROR: transducer frozen-enh wrapper currently supports only stage 0..2."
    log "ERROR: decode/score are not wired for EnhS2TTask transducer yet."
    exit 1
fi

if [ "${token_type}" != bpe ]; then
    log "ERROR: this wrapper is intended for BPE transducer training. Set TOKEN_TYPE=bpe."
    exit 1
fi

if [ "${channel_mode}" = allch ]; then
    train_set=tr05_simu_allch_track
    valid_set=dt05_simu_allch_track
else
    train_set=tr05_simu_isolated_1ch_track
    valid_set=dt05_simu_isolated_1ch_track
fi
base_train_set="${train_set}"

case "${enh_model_size}" in
    small)
        enh1_exp_dir="${_script_dir}/../enh1/exp/enh_train_enh_convtasnet_small_raw"
        config=conf/tuning/train_enh_s2t_frozen_enh_transducer_small.yaml
        stats_dir="exp/enh_s2t_frozen_enh_transducer_stats_bpe${nbpe}_small"
        expdir="exp/enh_s2t_frozen_enh_transducer_conformer_bpe${nbpe}_small"
        ;;
    *)
        log "ERROR: unsupported ENH_MODEL_SIZE=${enh_model_size} (expected: small)"
        exit 1
        ;;
esac

ENH_PRETRAINED="${ENH_PRETRAINED:-${enh1_exp_dir}/valid.loss.ave_1best.pth}"
token_dir="${_script_dir}/data/en_token_list/bpe_${bpemode}${nbpe}"
TOKEN_LIST="${TOKEN_LIST:-${token_dir}/tokens.txt}"
BPEMODEL="${BPEMODEL:-${token_dir}/bpe.model}"

if [ -n "${speed_perturb_factors}" ]; then
    stats_dir+="_sp"
    expdir+="_sp"
fi

ensure_data() {
    if [ ! -f "data/${base_train_set}/wav.scp" ] || [ ! -f "data/${base_train_set}/text_spk1" ]; then
        log "Stage 0: prepare enh_asr1 data"
        ./local/data.sh ${LOCAL_DATA_OPTS:-}
    fi
}

ensure_bpe_assets() {
    if [ -f "${TOKEN_LIST}" ] && [ -f "${BPEMODEL}" ]; then
        return 0
    fi

    local bpeprefix train_txt opts_spm bpe_nlsyms_list
    mkdir -p "${token_dir}"
    bpeprefix="${token_dir}/bpe"
    train_txt="${token_dir}/train.txt"

    log "Generate BPE${nbpe} assets -> ${token_dir}"
    cut -f 2- -d " " "data/${base_train_set}/text_spk1" > "${train_txt}"

    if [ -f "data/nlsyms.txt" ] && [ -s "data/nlsyms.txt" ]; then
        bpe_nlsyms_list="$(awk '{print $1}' data/nlsyms.txt | paste -s -d, -)"
        opts_spm="--user_defined_symbols=${bpe_nlsyms_list}"
    else
        opts_spm=""
    fi

    spm_train \
        --input="${train_txt}" \
        --vocab_size="${nbpe}" \
        --model_type="${bpemode}" \
        --model_prefix="${bpeprefix}" \
        --character_coverage=1.0 \
        --input_sentence_size=100000000 \
        ${opts_spm}

    {
        echo "<blank>"
        echo "<unk>"
        awk '{ if (NR != 1 && NR != 2 && NR != 3) { print $1; } }' "${bpeprefix}.vocab"
        echo "<sos/eos>"
    } > "${TOKEN_LIST}"
}

prepare_speed_perturb() {
    train_set="${base_train_set}"
    if [ -z "${speed_perturb_factors}" ]; then
        return 0
    fi

    local format_wav_scp="${_script_dir}/../../TEMPLATE/asr1/scripts/audio/format_wav_scp.sh"
    local perturb_script="${_script_dir}/../../TEMPLATE/asr1/scripts/utils/perturb_enh_data_dir_speed.sh"
    local scp_list extra_files utt_extra_files dirs factor scp_name stem scp_nj src_scp

    if [ ! -f "${format_wav_scp}" ] || [ ! -f "${perturb_script}" ]; then
        log "ERROR: speed perturb helper scripts not found"
        exit 1
    fi

    if [ ! -f "data/${base_train_set}_sp/wav.scp" ] || [ ! -f "data/${base_train_set}_sp/text_spk1" ]; then
        log "Stage 0.5: speed perturbation -> data/${base_train_set}_sp"
        scp_list="wav.scp spk1.scp"
        extra_files="spk1.scp text_spk1 utt2lang"
        utt_extra_files="text_spk1 utt2lang"
        dirs=""
        rm -rf "data/${base_train_set}_sp"
        if [ -f "data/${base_train_set}/noise1.scp" ]; then
            scp_list+=" noise1.scp"
            extra_files+=" noise1.scp"
        fi

        for factor in ${speed_perturb_factors}; do
            if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                rm -rf "data/${base_train_set}_sp${factor}"
                "${perturb_script}" \
                    --utt_extra_files "${utt_extra_files}" \
                    "${factor}" \
                    "data/${base_train_set}" \
                    "data/${base_train_set}_sp${factor}" \
                    "${scp_list}"
                dirs+="data/${base_train_set}_sp${factor} "
            else
                dirs+="data/${base_train_set} "
            fi
        done

        # shellcheck disable=SC2086
        utils/combine_data.sh --extra-files "${extra_files}" "data/${base_train_set}_sp" ${dirs}
        utils/fix_data_dir.sh "data/${base_train_set}_sp" >/dev/null
    fi

    for scp_name in wav.scp spk1.scp noise1.scp; do
        [ -f "data/${base_train_set}_sp/${scp_name}" ] || continue
        if grep -q '|' "data/${base_train_set}_sp/${scp_name}"; then
            stem="${scp_name%.scp}"
            scp_nj="$(min "${nj}" "$(<"data/${base_train_set}_sp/${scp_name}" wc -l)")"
            log "Materialize ${scp_name} for speed-perturbed train set"
            src_scp="data/${base_train_set}_sp/${scp_name}.src"
            cp "data/${base_train_set}_sp/${scp_name}" "${src_scp}"
            "${format_wav_scp}" \
                --nj "${scp_nj}" \
                --cmd "${train_cmd}" \
                --audio-format wav \
                --fs 16k \
                --out-filename "${scp_name}" \
                "${src_scp}" \
                "data/${base_train_set}_sp" \
                "data/${base_train_set}_sp/logs/${stem}" \
                "data/${base_train_set}_sp/data/${stem}"
            rm -f "${src_scp}"
        fi
    done

    utils/fix_data_dir.sh "data/${base_train_set}_sp" >/dev/null
    train_set="${base_train_set}_sp"
}

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    ensure_data
    ensure_bpe_assets
    prepare_speed_perturb
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 1 ]; then
    ensure_data
    ensure_bpe_assets
    prepare_speed_perturb
fi

if [ ! -f "${ENH_PRETRAINED}" ]; then
    log "ERROR: enhancement checkpoint not found: ${ENH_PRETRAINED}"
    exit 1
fi

if [ ! -f "${TOKEN_LIST}" ] || [ ! -f "${BPEMODEL}" ]; then
    log "ERROR: BPE assets not found: ${TOKEN_LIST} / ${BPEMODEL}"
    exit 1
fi

if [ -f "data/nlsyms.txt" ]; then
    _nlsyms_opts=(--non_linguistic_symbols data/nlsyms.txt)
else
    _nlsyms_opts=()
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
        --bpemodel "${BPEMODEL}" \
        --token_list "${TOKEN_LIST}" \
        "${_nlsyms_opts[@]}" \
        --cleaner none \
        --g2p none \
        --text_name text_spk1 \
        --train_data_path_and_name_and_type "data/${train_set}/wav.scp,speech,sound" \
        --train_data_path_and_name_and_type "data/${train_set}/spk1.scp,speech_ref1,sound" \
        --train_data_path_and_name_and_type "data/${train_set}/text_spk1,text_spk1,text" \
        --valid_data_path_and_name_and_type "data/${valid_set}/wav.scp,speech,sound" \
        --valid_data_path_and_name_and_type "data/${valid_set}/spk1.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/${valid_set}/text_spk1,text_spk1,text" \
        --train_shape_file "${_logdir}/train.JOB.scp" \
        --valid_shape_file "${_logdir}/valid.JOB.scp" \
        --output_dir "${_logdir}/stats.JOB"

    _agg=""
    for i in $(seq "${_nj}"); do
        _agg+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_agg} --output_dir "${stats_dir}"

    _ntok="$(<"${TOKEN_LIST}" wc -l)"
    for x in train valid; do
        awk -v N="${_ntok}" '{ print $0 "," N }' "${stats_dir}/${x}/text_spk1_shape" \
            > "${stats_dir}/${x}/text_spk1_shape.${token_type}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: train -> ${expdir}"
    _fold=80000
    _train_args=(
        --use_preprocessor true
        --config "${config}"
        --token_type "${token_type}"
        --bpemodel "${BPEMODEL}"
        --token_list "${TOKEN_LIST}"
        --cleaner none
        --g2p none
        --text_name text_spk1
        --train_data_path_and_name_and_type "data/${train_set}/wav.scp,speech,sound"
        --train_data_path_and_name_and_type "data/${train_set}/spk1.scp,speech_ref1,sound"
        --train_data_path_and_name_and_type "data/${train_set}/text_spk1,text_spk1,text"
        --valid_data_path_and_name_and_type "data/${valid_set}/wav.scp,speech,sound"
        --valid_data_path_and_name_and_type "data/${valid_set}/spk1.scp,speech_ref1,sound"
        --valid_data_path_and_name_and_type "data/${valid_set}/text_spk1,text_spk1,text"
        --train_shape_file "${stats_dir}/train/speech_shape"
        --train_shape_file "${stats_dir}/train/speech_ref1_shape"
        --train_shape_file "${stats_dir}/train/text_spk1_shape.${token_type}"
        --valid_shape_file "${stats_dir}/valid/speech_shape"
        --valid_shape_file "${stats_dir}/valid/speech_ref1_shape"
        --valid_shape_file "${stats_dir}/valid/text_spk1_shape.${token_type}"
        --fold_length "${_fold}"
        --fold_length "${_fold}"
        --fold_length 400
        --output_dir "${expdir}"
        --init_param "${ENH_PRETRAINED}::enh_model"
        --freeze_param enh_model
        --ignore_init_mismatch true
    )
    if [ "${#_nlsyms_opts[@]}" -gt 0 ]; then
        _train_args+=("${_nlsyms_opts[@]}")
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
