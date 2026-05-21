#!/usr/bin/env bash
# Frozen Conv-TasNet frontend + ASR-pretrained Conformer-CTC fine-tuning for CHiME4.
set -e
set -u
set -o pipefail

_script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${_script_dir}"
. ./path.sh
. ./cmd.sh

stage="${STAGE:-0}"
stop_stage="${STOP_STAGE:-4}"
ngpu="${NGPU:-4}"
nj="${NJ:-32}"
inference_nj="${INFERENCE_NJ:-32}"
gpu_inference="${GPU_INFERENCE:-false}"
python=python3
enh_model_size="${ENH_MODEL_SIZE:-base}"
token_type="${TOKEN_TYPE:-char}"
nbpe="${NBPE:-500}"
bpemode="${BPEMODE:-unigram}"
use_lm="${USE_LM:-true}"
lm_weight_tag="${LM_WEIGHT_TAG:-lm03}"
channel_mode="${CHANNEL_MODE:-ch1}"
speed_perturb_factors="${SPEED_PERTURB_FACTORS:-0.9 1.0 1.1}"

. utils/parse_options.sh

if [ "${channel_mode}" = allch ]; then
    train_set=tr05_simu_allch_track
    valid_set=dt05_simu_allch_track
    test_sets="\
et05_simu_allch_track \
et05_simu_snr-5_allch_track et05_simu_snr0_allch_track \
et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track \
"
else
    train_set=tr05_simu_isolated_1ch_track
    valid_set=dt05_simu_isolated_1ch_track
    test_sets="\
dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track dt05_multi_isolated_1ch_track \
et05_real_isolated_1ch_track et05_simu_isolated_1ch_track et05_multi_isolated_1ch_track \
et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track \
"
fi
base_train_set="${train_set}"

if [ "${token_type}" = bpe ]; then
    token_tag="bpe${nbpe}"
else
    token_tag="${token_type}"
fi

case "${enh_model_size}" in
    small)
        enh1_exp_dir="${_script_dir}/../enh1/exp/enh_train_enh_convtasnet_small_raw"
        config=conf/tuning/train_enh_s2t_frozen_enh_conformer_ctc_small.yaml
        stats_dir="exp/enh_s2t_frozen_enh_init_asr_ctc_stats_${token_tag}_small"
        expdir="exp/enh_s2t_frozen_enh_init_asr_ctc_conformer_${token_tag}_small"
        ;;
    base)
        enh1_exp_dir="${_script_dir}/../enh1/exp/enh_train_enh_conv_tasnet_raw"
        config=conf/tuning/train_enh_s2t_frozen_enh_conformer_ctc.yaml
        stats_dir="exp/enh_s2t_frozen_enh_init_asr_ctc_stats_${token_tag}"
        expdir="exp/enh_s2t_frozen_enh_init_asr_ctc_conformer_${token_tag}"
        ;;
    *)
        echo "ERROR: unsupported ENH_MODEL_SIZE=${enh_model_size}" >&2
        exit 1
        ;;
esac

ENH_PRETRAINED="${ENH_PRETRAINED:-${enh1_exp_dir}/valid.loss.ave_1best.pth}"
ASR_PRETRAINED="${ASR_PRETRAINED:-${_script_dir}/../asr1/exp/asr_train_conformer_ctc_raw_en_char_sp/valid.cer_ctc.ave_10best.pth}"

if [ "${token_type}" = bpe ]; then
    token_dir="${_script_dir}/../asr1/data/en_token_list/bpe_${bpemode}${nbpe}"
    TOKEN_LIST="${TOKEN_LIST:-${token_dir}/tokens.txt}"
    BPEMODEL="${BPEMODEL:-${token_dir}/bpe.model}"
    LM_EXP="${LM_EXP:-${_script_dir}/../asr1/exp/lm_train_lm_transformer_en_bpe${nbpe}}"
    inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_asr_bs10_lm03.yaml}"
    inference_tag="decode_asr_${lm_weight_tag}_init_asr_model_valid.acc.ave_10best"
    inference_model=valid.acc.ave_10best.pth
else
    TOKEN_LIST="${TOKEN_LIST:-${_script_dir}/data/en_token_list/char/tokens.txt}"
    BPEMODEL=none
    LM_EXP="${LM_EXP:-${_script_dir}/../asr1/exp/lm_train_lm_transformer_en_char}"
    inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_ctc_bs1.yaml}"
    inference_tag="decode_ctc_bs1_init_asr_model_valid.cer_ctc.ave_10best"
    inference_model=valid.cer_ctc.ave_10best.pth
fi
LM_CONFIG_PATH="${LM_CONFIG_PATH:-${LM_EXP}/config.yaml}"
LM_FILE="${LM_FILE:-${LM_EXP}/valid.loss.ave_10best.pth}"

if [ -n "${speed_perturb_factors}" ]; then
    stats_dir+="_sp"
    expdir+="_sp"
fi

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

log() {
    echo "[$(date '+%F %T')] $*"
}

bool() {
    case "$1" in
        true|True|TRUE|1|yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

ensure_data() {
    if [ ! -f "data/${base_train_set}/wav.scp" ] || [ ! -f "data/${base_train_set}/text_spk1" ]; then
        log "Stage 0: prepare enh_asr1 data"
        ./local/data.sh ${LOCAL_DATA_OPTS:-}
    fi
}

ensure_char_token_list() {
    if [ "${token_type}" != char ]; then
        return 0
    fi
    if [ -f "${TOKEN_LIST}" ]; then
        return 0
    fi
    local token_dir
    token_dir="$(dirname "${TOKEN_LIST}")"
    mkdir -p "${token_dir}"
    log "Generate char token list -> ${TOKEN_LIST}"
    ${python} -m espnet2.bin.tokenize_text \
        --input "data/${base_train_set}/text_spk1" \
        --output "${TOKEN_LIST}" \
        --field 2- \
        --token_type char \
        --space_symbol "<space>" \
        --non_linguistic_symbols data/nlsyms.txt \
        --write_vocabulary true \
        --add_symbol "<blank>:0" \
        --add_symbol "<unk>:1" \
        --add_symbol "<sos/eos>:-1"
}

prepare_speed_perturb() {
    train_set="${base_train_set}"
    if [ -z "${speed_perturb_factors}" ]; then
        return 0
    fi

    local format_wav_scp="${_script_dir}/../../TEMPLATE/asr1/scripts/audio/format_wav_scp.sh"
    local perturb_script="${_script_dir}/../../TEMPLATE/asr1/scripts/utils/perturb_enh_data_dir_speed.sh"

    if [ ! -f "data/${base_train_set}_sp/wav.scp" ] || [ ! -f "data/${base_train_set}_sp/text_spk1" ]; then
        log "Stage 0.5: speed perturbation -> data/${base_train_set}_sp"
        local scp_list="wav.scp spk1.scp"
        local extra_files="spk1.scp text_spk1 utt2lang"
        local utt_extra_files="text_spk1 utt2lang"
        local dirs=""
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
            local stem="${scp_name%.scp}"
            local scp_nj
            local src_scp
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
    ensure_char_token_list
    prepare_speed_perturb
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 1 ]; then
    ensure_data
    ensure_char_token_list
    prepare_speed_perturb
fi

[ -f "${ENH_PRETRAINED}" ] || { log "ERROR: enhancement checkpoint not found: ${ENH_PRETRAINED}"; exit 1; }
[ -f "${ASR_PRETRAINED}" ] || { log "ERROR: ASR checkpoint not found: ${ASR_PRETRAINED}"; exit 1; }
[ -f "${TOKEN_LIST}" ] || { log "ERROR: token list not found: ${TOKEN_LIST}"; exit 1; }
if [ "${token_type}" = bpe ] && [ ! -f "${BPEMODEL}" ]; then
    log "ERROR: bpe model not found: ${BPEMODEL}"
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
        --token_list "${TOKEN_LIST}" \
        $( [ "${token_type}" = bpe ] && printf '%s ' --bpemodel "${BPEMODEL}" ) \
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
        --init_param "${ASR_PRETRAINED}::s2t_model"
        --freeze_param enh_model
        --ignore_init_mismatch true
    )
    if [ "${token_type}" = bpe ]; then
        _train_args+=(--bpemodel "${BPEMODEL}")
    fi
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

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: decode -> ${expdir}/${inference_tag}"
    if bool "${gpu_inference}"; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    for dset in ${test_sets}; do
        _data="data/${dset}"
        _dir="${expdir}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        key_file="${_data}/wav.scp"
        _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            ${python} -m espnet2.bin.asr_inference \
            --enh_s2t_task true \
            --batch_size 1 \
            --ngpu "${_ngpu}" \
            --data_path_and_name_and_type "${_data}/wav.scp,speech,sound" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --asr_train_config "${expdir}/config.yaml" \
            --asr_model_file "${expdir}/${inference_model}" \
            --output_dir "${_logdir}"/output.JOB \
            --config "${inference_config}" \
            $( [ "${use_lm}" = true ] && printf '%s ' --lm_train_config "${LM_CONFIG_PATH}" --lm_file "${LM_FILE}" )

        for f in token token_int score text; do
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/1best_recog/${f}_spk1"
            done | LC_ALL=C sort -k1 > "${_dir}/${f}_spk1"
        done
    done
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: score"
    for dset in ${test_sets}; do
        _data="data/${dset}"
        _dir="${expdir}/${inference_tag}/${dset}"
        for _type in wer cer; do
            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            if [ "${_type}" = wer ]; then
                paste \
                    <(<"${_data}/text_spk1" ${python} -m espnet2.bin.tokenize_text -f 2- --input - --output - --token_type word --non_linguistic_symbols data/nlsyms.txt --remove_non_linguistic_symbols true --cleaner none) \
                    <(<"${_data}/utt2spk" awk '{print "(" $2 "-" $1 ")"}') \
                    > "${_scoredir}/ref_spk1.trn"
                paste \
                    <(<"${_dir}/text_spk1" ${python} -m espnet2.bin.tokenize_text -f 2- --input - --output - --token_type word --non_linguistic_symbols data/nlsyms.txt --remove_non_linguistic_symbols true) \
                    <(<"${_data}/utt2spk" awk '{print "(" $2 "-" $1 ")"}') \
                    > "${_scoredir}/hyp_spk1.trn"
            else
                paste \
                    <(<"${_data}/text_spk1" ${python} -m espnet2.bin.tokenize_text -f 2- --input - --output - --token_type char --non_linguistic_symbols data/nlsyms.txt --remove_non_linguistic_symbols true --cleaner none) \
                    <(<"${_data}/utt2spk" awk '{print "(" $2 "-" $1 ")"}') \
                    > "${_scoredir}/ref_spk1.trn"
                paste \
                    <(<"${_dir}/text_spk1" ${python} -m espnet2.bin.tokenize_text -f 2- --input - --output - --token_type char --non_linguistic_symbols data/nlsyms.txt --remove_non_linguistic_symbols true) \
                    <(<"${_data}/utt2spk" awk '{print "(" $2 "-" $1 ")"}') \
                    > "${_scoredir}/hyp_spk1.trn"
            fi

            sclite -r "${_scoredir}/ref_spk1.trn" trn -h "${_scoredir}/hyp_spk1.trn" trn -i rm -o all stdout > "${_scoredir}/result.txt"
        done
    done
fi
