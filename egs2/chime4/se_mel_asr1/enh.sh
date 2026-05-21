#!/usr/bin/env bash
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

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

. ./path.sh
. ./cmd.sh

log "$0 $*"

stage=1
stop_stage=3
channel_mode="${CHANNEL_MODE:-allch}"
speed_perturb_factors="${SPEED_PERTURB_FACTORS:-0.9 1.0 1.1}"
mapper_config="${MAPPER_CONFIG:-conf/train_logmel_mapper.yaml}"
enh_tag="${ENH_TAG:-enh_train_cleanmel_mapper80}"
feature_suffix="${FEATURE_SUFFIX:-_mel80cleanmel}"
mapper_checkpoint="${MAPPER_CHECKPOINT:-}"
ngpu="${NGPU:-1}"
nj="${NJ:-32}"

. ../asr1/utils/parse_options.sh

if [ "${channel_mode}" = allch ]; then
    base_train_set=tr05_multi_mixed_track
    valid_set=dt05_multi_mixed_track
    test_sets="et05_multi_mixed_track et05_simu_snr-5_allch_track et05_simu_snr0_allch_track et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    base_train_set=tr05_multi_isolated_1ch_track
    valid_set=dt05_multi_isolated_1ch_track
    test_sets="et05_multi_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

mapper_exp="exp/${enh_tag}"
if [ -n "${speed_perturb_factors}" ]; then
    mapper_exp+="_sp"
fi

train_set="${base_train_set}"

prepare_speed_perturb() {
    if [ -z "${speed_perturb_factors}" ]; then
        train_set="${base_train_set}"
        return 0
    fi

    if [ -f "data/${base_train_set}_sp/wav.scp" ] && [ -f "data/${base_train_set}_sp/spk1.scp" ]; then
        train_set="${base_train_set}_sp"
        return 0
    fi

    local perturb_script="${script_dir}/../../TEMPLATE/asr1/scripts/utils/perturb_enh_data_dir_speed.sh"
    local format_wav_scp="${script_dir}/../../TEMPLATE/asr1/scripts/audio/format_wav_scp.sh"
    local scp_list="wav.scp spk1.scp"
    local extra_files="spk1.scp text utt2lang"
    local utt_extra_files="text utt2lang"
    local dirs=""

    rm -rf "data/${base_train_set}_sp"
    for factor in ${speed_perturb_factors}; do
        if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
            rm -rf "data/${base_train_set}_sp${factor}"
            (
                cd "${script_dir}/../asr1"
                "${perturb_script}" \
                    --utt_extra_files "${utt_extra_files}" \
                    "${factor}" \
                    "${script_dir}/data/${base_train_set}" \
                    "${script_dir}/data/${base_train_set}_sp${factor}" \
                    "${scp_list}"
            )
            dirs+="data/${base_train_set}_sp${factor} "
        else
            dirs+="data/${base_train_set} "
        fi
    done

    (
        cd "${script_dir}/../asr1"
        # shellcheck disable=SC2086
        utils/combine_data.sh --extra-files "${extra_files}" "${script_dir}/data/${base_train_set}_sp" ${dirs//data\//${script_dir}/data/}
        utils/fix_data_dir.sh "${script_dir}/data/${base_train_set}_sp" >/dev/null
    )

    for scp_name in wav.scp spk1.scp; do
        if grep -q '|' "data/${base_train_set}_sp/${scp_name}"; then
            cp "data/${base_train_set}_sp/${scp_name}" "data/${base_train_set}_sp/${scp_name}.src"
            (
                cd "${script_dir}/../asr1"
                "${format_wav_scp}" \
                    --nj "${nj}" \
                    --cmd "${train_cmd}" \
                    --audio-format wav \
                    --fs 16k \
                    --out-filename "${scp_name}" \
                    "${script_dir}/data/${base_train_set}_sp/${scp_name}.src" \
                    "${script_dir}/data/${base_train_set}_sp" \
                    "${script_dir}/data/${base_train_set}_sp/logs/${scp_name%.scp}" \
                    "${script_dir}/data/${base_train_set}_sp/data/${scp_name%.scp}"
            )
            rm -f "data/${base_train_set}_sp/${scp_name}.src"
        fi
    done

    (
        cd "${script_dir}/../asr1"
        utils/fix_data_dir.sh "${script_dir}/data/${base_train_set}_sp" >/dev/null
    )
    train_set="${base_train_set}_sp"
}

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: prepare SE data"
    ./local/data.sh --channel_mode "${channel_mode}"
    prepare_speed_perturb
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    [ -f "data/${base_train_set}/wav.scp" ] || ./local/data.sh --channel_mode "${channel_mode}"
    prepare_speed_perturb
    log "Stage 2: train log-mel mapper -> ${mapper_exp}"
    mkdir -p "${mapper_exp}"
    {
        echo "# python3 ./local/train_logmel_mapper.py --config ${mapper_config} --train_wav_scp data/${train_set}/wav.scp --train_ref_scp data/${train_set}/spk1.scp --valid_wav_scp data/${valid_set}/wav.scp --valid_ref_scp data/${valid_set}/spk1.scp --output_dir ${mapper_exp}"
        echo "# Started at $(date)"
        echo "#"
        python3 ./local/train_logmel_mapper.py \
            --config "${mapper_config}" \
            --train_wav_scp "data/${train_set}/wav.scp" \
            --train_ref_scp "data/${train_set}/spk1.scp" \
            --valid_wav_scp "data/${valid_set}/wav.scp" \
            --valid_ref_scp "data/${valid_set}/spk1.scp" \
            --ngpu "${ngpu}" \
            --output_dir "${mapper_exp}"
        echo "#"
        echo "# Ended at $(date), elapsed time=${SECONDS}s"
    } > "${mapper_exp}/train.log" 2>&1
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    [ -f "data/${base_train_set}/wav.scp" ] || ./local/data.sh --channel_mode "${channel_mode}"
    prepare_speed_perturb
    if [ -n "${mapper_checkpoint}" ]; then
        current_mapper_checkpoint="${mapper_checkpoint}"
    else
        current_mapper_checkpoint="${mapper_exp}/valid.loss.best.pth"
    fi
    [ -f "${current_mapper_checkpoint}" ] || { log "Missing mapper checkpoint: ${current_mapper_checkpoint}"; exit 1; }
    log "Stage 3: dump extracted log-mel features"
    all_sets="${train_set} ${valid_set} ${test_sets}"
    for dset in ${all_sets}; do
        out_dir="data/${dset}${feature_suffix}"
        src_count=$(wc -l < "data/${dset}/wav.scp")
        dst_count=0
        if [ -f "${out_dir}/feats.scp" ]; then
            dst_count=$(wc -l < "${out_dir}/feats.scp")
        fi
        if [ "${dst_count}" -eq "${src_count}" ] && [ "${src_count}" -gt 0 ]; then
            log "Stage 3: skip ${dset} (${dst_count}/${src_count} features already dumped)"
        else
            if [ "${dst_count}" -gt 0 ]; then
                log "Stage 3: re-dump ${dset} (partial ${dst_count}/${src_count} found)"
            else
                log "Stage 3: dump ${dset} (${src_count} utterances)"
            fi
            rm -rf "${out_dir}"
            mkdir -p "${out_dir}"
            python3 ./local/dump_logmel_features.py \
                --config "${mapper_config}" \
                --checkpoint "${current_mapper_checkpoint}" \
                --wav_scp "data/${dset}/wav.scp" \
                --clean_wav_scp "data/${dset}/spk1.scp" \
                --text "data/${dset}/text" \
                --utt2spk "data/${dset}/utt2spk" \
                --output_dir "${out_dir}"
        fi
        rm -rf "../asr1/data/${dset}${feature_suffix}"
        ln -s "${script_dir}/${out_dir}" "../asr1/data/${dset}${feature_suffix}"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
