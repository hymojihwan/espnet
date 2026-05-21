#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100000
train_dev="dev_noisy_randm5to15"

log "$0 $*"
. utils/parse_options.sh
MUSAN_OVERRIDE="${MUSAN:-}"
. ./db.sh
. ./path.sh
. ./cmd.sh
if [ -z "${MUSAN}" ] && [ -n "${MUSAN_OVERRIDE}" ]; then
    MUSAN="${MUSAN_OVERRIDE}"
fi

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

if [ -z "${MUSAN}" ]; then
    log "Fill the value of 'MUSAN' of db.sh"
    exit 1
fi

if [ "${LIBRISPEECH}" = "downloads" ]; then
    LIBRI_ROOT="${PWD}/downloads/LibriSpeech"
else
    LIBRI_ROOT="${LIBRISPEECH}/LibriSpeech"
fi

noisy_root="${PWD}/data/local/noisy_spl"
speakers_txt="${LIBRI_ROOT}/SPEAKERS.TXT"

split_ready() {
    local clean_split=$1
    local out_split=$2
    local clean_count out_count
    clean_count=$(find "${LIBRI_ROOT}/${clean_split}" -name "*.flac" 2>/dev/null | wc -l)
    out_count=$(find "${out_split}" -name "*.flac" 2>/dev/null | wc -l)
    [ "${clean_count}" -gt 0 ] && [ "${clean_count}" -eq "${out_count}" ]
}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Ensure clean LibriSpeech data dirs exist"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        part_name=${part//-/_}
        [ -d "data/${part_name}" ] || local/data_prep.sh "${LIBRI_ROOT}/${part}" "data/${part_name}"
    done
    [ -d data/train_960 ] || utils/combine_data.sh --extra_files utt2num_frames data/train_960 data/train_clean_100 data/train_clean_360 data/train_other_500
    [ -d data/dev ] || utils/combine_data.sh --extra_files utt2num_frames data/dev data/dev_clean data/dev_other
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Create noisy train/dev/test sets"
    mkdir -p "${noisy_root}/rand_m5_15" "${noisy_root}/snr_bins"

    for split in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other; do
        if split_ready "${split}" "${noisy_root}/rand_m5_15/${split}"; then
            log "Skip noisy generation: ${split} already prepared"
        else
            python3 local/mix_noise_librispeech.py \
                --clean_root "${LIBRI_ROOT}" \
                --noise_root "${MUSAN}" \
                --out_root "${noisy_root}/rand_m5_15" \
                --split "${split}" \
                --mode random \
                --snr_min -5 \
                --snr_max 15 \
                --seed 100
        fi
    done

    for snr in -5 0 5 10 15; do
        tag=${snr}
        if [ "${snr}" = "-5" ]; then
            tag="m5"
        fi

        for split in test-clean test-other; do
            if split_ready "${split}" "${noisy_root}/snr_bins/snr_${tag}/${split}"; then
                log "Skip noisy generation: ${split} snr_${tag} already prepared"
            else
                python3 local/mix_noise_librispeech.py \
                    --clean_root "${LIBRI_ROOT}" \
                    --noise_root "${MUSAN}" \
                    --out_root "${noisy_root}/snr_bins/snr_${tag}" \
                    --split "${split}" \
                    --mode fixed \
                    --snr_value "${snr}" \
                    --seed $((300 + snr + 5))
            fi
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Build kaldi data dirs for noisy SPL"
    if [ ! -f "${speakers_txt}" ]; then
        log "Error: Missing ${speakers_txt}"
        exit 1
    fi

    cp -f "${speakers_txt}" "${noisy_root}/rand_m5_15/SPEAKERS.TXT"
    for snr in m5 0 5 10 15; do
        cp -f "${speakers_txt}" "${noisy_root}/snr_bins/snr_${snr}/SPEAKERS.TXT"
    done

    local/data_prep.sh "${noisy_root}/rand_m5_15/train-clean-100" data/train_clean_100_noisy_randm5to15
    local/data_prep.sh "${noisy_root}/rand_m5_15/train-clean-360" data/train_clean_360_noisy_randm5to15
    local/data_prep.sh "${noisy_root}/rand_m5_15/train-other-500" data/train_other_500_noisy_randm5to15
    utils/combine_data.sh --extra_files utt2num_frames \
        data/train_960_noisy_randm5to15 \
        data/train_clean_100_noisy_randm5to15 \
        data/train_clean_360_noisy_randm5to15 \
        data/train_other_500_noisy_randm5to15

    local/data_prep.sh "${noisy_root}/rand_m5_15/dev-clean" data/dev_clean_noisy_randm5to15
    local/data_prep.sh "${noisy_root}/rand_m5_15/dev-other" data/dev_other_noisy_randm5to15
    utils/combine_data.sh --extra_files utt2num_frames \
        data/dev_noisy_randm5to15 \
        data/dev_clean_noisy_randm5to15 \
        data/dev_other_noisy_randm5to15

    for snr in m5 0 5 10 15; do
        local/data_prep.sh "${noisy_root}/snr_bins/snr_${snr}/test-clean" "data/test_clean_noisy_snr${snr}"
        local/data_prep.sh "${noisy_root}/snr_bins/snr_${snr}/test-other" "data/test_other_noisy_snr${snr}"
    done

    make_spk1() {
        local noisy_set=$1
        local clean_set=$2
        utils/filter_scp.pl "data/${noisy_set}/wav.scp" "data/${clean_set}/wav.scp" > "data/${noisy_set}/spk1.scp"
    }

    make_spk1 train_clean_100_noisy_randm5to15 train_clean_100
    make_spk1 train_clean_360_noisy_randm5to15 train_clean_360
    make_spk1 train_other_500_noisy_randm5to15 train_other_500
    make_spk1 train_960_noisy_randm5to15 train_960
    make_spk1 dev_clean_noisy_randm5to15 dev_clean
    make_spk1 dev_other_noisy_randm5to15 dev_other
    make_spk1 dev_noisy_randm5to15 dev
    for snr in m5 0 5 10 15; do
        make_spk1 "test_clean_noisy_snr${snr}" test_clean
        make_spk1 "test_other_noisy_snr${snr}" test_other
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
