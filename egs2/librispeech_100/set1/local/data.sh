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
data_url=www.openslr.org/resources/12
train_dev="dev_noisy_randm5to15"

log "$0 $*"
. utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

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

noisy_root="${PWD}/data/local/noisy_set1"
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
    if [ ! -e "${LIBRI_ROOT}/LICENSE.TXT" ]; then
        log "stage 1: Data Download to ${LIBRISPEECH}"
        for part in dev-clean test-clean dev-other test-other train-clean-100; do
            local/download_and_untar.sh "${LIBRISPEECH}" "${data_url}" "${part}"
        done
    else
        log "stage 1: ${LIBRI_ROOT}/LICENSE.TXT already exists. Skip downloading."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare clean LibriSpeech data"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        part_name=${part//-/_}
        local/data_prep.sh "${LIBRI_ROOT}/${part}" "data/${part_name}"
    done
    utils/combine_data.sh --extra_files utt2num_frames data/dev data/dev_clean data/dev_other
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Create noisy sets (train/valid random -5~15 dB, test fixed SNR bins)"
    mkdir -p "${noisy_root}/rand_m5_15" "${noisy_root}/snr_bins"

    if split_ready train-clean-100 "${noisy_root}/rand_m5_15/train-clean-100"; then
        log "Skip noisy generation: train-clean-100 already prepared"
    else
        python3 local/mix_noise_librispeech.py \
            --clean_root "${LIBRI_ROOT}" \
            --noise_root "${MUSAN}" \
            --out_root "${noisy_root}/rand_m5_15" \
            --split train-clean-100 \
            --mode random \
            --snr_min -5 \
            --snr_max 15 \
            --seed 100
    fi

    if split_ready dev-clean "${noisy_root}/rand_m5_15/dev-clean"; then
        log "Skip noisy generation: dev-clean already prepared"
    else
        python3 local/mix_noise_librispeech.py \
            --clean_root "${LIBRI_ROOT}" \
            --noise_root "${MUSAN}" \
            --out_root "${noisy_root}/rand_m5_15" \
            --split dev-clean \
            --mode random \
            --snr_min -5 \
            --snr_max 15 \
            --seed 200
    fi

    if split_ready dev-other "${noisy_root}/rand_m5_15/dev-other"; then
        log "Skip noisy generation: dev-other already prepared"
    else
        python3 local/mix_noise_librispeech.py \
            --clean_root "${LIBRI_ROOT}" \
            --noise_root "${MUSAN}" \
            --out_root "${noisy_root}/rand_m5_15" \
            --split dev-other \
            --mode random \
            --snr_min -5 \
            --snr_max 15 \
            --seed 201
    fi

    for snr in -5 0 5 10 15; do
        tag=${snr}
        if [ "${snr}" = "-5" ]; then
            tag="m5"
        fi

        if split_ready test-clean "${noisy_root}/snr_bins/snr_${tag}/test-clean"; then
            log "Skip noisy generation: test-clean snr_${tag} already prepared"
        else
            python3 local/mix_noise_librispeech.py \
                --clean_root "${LIBRI_ROOT}" \
                --noise_root "${MUSAN}" \
                --out_root "${noisy_root}/snr_bins/snr_${tag}" \
                --split test-clean \
                --mode fixed \
                --snr_value "${snr}" \
                --seed $((300 + snr + 5))
        fi

        if split_ready test-other "${noisy_root}/snr_bins/snr_${tag}/test-other"; then
            log "Skip noisy generation: test-other snr_${tag} already prepared"
        else
            python3 local/mix_noise_librispeech.py \
                --clean_root "${LIBRI_ROOT}" \
                --noise_root "${MUSAN}" \
                --out_root "${noisy_root}/snr_bins/snr_${tag}" \
                --split test-other \
                --mode fixed \
                --snr_value "${snr}" \
                --seed $((400 + snr + 5))
        fi
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare noisy Kaldi-style data dirs"
    # local/data_prep.sh expects SPEAKERS.TXT in split/.. (one level above split dir)
    if [ -f "${speakers_txt}" ]; then
        cp -f "${speakers_txt}" "${noisy_root}/rand_m5_15/SPEAKERS.TXT"
        for snr in m5 0 5 10 15; do
            cp -f "${speakers_txt}" "${noisy_root}/snr_bins/snr_${snr}/SPEAKERS.TXT"
        done
    else
        log "Error: Missing ${speakers_txt}"
        exit 1
    fi

    local/data_prep.sh "${noisy_root}/rand_m5_15/train-clean-100" data/train_clean_100_noisy_randm5to15
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

    # Create clean references for enhancement training/evaluation.
    # spk1.scp contains clean speech aligned by utterance-id.
    make_spk1() {
        local noisy_set=$1
        local clean_set=$2
        utils/filter_scp.pl "data/${noisy_set}/wav.scp" "data/${clean_set}/wav.scp" > "data/${noisy_set}/spk1.scp"
    }

    make_spk1 train_clean_100_noisy_randm5to15 train_clean_100
    make_spk1 dev_clean_noisy_randm5to15 dev_clean
    make_spk1 dev_other_noisy_randm5to15 dev_other
    make_spk1 dev_noisy_randm5to15 dev

    for snr in m5 0 5 10 15; do
        make_spk1 "test_clean_noisy_snr${snr}" test_clean
        make_spk1 "test_other_noisy_snr${snr}" test_other
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ ! -e data/local/other_text/librispeech-lm-norm.txt.gz ]; then
        log "stage 5: download external text data"
        mkdir -p data/local/other_text
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/other_text/
    fi
    if [ ! -e data/local/other_text/text ]; then
        zcat data/local/other_text/librispeech-lm-norm.txt.gz | \
            awk '{ printf("librispeech_lng_%08d %s\n",NR,$0) } ' > data/local/other_text/text
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
