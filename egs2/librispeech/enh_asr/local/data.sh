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
SECONDS=0

stage=1
stop_stage=100000
data_url=www.openslr.org/resources/12
train_dev="dev"

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

if [ -z "${NOISY_LIBRISPEECH}" ]; then
    log "Fill the value of 'NOISY_LIBRISPEECH' of db.sh"
    exit 1
fi

# Define all LibriSpeech parts
all_parts="dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500"
existing_parts="dev-clean test-clean dev-other test-other train-clean-100"
missing_parts="train-clean-360 train-other-500"

# Stage 1: Download missing LibriSpeech data
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Download missing LibriSpeech data"
    for part in ${missing_parts}; do
        if [ ! -d "${LIBRISPEECH}/LibriSpeech/${part}" ]; then
            log "Downloading ${part}..."
            local/download_and_untar.sh ${LIBRISPEECH} ${data_url} ${part}
        else
            log "${part} already exists, skipping download"
        fi
    done
fi

# Stage 2: Download MUSAN dataset
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Download MUSAN dataset"
    if [ ! -d "${MUSAN}" ] || [ ! -d "${MUSAN}/music" ] || [ ! -d "${MUSAN}/noise" ] || [ ! -d "${MUSAN}/speech" ]; then
        bash local/download_musan.sh
    else
        log "stage 2: MUSAN dataset already exists at ${MUSAN}"
    fi
fi

# Stage 3: Create noisy LibriSpeech data for missing parts
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Create noisy LibriSpeech data for missing parts"
    if [ ! -d "${NOISY_LIBRISPEECH}" ]; then
        mkdir -p "${NOISY_LIBRISPEECH}"
    fi
    
    # Check if required Python packages are available
    if ! python3 -c "import soundfile, numpy" 2>/dev/null; then
        log "Installing required Python packages..."
        pip3 install soundfile numpy
    fi
    
    # Add noise to all parts, but skip if already exists
    for part in ${all_parts}; do
        if [ ! -d "${NOISY_LIBRISPEECH}/${part}" ]; then
            log "Adding noise to ${part}..."
            python3 local/accurate_add_noise.py \
                --clean_dir "${LIBRISPEECH}/LibriSpeech" \
                --noise_dir "${MUSAN}" \
                --noisy_dir "${NOISY_LIBRISPEECH}" \
                --snr_min -10.0 \
                --snr_max 10.0 \
                --seed 42
        else
            log "Noisy ${part} already exists, skipping"
        fi
    done
fi

# Stage 4: Prepare ESPnet format data from noisy LibriSpeech
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare ESPnet format data from noisy LibriSpeech"
    for part in ${all_parts}; do
        # use underscore-separated names in data directories.
        if [ -d "${NOISY_LIBRISPEECH}/${part}" ]; then
            local/data_prep.sh "${NOISY_LIBRISPEECH}/${part}" "data/${part//-/_}"
        else
            log "Warning: Noisy ${part} does not exist, skipping"
        fi
    done
fi

# Stage 5: Combine datasets
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Combine all training and development sets"
    
    # Combine train_clean_100, train_clean_360 and train_other_500 to create train_960
    if [ -d "data/train_clean_100" ] && [ -d "data/train_clean_360" ] && [ -d "data/train_other_500" ]; then
        log "Combining train_clean_100, train_clean_360 and train_other_500 to create train_960"
        utils/combine_data.sh --extra_files "utt2num_frames spk1.scp" data/train_960 data/train_clean_100 data/train_clean_360 data/train_other_500
    fi
    
    if [ -d "data/dev_clean" ] && [ -d "data/dev_other" ]; then
        # Include spk1.scp so dev set contains clean references
        utils/combine_data.sh --extra_files "utt2num_frames spk1.scp" data/${train_dev} data/dev_clean data/dev_other
    fi
    
fi

log "Successfully finished. [elapsed=${SECONDS}s]" 