#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh || exit 1;
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

min_snr=0
max_snr=20
noise_dir=""  # Optional: path to noise dataset

. utils/parse_options.sh || exit 1;

asr1_data=../asr1/data

log "Creating noisy training data..."

# Create train_clean_100_noisy if it doesn't exist
if [ ! -d data/train_clean_100_noisy ]; then
    mkdir -p data/train_clean_100_noisy
    
    # Check if we can directly use noisy data from asr1
    if [ -d ${asr1_data}/train_clean_100_noisy ]; then
        log "Using existing noisy data from asr1"
        for f in wav.scp text utt2spk spk2utt spk1.scp; do
            if [ -f ${asr1_data}/train_clean_100_noisy/${f} ]; then
                cp ${asr1_data}/train_clean_100_noisy/${f} data/train_clean_100_noisy/
            fi
        done
    else
        log "Creating noisy training data from clean speech"
        
        # Copy metadata
        for f in text utt2spk spk2utt; do
            cp ${asr1_data}/train_clean_100/${f} data/train_clean_100_noisy/
        done
        
        # Use clean speech as reference
        cp ${asr1_data}/train_clean_100/wav.scp data/train_clean_100_noisy/spk1.scp
        
        # Add noise to create noisy version
        mkdir -p data/train_clean_100_noisy/wav
        
        if [ -z "${noise_dir}" ]; then
            log "No noise directory specified, using white noise"
            noise_dir="none"
        fi
        
        python3 local/add_noise.py \
            --clean_scp ${asr1_data}/train_clean_100/wav.scp \
            --noise_dir ${noise_dir} \
            --output_dir data/train_clean_100_noisy/wav \
            --output_scp data/train_clean_100_noisy/wav.scp \
            --min_snr ${min_snr} \
            --max_snr ${max_snr}
    fi
    
    log "Created data/train_clean_100_noisy"
fi

log "Noisy training data preparation completed"

