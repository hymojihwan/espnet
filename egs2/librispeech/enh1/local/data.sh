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

log "$0 $*"

min_snr=0
max_snr=20
eval_snrs="-10 -5 0 5 10"

. utils/parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# Prepare enhancement data
log "Preparing enhancement data..."

# Link to asr1 data directory
asr1_data=../asr1/data
mkdir -p data

# Function to create noisy data with specific SNR
create_noisy_data_with_snr() {
    local clean_set=$1
    local output_name=$2
    local snr=$3
    local is_random=$4  # 1 for random SNR, 0 for fixed SNR
    
    log "Creating ${output_name} with SNR=${snr}dB (random=${is_random})"
    
    mkdir -p data/${output_name}
    
    # Copy metadata
    for f in text utt2spk spk2utt; do
        if [ -f ${asr1_data}/${clean_set}/${f} ]; then
            cp ${asr1_data}/${clean_set}/${f} data/${output_name}/
        fi
    done
    
    # Use clean speech as reference
    cp ${asr1_data}/${clean_set}/wav.scp data/${output_name}/spk1.scp
    
    # Add noise to create noisy version
    mkdir -p data/${output_name}/wav
    
    if [ "${is_random}" = "1" ]; then
        # Random SNR for training
        python3 local/create_noisy_data.py \
            --clean_scp ${asr1_data}/${clean_set}/wav.scp \
            --output_dir data/${output_name}/wav \
            --output_scp data/${output_name}/wav.scp \
            --min_snr ${min_snr} \
            --max_snr ${max_snr} \
            --random_snr
    else
        # Fixed SNR for evaluation
        python3 local/create_noisy_data.py \
            --clean_scp ${asr1_data}/${clean_set}/wav.scp \
            --output_dir data/${output_name}/wav \
            --output_scp data/${output_name}/wav.scp \
            --fixed_snr ${snr}
    fi
    
    log "Created data/${output_name}"
}

# Prepare training data with random SNR
if [ ! -d data/train_clean_100_noisy ]; then
    create_noisy_data_with_snr "train_clean_100" "train_clean_100_noisy" "${max_snr}" 1
fi

# Prepare validation data for each SNR
for snr in ${eval_snrs}; do
    snr_tag=$(echo ${snr} | sed 's/-/m/g')  # Convert -10 to m10
    if [ ! -d data/dev_clean_noisy_snr${snr_tag} ]; then
        create_noisy_data_with_snr "dev_clean" "dev_clean_noisy_snr${snr_tag}" "${snr}" 0
    fi
done

# Prepare test data for each SNR
for snr in ${eval_snrs}; do
    snr_tag=$(echo ${snr} | sed 's/-/m/g')  # Convert -10 to m10
    if [ ! -d data/test_clean_noisy_snr${snr_tag} ]; then
        create_noisy_data_with_snr "test_clean" "test_clean_noisy_snr${snr_tag}" "${snr}" 0
    fi
    if [ ! -d data/test_other_noisy_snr${snr_tag} ]; then
        create_noisy_data_with_snr "test_other" "test_other_noisy_snr${snr_tag}" "${snr}" 0
    fi
done

log "Data preparation completed successfully"

