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

# Load database configuration
. ./db.sh

# Configuration
clean_data_dir="data"  # Directory containing clean LibriSpeech data (ESPnet format)
noisy_data_dir="${NOISY_LIBRISPEECH}"  # Directory to store noisy data (from db.sh)
musan_dir="${MUSAN}"  # Directory containing MUSAN noise data (from db.sh)
snr_range="-10:10"  # SNR range (min:max)
noise_apply_prob=1.0  # Probability of applying noise
librispeech_dir="${LIBRISPEECH}/LibriSpeech"  # Original LibriSpeech directory

# Parse command line arguments
help_message=$(cat << EOF
Usage: $0 [options]

Options:
    --clean_data_dir    # Directory containing clean LibriSpeech data (default: data)
    --noisy_data_dir    # Directory to store noisy data (default: data_noisy)
    --musan_dir         # Directory containing MUSAN noise data (default: data/musan)
    --snr_range         # SNR range in format min:max (default: 5:15)
    --noise_apply_prob  # Probability of applying noise (default: 1.0)
    --help              # Show this help message
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# Check if MUSAN directory exists
if [ ! -d "${musan_dir}" ]; then
    log "Error: MUSAN directory ${musan_dir} does not exist"
    log "Please download MUSAN dataset and place it in ${musan_dir}"
    exit 1
fi

# Create noisy data directory
mkdir -p "${noisy_data_dir}"

# Function to add noise to a dataset
add_noise_to_dataset() {
    local dataset=$1
    local clean_dir="${clean_data_dir}/${dataset}"
    local noisy_dir="${noisy_data_dir}/${dataset}"
    
    if [ ! -d "${clean_dir}" ]; then
        log "Warning: Clean dataset ${clean_dir} does not exist, skipping"
        return
    fi
    
    log "Processing dataset: ${dataset}"
    
    # Create noisy data directory
    mkdir -p "${noisy_dir}"
    
    # Copy text files (no changes needed)
    if [ -f "${clean_dir}/text" ]; then
        cp "${clean_dir}/text" "${noisy_dir}/"
    fi
    
    if [ -f "${clean_dir}/utt2spk" ]; then
        cp "${clean_dir}/utt2spk" "${noisy_dir}/"
    fi
    
    if [ -f "${clean_dir}/spk2utt" ]; then
        cp "${clean_dir}/spk2utt" "${noisy_dir}/"
    fi
    
    # Create wav.scp for noisy data
    if [ -f "${clean_dir}/wav.scp" ]; then
        log "Creating noisy wav.scp for ${dataset}"
        
        # Parse SNR range
        snr_min=$(echo "${snr_range}" | cut -d':' -f1)
        snr_max=$(echo "${snr_range}" | cut -d':' -f2)
        
                    # Set seed based on dataset name for reproducibility
        case "${dataset}" in
            "train_clean_100")
                seed=42
                ;;
            "dev")
                seed=123
                ;;
            "test_clean")
                seed=456
                ;;
            "test_other")
                seed=789
                ;;
            *)
                seed=999
                ;;
        esac
        
        # Call Python script to add noise
        python3 local/add_noise.py \
            --input_scp "${clean_dir}/wav.scp" \
            --output_scp "${noisy_dir}/wav.scp" \
            --musan_dir "${musan_dir}" \
            --output_dir "${noisy_data_dir}" \
            --snr_min "${snr_min}" \
            --snr_max "${snr_max}" \
            --noise_prob "${noise_apply_prob}" \
            --seed "${seed}" \
            --dataset_name "${dataset//_/-}"
    fi
    
    log "Completed processing dataset: ${dataset}"
}

# Main processing
log "Starting noisy data preparation"
log "Clean data directory: ${clean_data_dir}"
log "Noisy data directory: ${noisy_data_dir}"
log "MUSAN directory: ${musan_dir}"
log "SNR range: ${snr_range}"
log "Noise apply probability: ${noise_apply_prob}"

# Process all datasets with original LibriSpeech structure
datasets=("train-clean-100" "dev-clean" "dev-other" "test-clean" "test-other")

for dataset in "${datasets[@]}"; do
    # Convert dataset name to ESPnet format
    espnet_dataset="${dataset//-/_}"
    
    # Check if ESPnet format data exists
    if [ -d "${clean_data_dir}/${espnet_dataset}" ]; then
        add_noise_to_dataset "${espnet_dataset}"
    else
        log "Warning: ESPnet format dataset ${clean_data_dir}/${espnet_dataset} does not exist, skipping"
    fi
done

log "Noisy data preparation completed!"
log "Noisy data saved to: ${noisy_data_dir}" 