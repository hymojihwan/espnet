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
librispeech_dir="${LIBRISPEECH}/LibriSpeech"
output_dir="data"

# Parse command line arguments
help_message=$(cat << EOF
Usage: $0 [options]

Options:
    --librispeech_dir    # Directory containing LibriSpeech data (default: /DB/LibriSpeech)
    --output_dir         # Directory to store prepared data (default: data)
    --help               # Show this help message
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# Check if LibriSpeech directory exists
if [ ! -d "${librispeech_dir}" ]; then
    log "Error: LibriSpeech directory ${librispeech_dir} does not exist"
    log "Please download LibriSpeech dataset and place it in ${librispeech_dir}"
    exit 1
fi

# Create output directory
mkdir -p "${output_dir}"

# Function to prepare a dataset
prepare_dataset() {
    local dataset=$1
    local src_dir="${librispeech_dir}/${dataset}"
    local dst_dir="${output_dir}/${dataset//-/_}"
    
    if [ ! -d "${src_dir}" ]; then
        log "Warning: Dataset ${src_dir} does not exist, skipping"
        return
    fi
    
    log "Preparing dataset: ${dataset}"
    
    # Prepare data using data_prep.sh
    local/data_prep.sh "${src_dir}" "${dst_dir}"
    
    log "Completed preparing dataset: ${dataset}"
}

# Main processing
log "Starting LibriSpeech data preparation"
log "LibriSpeech directory: ${librispeech_dir}"
log "Output directory: ${output_dir}"

# Process all datasets
datasets=("dev-clean" "dev-other" "test-clean" "test-other" "train-clean-100")

for dataset in "${datasets[@]}"; do
    prepare_dataset "${dataset}"
done

# Combine dev datasets
if [ -d "${output_dir}/dev_clean" ] && [ -d "${output_dir}/dev_other" ]; then
    log "Combining dev datasets..."
    utils/combine_data.sh --extra_files utt2num_frames "${output_dir}/dev" "${output_dir}/dev_clean" "${output_dir}/dev_other"
fi

# Combine test datasets
if [ -d "${output_dir}/test_clean" ] && [ -d "${output_dir}/test_other" ]; then
    log "Combining test datasets..."
    utils/combine_data.sh --extra_files utt2num_frames "${output_dir}/test" "${output_dir}/test_clean" "${output_dir}/test_other"
fi

log "LibriSpeech data preparation completed!"
log "Prepared data saved to: ${output_dir}" 