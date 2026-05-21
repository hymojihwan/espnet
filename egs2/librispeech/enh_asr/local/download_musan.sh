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
musan_dir="${MUSAN}"
musan_url="https://www.openslr.org/resources/17/musan.tar.gz"
musan_tar="musan.tar.gz"

# Parse command line arguments
help_message=$(cat << EOF
Usage: $0 [options]

Options:
    --musan_dir    # Directory to store MUSAN data (default: data/musan)
    --help         # Show this help message
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# Create MUSAN directory
mkdir -p "${musan_dir}"

# Check if MUSAN data already exists
if [ -d "${musan_dir}/music" ] && [ -d "${musan_dir}/noise" ] && [ -d "${musan_dir}/speech" ]; then
    log "MUSAN data already exists in ${musan_dir}"
    log "Skipping download..."
    exit 0
fi

# Download MUSAN dataset
log "Downloading MUSAN dataset..."
if command -v wget >/dev/null 2>&1; then
    wget -O "${musan_tar}" "${musan_url}"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${musan_tar}" "${musan_url}"
else
    log "Error: Neither wget nor curl is available"
    exit 1
fi

# Extract MUSAN dataset
log "Extracting MUSAN dataset..."
tar -xzf "${musan_tar}" -C "${musan_dir}" --strip-components=1

# Clean up
rm -f "${musan_tar}"

# Verify download
if [ -d "${musan_dir}/music" ] && [ -d "${musan_dir}/noise" ] && [ -d "${musan_dir}/speech" ]; then
    log "MUSAN dataset downloaded successfully!"
    log "MUSAN data location: ${musan_dir}"
    log "Contents:"
    log "  - music/: ${musan_dir}/music"
    log "  - noise/: ${musan_dir}/noise"
    log "  - speech/: ${musan_dir}/speech"
else
    log "Error: MUSAN dataset download failed or incomplete"
    exit 1
fi 