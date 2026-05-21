#!/usr/bin/env bash
# Script to rename existing data directories to add _noisy suffix
# This script renames directories like train_clean_100 to train_clean_100_noisy
# and creates corresponding _clean directories if needed

set -e
set -u

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_dir="data"

if [ ! -d "${data_dir}" ]; then
    log "Error: ${data_dir} directory does not exist"
    exit 1
fi

# List of data directories to rename
parts=("train_clean_100" "dev_clean" "dev_other" "test_clean" "test_other")

for part in "${parts[@]}"; do
    old_dir="${data_dir}/${part}"
    new_dir="${data_dir}/${part}_noisy"
    clean_dir="${data_dir}/${part}_clean"
    
    if [ -d "${old_dir}" ]; then
        # Check if it already has _noisy suffix
        if [[ "${part}" == *_noisy ]]; then
            log "Skipping ${old_dir} (already has _noisy suffix)"
            continue
        fi
        
        # Check if _noisy directory already exists
        if [ -d "${new_dir}" ]; then
            log "Warning: ${new_dir} already exists. Skipping rename of ${old_dir}"
        else
            log "Renaming ${old_dir} -> ${new_dir}"
            mv "${old_dir}" "${new_dir}"
        fi
        
        # Note: Clean data directory should already exist as ${part} (without suffix)
        # This script only renames noisy data directories
    else
        log "Skipping ${old_dir} (does not exist)"
    fi
done

log "Finished renaming data directories"

