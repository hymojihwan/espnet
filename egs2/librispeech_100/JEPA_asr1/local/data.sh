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

if [ -z "${NOISY_LIBRISPEECH}" ]; then
    log "Fill the value of 'NOISY_LIBRISPEECH' of db.sh"
    exit 1
fi

# Stage 1: Download LibriSpeech data (optional)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
	echo "stage 1: Data Download to ${LIBRISPEECH}"
	for part in dev-clean test-clean dev-other test-other train-clean-100; do
            local/download_and_untar.sh ${LIBRISPEECH} ${data_url} ${part}
	done
    else
        log "stage 1: ${LIBRISPEECH}/LibriSpeech/LICENSE.TXT is already existing. Skip data downloading"
    fi
fi

# Stage 2: Prepare clean LibriSpeech data (for clean_wav.scp)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare clean LibriSpeech data from ${LIBRISPEECH}/LibriSpeech"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        # use underscore-separated names in data directories (no suffix for clean)
        part_name=${part//-/_}
        local/data_prep.sh ${LIBRISPEECH}/LibriSpeech/${part} data/${part_name}
    done
fi

# Stage 3: Prepare noisy LibriSpeech data (for wav.scp)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare noisy LibriSpeech data from ${NOISY_LIBRISPEECH}"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        part_name=${part//-/_}
        noisy_part_dir="${NOISY_LIBRISPEECH}/${part}"
        
        if [ -d "${noisy_part_dir}" ]; then
            log "Preparing noisy data for ${part_name} from ${noisy_part_dir}"
            local/data_prep.sh "${noisy_part_dir}" "data/${part_name}_noisy"
        else
            log "Warning: ${noisy_part_dir} does not exist, skipping ${part_name}"
        fi
    done
fi

# Stage 4: Prepare JEPA data (combine noisy wav.scp, clean_wav.scp, text)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare JEPA data (noisy wav.scp, clean_wav.scp, text)"
    
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        part_name=${part//-/_}
        clean_data_dir="data/${part_name}"  # Clean data without suffix
        noisy_data_dir="data/${part_name}_noisy"
        jepa_data_dir="data/${part_name}_noisy"  # Final JEPA data goes to _noisy directory
        
        if [ ! -d "${clean_data_dir}" ]; then
            log "Error: ${clean_data_dir} does not exist. Run stage 2 first."
            exit 1
        fi
        
        if [ ! -d "${noisy_data_dir}" ]; then
            log "Error: ${noisy_data_dir} does not exist. Run stage 3 first."
            exit 1
        fi
        
        # Create JEPA data directory (may already exist from stage 3)
        mkdir -p "${jepa_data_dir}"
        
        # FIRST: Create clean_wav.scp from clean data
        if [ -f "${clean_data_dir}/wav.scp" ]; then
            cp "${clean_data_dir}/wav.scp" "${jepa_data_dir}/clean_wav.scp"
            log "Created clean_wav.scp from ${clean_data_dir}"
        else
            log "Error: ${clean_data_dir}/wav.scp not found"
            exit 1
        fi
        
        # Noisy wav.scp is already in place from stage 3
        if [ ! -f "${noisy_data_dir}/wav.scp" ]; then
            log "Error: ${noisy_data_dir}/wav.scp not found"
            exit 1
        fi
        
        # Copy text file from clean data (text is the same for clean and noisy)
        if [ -f "${clean_data_dir}/text" ]; then
            cp "${clean_data_dir}/text" "${jepa_data_dir}/text"
            log "Copied text from ${clean_data_dir}"
        else
            log "Error: ${clean_data_dir}/text not found"
            exit 1
        fi
        
        # Copy other necessary files from clean data
        for file in utt2spk spk2utt spk2gender; do
            if [ -f "${clean_data_dir}/${file}" ]; then
                # Only copy if files are different or target doesn't exist
                if [ ! -f "${jepa_data_dir}/${file}" ] || ! cmp -s "${clean_data_dir}/${file}" "${jepa_data_dir}/${file}"; then
                    cp "${clean_data_dir}/${file}" "${jepa_data_dir}/${file}"
                fi
            fi
        done
        
        # Validate that we have all required files
        if [ -f "${jepa_data_dir}/wav.scp" ] && [ -f "${jepa_data_dir}/clean_wav.scp" ] && [ -f "${jepa_data_dir}/text" ]; then
            log "Successfully prepared JEPA data for ${part_name}_noisy"
            log "  - noisy: ${jepa_data_dir}/wav.scp"
            log "  - clean: ${jepa_data_dir}/clean_wav.scp (from ${clean_data_dir})"
            log "  - text: ${jepa_data_dir}/text"
        else
            log "Error: Missing required files for ${part_name}_noisy"
            exit 1
        fi
    done
fi

# Stage 5: Combine datasets
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Combine all training and development sets"
    if [ -d "data/dev_clean_noisy" ] && [ -d "data/dev_other_noisy" ]; then
        # Include clean_wav.scp in the combined data
        utils/combine_data.sh --extra_files "utt2num_frames clean_wav.scp" data/${train_dev} data/dev_clean_noisy data/dev_other_noisy
    fi
    if [ -d "data/train_clean_100" ] && [ -d "data/train_clean_100_noisy" ]; then
        # Combine train_clean_100 and train_clean_100_noisy into train
        log "Combining train_clean_100 and train_clean_100_noisy into train"
        
        # Add suffix to utterance IDs to avoid conflicts
        # Create temporary directories with modified IDs
        mkdir -p data/train_clean_100_clean_id
        mkdir -p data/train_clean_100_noisy_id
        
        # Copy and modify train_clean_100 (add _clean suffix to utterance IDs)
        for file in wav.scp text utt2spk utt2num_frames clean_wav.scp; do
            if [ -f "data/train_clean_100/${file}" ]; then
                awk '{printf "%s_clean", $1; for(i=2;i<=NF;i++) printf " %s", $i; printf "\n"}' "data/train_clean_100/${file}" > "data/train_clean_100_clean_id/${file}"
            fi
        done
        
        # Copy and modify train_clean_100_noisy (add _noisy suffix to utterance IDs)
        for file in wav.scp text utt2spk utt2num_frames clean_wav.scp; do
            if [ -f "data/train_clean_100_noisy/${file}" ]; then
                awk '{printf "%s_noisy", $1; for(i=2;i<=NF;i++) printf " %s", $i; printf "\n"}' "data/train_clean_100_noisy/${file}" > "data/train_clean_100_noisy_id/${file}"
            fi
        done
        
        # Generate spk2utt from utt2spk for modified IDs
        if [ -f "data/train_clean_100_clean_id/utt2spk" ]; then
            utils/utt2spk_to_spk2utt.pl data/train_clean_100_clean_id/utt2spk > data/train_clean_100_clean_id/spk2utt
        fi
        if [ -f "data/train_clean_100_noisy_id/utt2spk" ]; then
            utils/utt2spk_to_spk2utt.pl data/train_clean_100_noisy_id/utt2spk > data/train_clean_100_noisy_id/spk2utt
        fi
        
        # Now combine with modified IDs
        utils/combine_data.sh --extra_files "utt2num_frames clean_wav.scp" data/train data/train_clean_100_clean_id data/train_clean_100_noisy_id
        
        # Clean up temporary directories
        rm -rf data/train_clean_100_clean_id data/train_clean_100_noisy_id
    fi
fi

# Stage 6: Prepare external text data (for language model)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # use external data
    if [ ! -e data/local/other_text/librispeech-lm-norm.txt.gz ]; then
	log "stage 6: prepare external text data from http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
        mkdir -p data/local/other_text
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/other_text/ 2>/dev/null || true
    fi
    if [ ! -e data/local/other_text/text ]; then
	# provide utterance id to each texts
	# e.g., librispeech_lng_00003686 A BANK CHECK
	if [ -f data/local/other_text/librispeech-lm-norm.txt.gz ]; then
	    zcat data/local/other_text/librispeech-lm-norm.txt.gz | \
		awk '{ printf("librispeech_lng_%08d %s\n",NR,$0) } ' > data/local/other_text/text
	fi
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
