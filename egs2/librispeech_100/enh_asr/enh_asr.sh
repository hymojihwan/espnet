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

# General configuration
stage=3                 # Processes starts from the specified stage.
stop_stage=10000        # Processes is stopped at the specified stage.
skip_stages=            # Spicify the stage to be skipped
skip_data_prep=false    # Skip data preparation stages.
skip_train=false        # Skip training stages.
skip_eval=false         # Skip decoding and evaluation stages.
ngpu=4                  # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1             # The number of nodes.
nj=16                   # The number of parallel jobs.
inference_nj=16         # The number of parallel jobs in decoding.
gpu_inference=true      # Whether to perform gpu decoding.
dumpdir=dump            # Directory to dump features.
expdir=exp              # Directory to save experiments.
python=python3          # Specify python to execute espnet commands.

# Enhancement + ASR model related
enh_asr_task=enh_asr_transducer   # Task mode.
enh_asr_tag=                      # Suffix to the result dir for enh_asr model training.
enh_asr_exp=                      # Specify the directory path for enh_asr experiment.
enh_asr_config=                   # Config for enh_asr model training.
inference_config=                  # Config for enh_asr model inference.
inference_enh_asr_model=          # enh_asr model path for inference.

# Data preparation related
local_data_opts= # The options given to local/data.sh.
post_process_local_data_opts= # The options given to local/data.sh for additional processing in stage 4.

# Dataset configuration
train_set="train_clean_100"  # Training dataset
valid_set="dev"              # Validation dataset
test_sets="test_clean test_other"  # Test datasets

# Language configuration
lang=en  # Language code for tokenization

# Text files for tokenization and LM training
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=30  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=2048           # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma or a file containing 1 symbol per line, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Language model related
use_lm=false       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".

# Training related
use_prompt=false # Use prompt ids for multi tasking
use_lang_prompt=false # Use language prompt ids for multi lingual multi tasking
use_nlp_prompt=false # Use text prompt ids for multi lingual multi tasking

# Enhancement related
pretrained_asr_model=  # Path to pre-trained ASR model for SE-only training.

# Noise augmentation is not needed when using pre-generated noisy data

# Parse command line arguments
help_message=$(cat << EOF
Usage: $0 --train_set "<train_set_name>" --valid_set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    --lang                        # Language code for tokenization
    --ngpu                        # Number of GPUs
    --nj                          # Number of parallel jobs
    --gpu_inference              # Whether to perform gpu decoding
    --inference_nj               # Number of parallel jobs in decoding
    --nbpe                       # Number of BPE vocabulary
    --max_wav_duration           # Maximum duration in second
    --enh_asr_config             # Config for enh_asr model training
    --enh_asr_se_only_config     # Config for SE-only training
    --inference_config           # Config for enh_asr model inference
    --inference_enh_asr_model   # enh_asr model path for inference
    --pretrained_asr_model      # Path to pre-trained ASR model
    --lm_train_text             # Text file for LM training
    --bpe_train_text            # Text file for BPE training
EOF
)

log "$0 $*"
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check required arguments
if [ -z "${enh_asr_config}" ]; then
    log "Error: --enh_asr_config is required"
    exit 2
fi

if [ -z "${inference_config}" ]; then
    log "Error: --inference_config is required"
    exit 2
fi

if [ -z "${inference_enh_asr_model}" ]; then
    log "Error: --inference_enh_asr_model is required"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${enh_asr_tag}" ]; then
    if [ -n "${enh_asr_config}" ]; then
        enh_asr_tag="$(basename "${enh_asr_config}" .yaml)_${feats_type}"
    else
        enh_asr_tag="train_${feats_type}"
    fi
fi

# The directory set by this line is the directory that contains writing tools (score_*.py, filter_scp.pl, etc.)
if [ -z "${enh_asr_exp}" ]; then
    enh_asr_exp="${expdir}/${enh_asr_task}_${enh_asr_tag}"
fi

# Data preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"
    
    # Load database configuration
    . ./db.sh
    
    # Prepare all data (LibriSpeech download, MUSAN download, noisy data creation, ESPnet format preparation)
    log "Preparing all data..."
    bash local/data.sh
fi

# Tokenization
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Tokenization"
    
    # Set token list directory
    if [ "${lang}" != noinfo ]; then
        token_listdir=data/${lang}_token_list
    else
        token_listdir=data/token_list
    fi
    
    # Set token list paths based on token type
    if [ "${token_type}" = bpe ]; then
        bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
        bpeprefix="${bpedir}"/bpe
        bpemodel="${bpeprefix}".model
        bpetoken_list="${bpedir}"/tokens.txt
        token_list="${bpetoken_list}"
    elif [ "${token_type}" = char ]; then
        chartoken_list="${token_listdir}"/char/tokens.txt
        token_list="${chartoken_list}"
        bpemodel=none
    elif [ "${token_type}" = word ]; then
        wordtoken_list="${token_listdir}"/word/tokens.txt
        token_list="${wordtoken_list}"
        bpemodel=none
    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi
    
    # Create token list for ASR training
    if [ ! -f "${token_list}" ]; then
        log "Creating token list for ${train_set}"
        
        # Prepare text for tokenization
        if [ ! -f "data/${train_set}/lm_train.txt" ]; then
            if [ -n "${bpe_train_text}" ]; then
                cp "${bpe_train_text}" "data/${train_set}/lm_train.txt"
            else
                cp "data/${train_set}/text" "data/${train_set}/lm_train.txt"
            fi
        fi
        
        # Create token list based on token type
        if [ "${token_type}" = bpe ]; then
            log "Stage 1: Generate token_list from ${bpe_train_text} using BPE"
            
            mkdir -p "${bpedir}"
            # shellcheck disable=SC2002
            cat ${bpe_train_text} | cut -f 2- -d" " > "${bpedir}"/train.txt
            
            if [ -n "${bpe_nlsyms}" ]; then
                if test -f "${bpe_nlsyms}"; then
                    bpe_nlsyms_list=$(awk '{print $1}' ${bpe_nlsyms} | paste -s -d, -)
                    _opts_spm="--user_defined_symbols=${bpe_nlsyms_list}"
                else
                    _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
                fi
            else
                _opts_spm=""
            fi
            
            spm_train \
                --input="${bpedir}"/train.txt \
                --vocab_size="${nbpe}" \
                --model_type="${bpemode}" \
                --model_prefix="${bpeprefix}" \
                --character_coverage=${bpe_char_cover} \
                --input_sentence_size="${bpe_input_sentence_size}" \
                ${_opts_spm}
            
            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${token_list}"
            
        elif [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 1: Generate character level token_list from data/${train_set}/lm_train.txt"
            
            # Create token list
            ${python} -m espnet2.bin.tokenize_text \
                --token_type "${token_type}" \
                --input "data/${train_set}/lm_train.txt" \
                --output "${token_list}" \
                --field 2- \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"
        fi
        
        log "Token list created: ${token_list}"
    else
        log "Token list already exists: ${token_list}"
    fi
fi

# Stage 2: Collect stats
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Collect stats for batching"
    # Set token list directory and paths (same as in tokenization stage)
    if [ "${lang}" != noinfo ]; then
        token_listdir=data/${lang}_token_list
    else
        token_listdir=data/token_list
    fi
    if [ "${token_type}" = bpe ]; then
        bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
        bpeprefix="${bpedir}"/bpe
        bpemodel="${bpeprefix}".model
        bpetoken_list="${bpedir}"/tokens.txt
        token_list="${bpetoken_list}"
        _stats_tag="${feats_type}_${lang}_bpe${nbpe}"
    elif [ "${token_type}" = char ]; then
        chartoken_list="${token_listdir}"/char/tokens.txt
        token_list="${chartoken_list}"
        bpemodel=none
        _stats_tag="${feats_type}_${lang}_char"
    elif [ "${token_type}" = word ]; then
        wordtoken_list="${token_listdir}"/word/tokens.txt
        token_list="${wordtoken_list}"
        bpemodel=none
        _stats_tag="${feats_type}_${lang}_word"
    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi
    . ./db.sh
    train_data_dir="${NOISY_LIBRISPEECH}"
    valid_data_dir="${NOISY_LIBRISPEECH}"
    if [ ! -d "${train_data_dir}/${train_set}" ]; then
        log "Warning: Noisy data not found, using clean data"
        train_data_dir="data"
        valid_data_dir="data"
    fi
    _statsdir="exp/enh_asr_stats_${_stats_tag}"
    ${python} ${MAIN_ROOT}/espnet2/bin/enh_asr_transducer_train.py \
        --config "${enh_asr_config}" \
        --train_data_path_and_name_and_type "${train_data_dir}/${train_set}/wav.scp,speech_mix,sound" \
        --train_data_path_and_name_and_type "${train_data_dir}/${train_set}/spk1.scp,speech_ref1,sound" \
        --train_data_path_and_name_and_type "${train_data_dir}/${train_set}/text,text,text" \
        --valid_data_path_and_name_and_type "${valid_data_dir}/${valid_set}/wav.scp,speech_mix,sound" \
        --valid_data_path_and_name_and_type "${valid_data_dir}/${valid_set}/spk1.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "${valid_data_dir}/${valid_set}/text,text,text" \
        --token_list "${token_list}" \
        --bpemodel "${bpemodel}" \
        --output_dir "${_statsdir}" \
        --ngpu 0 \
        --collect_stats true \
        --model_conf extract_feats_in_collect_stats=false \
        --num_workers "${nj}"
fi

# Training
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Enhancement + ASR Transducer training"
    log "enh_asr_config: ${enh_asr_config}"
    log "enh_asr_exp: ${enh_asr_exp}"
    
    # Set token list directory and paths (same as in tokenization stage)
    if [ "${lang}" != noinfo ]; then
        token_listdir=data/${lang}_token_list
    else
        token_listdir=data/token_list
    fi
    if [ "${token_type}" = bpe ]; then
        bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
        bpeprefix="${bpedir}"/bpe
        bpemodel="${bpeprefix}".model
        bpetoken_list="${bpedir}"/tokens.txt
        token_list="${bpetoken_list}"
        _stats_tag="${feats_type}_${lang}_bpe${nbpe}"
    elif [ "${token_type}" = char ]; then
        chartoken_list="${token_listdir}"/char/tokens.txt
        token_list="${chartoken_list}"
        bpemodel=none
        _stats_tag="${feats_type}_${lang}_char"
    elif [ "${token_type}" = word ]; then
        wordtoken_list="${token_listdir}"/word/tokens.txt
        token_list="${wordtoken_list}"
        bpemodel=none
        _stats_tag="${feats_type}_${lang}_word"
    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi
    . ./db.sh
    if [ ! -f "${token_list}" ]; then
        log "Error: Token list not found at ${token_list}"
        exit 1
    fi
    train_data_dir="${NOISY_LIBRISPEECH}"
    valid_data_dir="${NOISY_LIBRISPEECH}"
    if [ ! -d "${train_data_dir}/${train_set}" ]; then
        log "Warning: Noisy data not found, using clean data"
        train_data_dir="data"
        valid_data_dir="data"
    fi
    _statsdir="exp/enh_asr_stats_${_stats_tag}"
    # Use cuda_cmd like ASR (run.pl, queue.pl, etc.)
    _launcher="${cuda_cmd}"
    _train_ngpu="${ngpu}"
    # Create output directory
    mkdir -p "${enh_asr_exp}"
    
    log_file="${enh_asr_exp}/train.log"
    
    # Generate run.sh for resuming
    log "Generate '${enh_asr_exp}/run.sh'. You can resume the process from stage 3 using this script"
    mkdir -p "${enh_asr_exp}"; echo "${run_args} --stage 3 \"\$@\"; exit \$?" > "${enh_asr_exp}/run.sh"; chmod +x "${enh_asr_exp}/run.sh"
    
    log "Enhancement + ASR training started... log: '${enh_asr_exp}/train.log'"
    if echo "${_launcher}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${enh_asr_exp})"
    else
        jobname="${enh_asr_exp}/train.log"
    fi
    
    (
    ${python} -m espnet2.bin.launch \
        --cmd "${_launcher} --name ${jobname}" \
        --log "${log_file}" \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${enh_asr_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.enh_asr_transducer_train \
            --config "${enh_asr_config}" \
            --train_data_path_and_name_and_type "${train_data_dir}/${train_set}/wav.scp,speech_mix,sound" \
            --train_data_path_and_name_and_type "${train_data_dir}/${train_set}/spk1.scp,speech_ref1,sound" \
            --train_data_path_and_name_and_type "${train_data_dir}/${train_set}/text,text,text" \
            --valid_data_path_and_name_and_type "${valid_data_dir}/${valid_set}/wav.scp,speech_mix,sound" \
            --valid_data_path_and_name_and_type "${valid_data_dir}/${valid_set}/spk1.scp,speech_ref1,sound" \
            --valid_data_path_and_name_and_type "${valid_data_dir}/${valid_set}/text,text,text" \
            --token_list "${token_list}" \
            --bpemodel "${bpemodel}" \
            --output_dir "${enh_asr_exp}" \
            --ngpu "${_train_ngpu}" \
            --num_workers "${nj}" \
            --pretrained_asr_model "${pretrained_asr_model}" \
            --train_shape_file "${_statsdir}/train/speech_mix_shape" \
            --valid_shape_file "${_statsdir}/valid/speech_mix_shape"
    )
fi

# Inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Enhancement + ASR Transducer inference"
    for dset in ${test_sets}; do
        ${python} ${MAIN_ROOT}/espnet2/bin/enh_asr_transducer_inference.py \
            --config "${inference_config}" \
            --model_path "${enh_asr_exp}/${inference_enh_asr_model}" \
            --data_path_and_name_and_type "data/${dset}/wav.scp,speech_mix,sound" \
            --output_dir "${enh_asr_exp}/inference/${dset}" \
            --ngpu "${ngpu}" \
            --num_workers "${inference_nj}"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]" 