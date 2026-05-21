#!/usr/bin/env bash
# ASR1 data prep: same pipeline as enh1 (CHIME3/CHIME4 + simulation), no WSJ1.
# Output has ASR-style text (utt_id transcript) in data/*/text; use symlinks from enh1/data.

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 --extra-annotations <path> [--stage <stage>] [--stop_stage <stop_stage>] [--nj <nj>] [--simulate-valid-only true]

  Prepare data the same way as enh1 (no WSJ1): simulation + prep with text targets.
  Then symlink enh1/data/* into asr1/data so ASR has wav.scp, text, utt2spk, etc.

  required argument:
    --extra-annotations: path to CHiME4 extra annotations (same as enh1).

  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
    [--nj]: number of parallel workers for MATLAB
    [--simulate-valid-only true]: only regenerate dt05 simulation (see enh1).
EOF
)

stage=1
stop_stage=2
extra_annotations=
nj=32
simulate_valid_only=false
log "$0 $*"
. utils/parse_options.sh

if [ -z "${extra_annotations}" ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ ! -e "${CHIME3}" ]; then
    log "Fill the value of 'CHIME3' in db.sh"
    exit 1
fi
if [ ! -e "${CHIME4}" ]; then
    log "Fill the value of 'CHIME4' in db.sh"
    exit 1
fi

enh1_dir="${PWD}/../enh1"
if [ ! -d "${enh1_dir}" ] || [ ! -f "${enh1_dir}/local/data.sh" ]; then
    log "Error: enh1 not found at ${enh1_dir}; need enh1/local/data.sh"
    exit 1
fi

# Run enh1 data prep (same pipeline, keeps text in data/*/text)
log "Running enh1 data preparation (stage ${stage}..${stop_stage})"
run_opts=( --extra-annotations "${extra_annotations}" --stage ${stage} --stop_stage ${stop_stage} --nj ${nj} )
if "${simulate_valid_only}"; then
    run_opts+=( --simulate-valid-only true )
fi
( cd "${enh1_dir}" && ./local/data.sh "${run_opts[@]}" ) || exit 1

# Symlink enh1/data into asr1/data (target relative to link dir: data/ -> ../../enh1/data/)
log "Symlinking enh1/data into asr1/data (text and wav.scp preserved)"
mkdir -p data
for d in tr05_simu_isolated_1ch_track dt05_simu_isolated_1ch_track et05_simu_isolated_1ch_track \
         et05_multi_isolated_1ch_track \
         et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
         et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track; do
    if [ -d "${enh1_dir}/data/${d}" ]; then
        rm -f "data/${d}"
        ln -sf "../../enh1/data/${d}" "data/${d}"
        log "  data/${d} -> ../../enh1/data/${d}"
    fi
done

if [ ! -f "data/tr05_simu_isolated_1ch_track/text" ]; then
    log "Error: data/tr05_simu_isolated_1ch_track/text not found after symlink"
    exit 1
fi
log "Data preparation (enh1-style with text targets) finished."
