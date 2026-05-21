#!/usr/bin/env bash
# Data preparation for CHiME4 JEPA_asr1 (SE+ASR).
# Reuses enh1 data: create symlinks from ../enh1/data and build nlsyms.
# Run ../enh1/local/data.sh first to prepare enh1 data.

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${LINENO}) $*"
}

stage=1
stop_stage=2
enh1_data=../enh1/data

. utils/parse_options.sh || true

. ./path.sh || exit 1
. ./cmd.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Symlink data from enh1 (${enh1_data})"
    if [ ! -d "${enh1_data}" ]; then
        log "Error: Run enh1 data preparation first: cd egs2/chime4/enh1 && local/data.sh ..."
        exit 1
    fi
    mkdir -p data
    # multi + simu-only + per-SNR et05 test sets (et05_simu_snr* from MATLAB simu)
    for d in tr05_multi_isolated_1ch_track dt05_multi_isolated_1ch_track et05_multi_isolated_1ch_track \
             tr05_simu_isolated_1ch_track dt05_simu_isolated_1ch_track et05_simu_isolated_1ch_track \
             et05_simu_snr-10_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
             et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track; do
        if [ -d "${enh1_data}/${d}" ]; then
            rm -f "data/${d}"
            ln -sf "../enh1/data/${d}" "data/${d}"
            log "Linked data/${d} -> ../enh1/data/${d}"
        else
            log "Warning: ${enh1_data}/${d} not found, skipping"
        fi
    done
fi

nlsyms=data/nlsyms.txt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Create nlsyms from training text"
    _train_text=""
    for d in data/tr05_multi_isolated_1ch_track data/tr05_simu_isolated_1ch_track; do
        [ -f "${d}/text" ] && _train_text="${d}/text" && break
    done
    if [ -n "${_train_text}" ]; then
        cut -f 2- "${_train_text}" | tr " " "\n" | sort -u | grep "^<" > "${nlsyms}" || true
        [ -s "${nlsyms}" ] || echo "<NOISE>" > "${nlsyms}"
        log "Created ${nlsyms}"
    else
        log "Warning: data/tr05_simu_isolated_1ch_track/text not found; creating minimal nlsyms"
        echo "<NOISE>" > "${nlsyms}"
    fi
fi

log "JEPA_asr1 data preparation done."
