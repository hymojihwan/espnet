#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
recipe_dir="$(cd "${script_dir}/.." && pwd)"
cd "${recipe_dir}"

. ./path.sh
. ./cmd.sh
. ../asr1/utils/parse_options.sh

channel_mode="${CHANNEL_MODE:-allch}"
speed_perturb_factors="${SPEED_PERTURB_FACTORS:-0.9 1.0 1.1}"
feature_suffix="${FEATURE_SUFFIX:-_mel80enh}"
ngpu="${NGPU:-4}"
settle_seconds="${SETTLE_SECONDS:-120}"
poll_seconds="${POLL_SECONDS:-60}"
run_stage4="${RUN_STAGE4_CMD:-./run.sh --stage 4 --stop_stage 13 --ngpu ${ngpu}}"

if [ "${channel_mode}" = allch ]; then
    base_train_set=tr05_multi_mixed_track
    valid_set=dt05_multi_mixed_track
    test_sets="et05_multi_mixed_track et05_simu_snr-5_allch_track et05_simu_snr0_allch_track et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    base_train_set=tr05_multi_isolated_1ch_track
    valid_set=dt05_multi_isolated_1ch_track
    test_sets="et05_multi_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

if [ -n "${speed_perturb_factors}" ]; then
    train_set="${base_train_set}_sp${feature_suffix}"
else
    train_set="${base_train_set}${feature_suffix}"
fi

all_sets="${train_set} ${valid_set}${feature_suffix}"
for dset in ${test_sets}; do
    all_sets="${all_sets} ${dset}${feature_suffix}"
done

feature_ready() {
    local dset=$1
    [ -f "data/${dset}/feats.scp" ] || return 1
    [ -s "data/${dset}/feats.scp" ] || return 1
}

all_ready() {
    local dset
    for dset in ${all_sets}; do
        feature_ready "${dset}" || return 1
    done
    return 0
}

log "Waiting for stage 3 feature dump to finish"
log "Expected feature sets: ${all_sets}"

while true; do
    if pgrep -f 'dump_logmel_features.py' >/dev/null; then
        sleep "${poll_seconds}"
        continue
    fi

    if all_ready; then
        log "All expected feature dumps are present; waiting ${settle_seconds}s to ensure stage 3 is settled"
        sleep "${settle_seconds}"
        if ! pgrep -f 'dump_logmel_features.py' >/dev/null && all_ready; then
            break
        fi
    else
        sleep "${poll_seconds}"
    fi
done

log "Stage 3 finished. Starting stage 4 ASR fine-tuning"
log "Command: ${run_stage4}"
eval "${run_stage4}"
