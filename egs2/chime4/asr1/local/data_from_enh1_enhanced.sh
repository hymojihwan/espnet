#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

channel_mode="${CHANNEL_MODE:-ch1}"
enh_exp="${ENH_EXP:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../enh1/exp/enh_train_enh_convtasnet_small_raw" && pwd)}"
inference_tag="${INFERENCE_TAG:-enhanced_for_asr}"
suffix="${SUFFIX:-enhsmall}"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr1_dir="$(cd "${script_dir}/.." && pwd)"
enh1_data_root="$(cd "${script_dir}/../../enh1/data" && pwd)"

copy_set() {
    local src_data="$1"
    local src_enh="$2"
    local dst="$3"

    [ -d "${src_data}" ] || { log "missing source data dir: ${src_data}"; exit 1; }
    [ -f "${src_enh}/spk1.scp" ] || { log "missing enhanced scp: ${src_enh}/spk1.scp"; exit 1; }

    rm -rf "${dst}"
    mkdir -p "${dst}"
    cp "${src_enh}/spk1.scp" "${dst}/wav.scp"

    for f in text utt2spk spk2utt utt2lang; do
        if [ -f "${src_data}/${f}" ]; then
            cp "${src_data}/${f}" "${dst}/${f}"
        fi
    done

    echo "raw" > "${dst}/feats_type"
    echo "wav" > "${dst}/audio_format"
    utils/fix_data_dir.sh "${dst}" >/dev/null
    log "Prepared ${dst}"
}

if [ "${channel_mode}" = allch ]; then
    train_base=tr05_simu_allch_track
    valid_base=dt05_simu_allch_track
    test_sets="\
et05_simu_allch_track \
et05_simu_snr-5_allch_track et05_simu_snr0_allch_track \
et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    train_base=tr05_simu_isolated_1ch_track
    valid_base=dt05_simu_isolated_1ch_track
    test_sets="\
et05_multi_isolated_1ch_track \
et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

copy_set \
    "${enh1_data_root}/${train_base}" \
    "${enh_exp}/${inference_tag}_${train_base}" \
    "${asr1_dir}/data/${train_base}_${suffix}"

copy_set \
    "${enh1_data_root}/${valid_base}" \
    "${enh_exp}/${inference_tag}_${valid_base}" \
    "${asr1_dir}/data/${valid_base}_${suffix}"

for dset in ${test_sets}; do
    copy_set \
        "${enh1_data_root}/${dset}" \
        "${enh_exp}/${inference_tag}_${dset}" \
        "${asr1_dir}/data/${dset}_${suffix}"
done

log "Done."
