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

channel_mode=allch

. "${recipe_dir}/../asr1/utils/parse_options.sh"

log "$0 $*"

src_root_enh="${recipe_dir}/../enh1/data"
src_root_asr="${recipe_dir}/../asr1/data"

copy_set() {
    local src=$1
    local dst=$2
    mkdir -p "${dst}"
    cp "${src}/wav.scp" "${dst}/wav.scp"
    cp "${src}/spk1.scp" "${dst}/spk1.scp"
    cp "${src}/text" "${dst}/text"
    cp "${src}/utt2spk" "${dst}/utt2spk"
    [ -f "${src}/utt2lang" ] && cp "${src}/utt2lang" "${dst}/utt2lang"
    (
        cd "${recipe_dir}/../asr1"
        utils/fix_data_dir.sh "${recipe_dir}/${dst}" >/dev/null
    )
}

combine_sets() {
    local dst=$1
    shift
    (
        cd "${recipe_dir}/../asr1"
        utils/combine_data.sh --extra-files "spk1.scp text utt2lang" "${recipe_dir}/data/${dst}" "$@"
        utils/fix_data_dir.sh "${recipe_dir}/data/${dst}" >/dev/null
    )
}

if [ "${channel_mode}" = allch ]; then
    train_set=tr05_multi_mixed_track
    valid_set=dt05_multi_mixed_track
    test_sets="et05_multi_mixed_track et05_simu_snr-5_allch_track et05_simu_snr0_allch_track et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    train_set=tr05_multi_isolated_1ch_track
    valid_set=dt05_simu_isolated_1ch_track
    test_sets="et05_multi_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

if [ "${channel_mode}" = allch ]; then
    copy_set "${src_root_enh}/tr05_simu_allch_track" "data/tr05_simu_allch_track"
    copy_set "${src_root_enh}/tr05_real_isolated_1ch_track" "data/tr05_real_isolated_1ch_track"
    copy_set "${src_root_enh}/dt05_simu_allch_track" "data/dt05_simu_allch_track"
    copy_set "${src_root_enh}/dt05_real_isolated_1ch_track" "data/dt05_real_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_allch_track" "data/et05_simu_allch_track"
    copy_set "${src_root_enh}/et05_real_isolated_1ch_track" "data/et05_real_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_snr-5_allch_track" "data/et05_simu_snr-5_allch_track"
    copy_set "${src_root_enh}/et05_simu_snr0_allch_track" "data/et05_simu_snr0_allch_track"
    copy_set "${src_root_enh}/et05_simu_snr5_allch_track" "data/et05_simu_snr5_allch_track"
    copy_set "${src_root_enh}/et05_simu_snr10_allch_track" "data/et05_simu_snr10_allch_track"
    copy_set "${src_root_enh}/et05_simu_snr15_allch_track" "data/et05_simu_snr15_allch_track"

    combine_sets "tr05_multi_mixed_track" \
        "${recipe_dir}/data/tr05_simu_allch_track" \
        "${recipe_dir}/data/tr05_real_isolated_1ch_track"
    combine_sets "dt05_multi_mixed_track" \
        "${recipe_dir}/data/dt05_simu_allch_track" \
        "${recipe_dir}/data/dt05_real_isolated_1ch_track"
    combine_sets "et05_multi_mixed_track" \
        "${recipe_dir}/data/et05_simu_allch_track" \
        "${recipe_dir}/data/et05_real_isolated_1ch_track"
else
    copy_set "${src_root_enh}/${train_set}" "data/${train_set}"
    copy_set "${src_root_enh}/dt05_multi_isolated_1ch_track" "data/dt05_multi_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_multi_isolated_1ch_track" "data/et05_multi_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_snr-5_isolated_1ch_track" "data/et05_simu_snr-5_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_snr0_isolated_1ch_track" "data/et05_simu_snr0_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_snr5_isolated_1ch_track" "data/et05_simu_snr5_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_snr10_isolated_1ch_track" "data/et05_simu_snr10_isolated_1ch_track"
    copy_set "${src_root_enh}/et05_simu_snr15_isolated_1ch_track" "data/et05_simu_snr15_isolated_1ch_track"
fi

mkdir -p data/en_token_list
[ -d "${src_root_asr}/en_token_list" ] && cp -r "${src_root_asr}/en_token_list/." data/en_token_list/
[ -f "${src_root_asr}/nlsyms.txt" ] && cp "${src_root_asr}/nlsyms.txt" data/nlsyms.txt || true

log "Successfully finished."
