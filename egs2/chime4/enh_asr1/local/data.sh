#!/usr/bin/env bash
# CHiME4 enh_asr1 data: same preparation as asr1/run_ctc.sh (enh1 data + absolute paths + CH1-only + nlsyms).
# Optional: run enh1/local/data.sh first (same as asr1/local/data_enh1_style.sh).
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(
    cat << EOF
Usage: $0 [options]

  Same recipe as ../asr1/run_ctc.sh (copy from enh1, fix wav/spk scp, CH2–CH6 -> CH1) plus enh_asr extras (text_spk1, utt2lang).

  Optional (same as ../asr1/local/data_enh1_style.sh): run MATLAB enh1 simulation/prep first
    --extra-annotations <path>   CHiME4 annotations path (if set, runs ../enh1/local/data.sh)
    --stage <n>                  enh1 stage (default: 1)
    --stop_stage <n>             enh1 stop stage (default: 2)
    --nj <n>                     enh1 MATLAB nj (default: 32)
    --simulate-valid-only true   forward to enh1 (optional)

  Paths:
    --enh1-data-root <dir>       Default: <recipe>/../enh1/data
EOF
)

extra_annotations=
stage=1
stop_stage=2
nj=32
simulate_valid_only=false
enh1_data_root=

train_set=tr05_simu_isolated_1ch_track

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    log "Error: unexpected arguments: $*"
    exit 2
fi

. ./path.sh || exit 1
. ./cmd.sh || exit 1

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_recipe_dir="$(cd "${_script_dir}/.." && pwd)"
enh1_repo="$(cd "${_recipe_dir}/../enh1" && pwd)"

# Optional: full enh1 pipeline (same as asr1 local/data_enh1_style.sh)
if [ -n "${extra_annotations}" ]; then
    # Use corpus paths (override TEMPLATE db: cp ../asr1/db.sh ./db.sh in enh_asr1, or export CHIME3/CHIME4).
    . ./db.sh || exit 1
    if [ ! -e "${CHIME3:-}" ] || [ ! -e "${CHIME4:-}" ]; then
        log "Set CHIME3 and CHIME4 (e.g. use ../asr1/db.sh as db.sh) for --extra-annotations prep"
        exit 1
    fi
    if [ ! -f "${enh1_repo}/local/data.sh" ]; then
        log "enh1 not found at ${enh1_repo}"
        exit 1
    fi
    log "Running enh1/local/data.sh (stage ${stage}..${stop_stage})"
    _run_opts=(--extra-annotations "${extra_annotations}" --stage "${stage}" --stop_stage "${stop_stage}" --nj "${nj}")
    if "${simulate_valid_only}"; then
        _run_opts+=(--simulate-valid-only true)
    fi
    (cd "${enh1_repo}" && ./local/data.sh "${_run_opts[@]}") || exit 1
fi

if [ -z "${enh1_data_root}" ]; then
    enh1_data_root="${enh1_repo}/data"
fi
if [ ! -d "${enh1_data_root}" ]; then
    log "enh1 data not found: ${enh1_data_root}"
    log "Use --extra-annotations <path> or prepare ../enh1/data first."
    exit 1
fi
enh1_data_root="$(cd "${enh1_data_root}" && pwd)"
# Prefix for relative paths in scp (parent of enh1 data dir; matches ../asr1/run_ctc.sh when data is <enh1>/data)
_pre="$(cd "$(dirname "${enh1_data_root}")" && pwd)/"

mkdir -p "${_recipe_dir}/data"

_fix_scp() {
    local f="$1"
    [ -f "${f}" ] || return 0
    awk -v pre="${_pre}" 'NF>=2 && $2 !~ /^\// { $2 = pre $2 } { print }' "${f}" > "${f}.tmp"
    perl -0pi -e 's/\.CH[2-6](\.(?:Clean|Noise))?\.wav/.CH1$1.wav/g' "${f}.tmp"
    mv "${f}.tmp" "${f}"
}

_subset_by_id_file() {
    local src="$1"
    local ids="$2"
    local dst="$3"
    [ -f "${src}" ] || return 0
    grep -F -f "${ids}" "${src}" > "${dst}"
}

# Identical set list as ../asr1/run_ctc.sh
for d in \
    tr05_simu_isolated_1ch_track tr05_real_isolated_1ch_track tr05_multi_isolated_1ch_track \
    tr05_simu_allch_track \
    dt05_simu_isolated_1ch_track dt05_real_isolated_1ch_track dt05_multi_isolated_1ch_track \
    dt05_simu_allch_track \
    et05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_multi_isolated_1ch_track \
    et05_simu_allch_track \
    et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
    et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track \
    et05_simu_snr-5_allch_track et05_simu_snr0_allch_track \
    et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track; do
    if [ ! -d "${enh1_data_root}/${d}" ]; then
        continue
    fi
    log "Copy ${d} from ${enh1_data_root}/${d}"
    rm -rf "${_recipe_dir}/data/${d}"
    cp -a "${enh1_data_root}/${d}" "${_recipe_dir}/data/${d}"

    _fix_scp "${_recipe_dir}/data/${d}/wav.scp"
    _fix_scp "${_recipe_dir}/data/${d}/spk1.scp"
    _fix_scp "${_recipe_dir}/data/${d}/noise1.scp"

    if [ -f "${_recipe_dir}/data/${d}/text" ]; then
        LC_ALL=C sort -k1,1 "${_recipe_dir}/data/${d}/text" > "${_recipe_dir}/data/${d}/text.tmp"
        mv "${_recipe_dir}/data/${d}/text.tmp" "${_recipe_dir}/data/${d}/text"
        cp -f "${_recipe_dir}/data/${d}/text" "${_recipe_dir}/data/${d}/text_spk1"
    else
        log "Warning: ${d} has no text"
    fi
    if [ ! -f "${_recipe_dir}/data/${d}/utt2lang" ]; then
        awk '{ print $1, "en" }' "${_recipe_dir}/data/${d}/wav.scp" > "${_recipe_dir}/data/${d}/utt2lang"
    fi
    utils/fix_data_dir.sh "${_recipe_dir}/data/${d}" >/dev/null
done

# Fairer comparison target for enh_asr1 vs asr1:
# build a CH1-only view of asr1's tr05_multi_noisy while keeping the clean
# reference from enh1 tr05_multi_isolated_1ch_track.
asr1_data_root="${_recipe_dir}/../asr1/data"
if [ -d "${asr1_data_root}/tr05_multi_noisy" ] && [ -d "${_recipe_dir}/data/tr05_multi_isolated_1ch_track" ]; then
    dst="${_recipe_dir}/data/tr05_multi_noisy_ch1"
    ids="${dst}.ids"
    rm -rf "${dst}"
    mkdir -p "${dst}"
    awk '$1 ~ /\.CH1_/ {print $1}' "${asr1_data_root}/tr05_multi_noisy/wav.scp" > "${ids}"
    _subset_by_id_file "${asr1_data_root}/tr05_multi_noisy/wav.scp" "${ids}" "${dst}/wav.scp"
    _subset_by_id_file "${asr1_data_root}/tr05_multi_noisy/text" "${ids}" "${dst}/text"
    _subset_by_id_file "${asr1_data_root}/tr05_multi_noisy/utt2spk" "${ids}" "${dst}/utt2spk"
    LC_ALL=C sort -k1,1 "${dst}/text" > "${dst}/text.tmp"
    mv "${dst}/text.tmp" "${dst}/text"
    cp -f "${dst}/text" "${dst}/text_spk1"
    awk '{print $1, "en"}' "${dst}/wav.scp" > "${dst}/utt2lang"
    _subset_by_id_file "${_recipe_dir}/data/tr05_multi_isolated_1ch_track/spk1.scp" "${ids}" "${dst}/spk1.scp"
    if [ -f "${_recipe_dir}/data/tr05_multi_isolated_1ch_track/noise1.scp" ]; then
        _subset_by_id_file "${_recipe_dir}/data/tr05_multi_isolated_1ch_track/noise1.scp" "${ids}" "${dst}/noise1.scp"
    fi
    utils/fix_data_dir.sh "${dst}" >/dev/null
    rm -f "${ids}"
    log "Prepared ${dst##*/} from asr1/tr05_multi_noisy (CH1 subset) + enh1 clean refs"
fi

if [ ! -f "${_recipe_dir}/data/${train_set}/text" ]; then
    log "Error: ${_recipe_dir}/data/${train_set}/text missing after copy."
    exit 1
fi

# Same nlsyms recipe as ../asr1/run_ctc.sh
nlsyms="${_recipe_dir}/data/nlsyms.txt"
cut -f 2- "${_recipe_dir}/data/${train_set}/text" | tr " " "\n" | sort -u | grep "^<" > "${nlsyms}" || true
if [ ! -s "${nlsyms}" ]; then
    echo "<NOISE>" > "${nlsyms}"
fi

log "Done (asr1-style). nlsyms -> ${nlsyms}"
