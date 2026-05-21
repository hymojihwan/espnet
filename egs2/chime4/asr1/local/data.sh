#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

link_optional_data_dir() {
    local src="$1"
    local dst="$2"
    if [ -d "${src}" ] && [ ! -e "${dst}" ]; then
        ln -s "${src}" "${dst}"
        log "Linked optional data dir: ${dst} -> ${src}"
    fi
}

sync_data_dir_from_enh1() {
    local src="$1"
    local dst="$2"
    if [ -d "${src}" ] && [ -s "${src}/wav.scp" ] && [ -s "${src}/text" ] && [ -s "${src}/utt2spk" ]; then
        rm -rf "${dst}"
        ln -s "${src}" "${dst}"
        log "Synced data dir from enh1: ${dst} -> ${src}"
    fi
}

sync_data_dir_from_source() {
    local primary_src="$1"
    local fallback_src="$2"
    local dst="$3"

    if [ -d "${primary_src}" ] && [ -s "${primary_src}/wav.scp" ] && [ -s "${primary_src}/text" ] && [ -s "${primary_src}/utt2spk" ]; then
        rm -rf "${dst}"
        ln -s "${primary_src}" "${dst}"
        log "Synced data dir: ${dst} -> ${primary_src}"
    elif [ -d "${fallback_src}" ] && [ -s "${fallback_src}/wav.scp" ] && [ -s "${fallback_src}/text" ] && [ -s "${fallback_src}/utt2spk" ]; then
        rm -rf "${dst}"
        ln -s "${fallback_src}" "${dst}"
        log "Synced data dir from fallback: ${dst} -> ${fallback_src}"
    fi
}

link_clean_ref_from_enh_asr1() {
    local src="$1"
    local dst="$2"
    local clean_src="${src}/spk1.scp"
    local clean_dst="${dst}/clean_speech.scp"
    if [ -f "${clean_src}" ]; then
        rm -f "${clean_dst}"
        ln -s "${clean_src}" "${clean_dst}"
        log "Linked clean reference: ${clean_dst} -> ${clean_src}"
    fi
}

subset_by_id_file() {
    local src="$1"
    local ids="$2"
    local dst="$3"
    [ -f "${src}" ] || return 0
    grep -F -f "${ids}" "${src}" > "${dst}"
}

prepare_ch1_subset() {
    local src_dir="$1"
    local dst_dir="$2"
    local ids
    ids="$(mktemp)"

    rm -rf "${dst_dir}"
    mkdir -p "${dst_dir}"

    awk '$1 ~ /\.CH1_/ {print $1}' "${src_dir}/wav.scp" > "${ids}"
    subset_by_id_file "${src_dir}/wav.scp" "${ids}" "${dst_dir}/wav.scp"
    subset_by_id_file "${src_dir}/text" "${ids}" "${dst_dir}/text"
    subset_by_id_file "${src_dir}/utt2spk" "${ids}" "${dst_dir}/utt2spk"
    for f in utt2lang segments reco2file_and_channel; do
        subset_by_id_file "${src_dir}/${f}" "${ids}" "${dst_dir}/${f}"
    done
    utils/fix_data_dir.sh "${dst_dir}" >/dev/null
    rm -f "${ids}"
    log "Prepared ${dst_dir} from CH1 subset of ${src_dir}"
}

prune_bad_wavs() {
    local d="$1"
    [ -f "${d}/wav.scp" ] || return 0

    local tmp_keep tmp_bad
    tmp_keep="$(mktemp)"
    tmp_bad="$(mktemp)"
    local known_bad_re='014_014C0211_BUS\.CH1\.wav'

    while read -r utt wavpath; do
        [ -n "${utt}" ] || continue
        if [[ "${wavpath}" =~ ${known_bad_re} ]]; then
            printf '%s\n' "${utt}" >> "${tmp_bad}"
        else
            printf '%s %s\n' "${utt}" "${wavpath}" >> "${tmp_keep}"
        fi
    done < "${d}/wav.scp"

    if [ -s "${tmp_bad}" ]; then
        local nbad
        nbad=$(wc -l < "${tmp_bad}")
        log "Removing ${nbad} unreadable wav entries from ${d}"
        cp "${d}/wav.scp" "${d}/wav.scp.bak_before_prune"
        mv "${tmp_keep}" "${d}/wav.scp"
        for f in text utt2spk utt2lang segments feats.scp vad.scp reco2file_and_channel utt2uniq utt2dur utt2num_frames; do
            if [ -f "${d}/${f}" ]; then
                grep -F -v -f "${tmp_bad}" "${d}/${f}" > "${d}/${f}.tmp" || true
                mv "${d}/${f}.tmp" "${d}/${f}"
            fi
        done
        utils/fix_data_dir.sh "${d}" >/dev/null
    else
        rm -f "${tmp_keep}"
    fi
    rm -f "${tmp_bad}"
}


stage=0
stop_stage=2
train_dev=dt05_multi_isolated_1ch_track
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

has_beamformit=false
if command -v BeamformIt >/dev/null 2>&1; then
    has_beamformit=true
elif [ -n "${BEAMFORMIT:-}" ] && [ -x "${BEAMFORMIT}/BeamformIt" ]; then
    has_beamformit=true
elif [ -n "${KALDI_ROOT:-}" ] && [ -x "${KALDI_ROOT}/tools/BeamformIt/BeamformIt" ]; then
    has_beamformit=true
fi



if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi

# WSJ1 is optional: if missing, skip WSJ clean-data augmentation and LM text.
use_wsj1=false
if [ -n "${WSJ1:-}" ] && [ -e "${WSJ1}" ]; then
    use_wsj1=true
fi

if [ ! -e "${CHIME4}" ]; then
    log "Fill the value of 'CHIME4' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    wsj0_data=${CHIME4}/data/WSJ0
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh

    # create data for 1ch and 2ch tracks
    if [ ! -d ${CHIME4}/data/audio/16kHz/isolated_1ch_track ]; then
        log "create data for 1ch tracks"
        python local/sym_channel.py ${CHIME4} 1ch
    fi

    if [ ! -d ${CHIME4}/data/audio/16kHz/isolated_2ch_track ]; then
        log "create data for 2ch tracks"
        python local/sym_channel.py ${CHIME4} 2ch
    fi

    # beamforming for multich
    if "${has_beamformit}"; then
        local/run_beamform_2ch_track.sh --cmd "${train_cmd}" --nj 20 \
	        ${CHIME4}/data/audio/16kHz/isolated_2ch_track enhan/beamformit_2mics
        local/run_beamform_6ch_track.sh --cmd "${train_cmd}" --nj 20 \
	        ${CHIME4}/data/audio/16kHz/isolated_6ch_track enhan/beamformit_5mics
    else
        log "BeamformIt not found; skipping 2ch/5ch beamforming data preparation"
    fi

    # preparation for chime4 data
    local/real_noisy_chime4_data_prep.sh ${CHIME4}
    local/simu_noisy_chime4_data_prep.sh ${CHIME4}

    # test data for 1ch track
    local/real_enhan_chime4_data_prep.sh isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track
    local/simu_enhan_chime4_data_prep.sh isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track

    # test data for 2ch/6ch tracks
    if "${has_beamformit}"; then
        local/real_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics
        local/simu_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics

        local/real_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
        local/simu_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
    fi

    # Additionally use WSJ clean data when WSJ1 is available.
    if "${use_wsj1}"; then
        local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
        local/wsj_format_data.sh
    else
        log "WSJ1 not set or missing; skipping wsj_data_prep/wsj_format_data"
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "combine real and simulation data"

    # TO DO:--extra-files but no utt2num_frames
    utils/combine_data.sh data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy
    if "${use_wsj1}" && [ -d data/train_si284 ]; then
        utils/combine_data.sh data/tr05_multi_noisy_si284 data/tr05_multi_noisy data/train_si284
    else
        log "Skipping tr05_multi_noisy_si284 because WSJ1/train_si284 is unavailable"
    fi
    utils/combine_data.sh data/${train_dev} data/dt05_simu_isolated_1ch_track data/dt05_real_isolated_1ch_track

    enh1_data_root="${PWD}/../enh1/data"
    enh_asr1_data_root="${PWD}/../enh_asr1/data"

    # For train/valid simulated sets, keep the local noisy-data preparation as
    # the source of truth and only attach clean references from enh_asr1.
    if [ -f data/tr05_simu_noisy/wav.scp ]; then
        prepare_ch1_subset data/tr05_simu_noisy data/tr05_simu_isolated_1ch_track
    fi
    if [ -f data/dt05_simu_noisy/wav.scp ] && [ ! -s data/dt05_simu_isolated_1ch_track/wav.scp ]; then
        prepare_ch1_subset data/dt05_simu_noisy data/dt05_simu_isolated_1ch_track
    fi
    for dset in \
        tr05_simu_isolated_1ch_track \
        dt05_simu_isolated_1ch_track \
        tr05_simu_allch_track \
        dt05_simu_allch_track; do
        link_clean_ref_from_enh_asr1 "${enh_asr1_data_root}/${dset}" "data/${dset}"
    done

    # Reuse / sync evaluation simulated sets prepared in enh1 when available.
    for dset in \
        et05_simu_isolated_1ch_track \
        et05_simu_allch_track \
        et05_simu_snr-5_isolated_1ch_track \
        et05_simu_snr0_isolated_1ch_track \
        et05_simu_snr5_isolated_1ch_track \
        et05_simu_snr10_isolated_1ch_track \
        et05_simu_snr15_isolated_1ch_track \
        et05_simu_snr-5_allch_track \
        et05_simu_snr0_allch_track \
        et05_simu_snr5_allch_track \
        et05_simu_snr10_allch_track \
        et05_simu_snr15_allch_track; do
        sync_data_dir_from_enh1 "${enh1_data_root}/${dset}" "data/${dset}"
        link_clean_ref_from_enh_asr1 "${enh_asr1_data_root}/${dset}" "data/${dset}"
    done

    # Keep older optional-link behavior for any remaining eval sets.
    for dset in \
        dt05_simu_allch_track \
        et05_simu_allch_track \
        et05_simu_snr-5_allch_track \
        et05_simu_snr0_allch_track \
        et05_simu_snr5_allch_track \
        et05_simu_snr10_allch_track \
        et05_simu_snr15_allch_track \
        et05_simu_snr-5_isolated_1ch_track \
        et05_simu_snr0_isolated_1ch_track \
        et05_simu_snr5_isolated_1ch_track \
        et05_simu_snr10_isolated_1ch_track \
        et05_simu_snr15_isolated_1ch_track; do
        link_optional_data_dir "${enh1_data_root}/${dset}" "data/${dset}"
        link_clean_ref_from_enh_asr1 "${enh_asr1_data_root}/${dset}" "data/${dset}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for d in data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy \
             data/dt05_multi_isolated_1ch_track data/dt05_simu_isolated_1ch_track data/dt05_real_isolated_1ch_track \
             data/et05_simu_isolated_1ch_track data/et05_real_isolated_1ch_track; do
        [ -d "${d}" ] && prune_bad_wavs "${d}"
    done
fi

other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ -d data/tr05_multi_noisy ]; then
        ids=data/tr05_multi_noisy_ch1.ids
        rm -rf data/tr05_multi_noisy_ch1
        mkdir -p data/tr05_multi_noisy_ch1
        awk '$1 ~ /\.CH1_/ {print $1}' data/tr05_multi_noisy/wav.scp > "${ids}"
        subset_by_id_file data/tr05_multi_noisy/wav.scp "${ids}" data/tr05_multi_noisy_ch1/wav.scp
        subset_by_id_file data/tr05_multi_noisy/text "${ids}" data/tr05_multi_noisy_ch1/text
        subset_by_id_file data/tr05_multi_noisy/utt2spk "${ids}" data/tr05_multi_noisy_ch1/utt2spk
        utils/fix_data_dir.sh data/tr05_multi_noisy_ch1 >/dev/null
        rm -f "${ids}"
        log "Prepared data/tr05_multi_noisy_ch1 from CH1 subset of tr05_multi_noisy"
    fi

    log "stage 2: Srctexts preparation"

    mkdir -p "$(dirname ${other_text})"

    if "${use_wsj1}"; then
        # NOTE(kamo): Give utterance id to each texts.
        zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
	        grep -v "<" | tr "[:lower:]" "[:upper:]" | \
	        awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}
    else
        touch "${other_text}"
        log "WSJ1 not used; other_text left empty"
    fi

    log "Create non linguistic symbols: ${nlsyms}"
    if [ -f data/train_si284/text ]; then
        cut -f 2- data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    else
        for f in data/tr05_multi_noisy/text data/tr05_simu_noisy/text data/tr05_real_noisy/text; do
            if [ -f "$f" ]; then
                cut -f 2- "$f" | tr " " "\n" | sort -u | grep "^<" > "${nlsyms}" || true
                break
            fi
        done
        [ -s "${nlsyms}" ] || echo "<NOISE>" > "${nlsyms}"
        log "nlsyms from CHiME4 train text (no WSJ1)"
    fi
    cat ${nlsyms}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
