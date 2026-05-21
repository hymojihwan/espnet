#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

create_allch_single_set() {
    local split="$1"
    local out_name="$2"
    local flist text_src
    flist="$(mktemp)"
    text_src="data/${split}_simu_isolated_1ch_track/text"

    find "${odir}/audio/16kHz/isolated" -name '*.wav' \
        | grep "${split}_bus_simu\|${split}_caf_simu\|${split}_ped_simu\|${split}_str_simu" \
        | grep -v '_snr' | sort -u > "${flist}"

    mkdir -p "data/${out_name}"
    awk -F'/' '{print $NF}' "${flist}" | sed 's/\.wav/_SIMU/' > "${flist}.ids"
    paste -d" " "${flist}.ids" "${flist}" | sort -k1 > "data/${out_name}/wav.scp"
    sed -E "s#${odir}/audio/16kHz/isolated/(.*)\.wav#${odir}/audio/16kHz/isolated_ext/\1.Clean.wav#g" \
        "data/${out_name}/wav.scp" > "data/${out_name}/spk1.scp"
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" "data/${out_name}/spk1.scp" > "data/${out_name}/noise1.scp"

    awk '
        NR==FNR {
            key=$1
            sub(/\.CH[1-6]_SIMU$/, "_SIMU", key)
            $1=""
            sub(/^ /, "")
            txt[key]=$0
            next
        }
        {
            key=$1
            orig=$1
            sub(/\.CH[1-6]_SIMU$/, "_SIMU", key)
            print orig, txt[key]
        }
    ' "${text_src}" "${flist}.ids" > "data/${out_name}/text"

    awk '{print $1, $1}' "data/${out_name}/wav.scp" | awk -F'_' '{print $1, $2}' >/dev/null 2>&1
    awk -F'_' '{print $1}' "data/${out_name}/wav.scp" > "${flist}.spk"
    awk '{print $1}' "data/${out_name}/wav.scp" > "${flist}.utt"
    paste -d" " "${flist}.utt" "${flist}.spk" > "data/${out_name}/utt2spk"
    utils/utt2spk_to_spk2utt.pl "data/${out_name}/utt2spk" > "data/${out_name}/spk2utt"
    utils/fix_data_dir.sh "data/${out_name}" >/dev/null
    rm -f "${flist}" "${flist}.ids" "${flist}.spk" "${flist}.utt"
}

create_allch_single_snr_set() {
    local snr="$1"
    local out_name="et05_simu_snr${snr}_allch_track"
    local flist text_src
    flist="$(mktemp)"
    text_src="data/et05_simu_isolated_1ch_track/text"

    find "${odir}/audio/16kHz/isolated" -name '*.wav' \
        | grep "et05_bus_simu_snr${snr}\|et05_caf_simu_snr${snr}\|et05_ped_simu_snr${snr}\|et05_str_simu_snr${snr}" \
        | sort -u > "${flist}"

    [ -s "${flist}" ] || { rm -f "${flist}"; return 0; }

    mkdir -p "data/${out_name}"
    awk -F'/' '{print $NF}' "${flist}" | sed 's/\.wav/_SIMU/' > "${flist}.ids"
    paste -d" " "${flist}.ids" "${flist}" | sort -k1 > "data/${out_name}/wav.scp"
    sed -E "s#${odir}/audio/16kHz/isolated/(.*)\.wav#${odir}/audio/16kHz/isolated_ext/\1.Clean.wav#g" \
        "data/${out_name}/wav.scp" > "data/${out_name}/spk1.scp"
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" "data/${out_name}/spk1.scp" > "data/${out_name}/noise1.scp"

    awk '
        NR==FNR {
            key=$1
            sub(/\.CH[1-6]_SIMU$/, "_SIMU", key)
            $1=""
            sub(/^ /, "")
            txt[key]=$0
            next
        }
        {
            key=$1
            orig=$1
            sub(/\.CH[1-6]_SIMU$/, "_SIMU", key)
            print orig, txt[key]
        }
    ' "${text_src}" "${flist}.ids" > "data/${out_name}/text"

    awk -F'_' '{print $1}' "data/${out_name}/wav.scp" > "${flist}.spk"
    awk '{print $1}' "data/${out_name}/wav.scp" > "${flist}.utt"
    paste -d" " "${flist}.utt" "${flist}.spk" > "data/${out_name}/utt2spk"
    utils/utt2spk_to_spk2utt.pl "data/${out_name}/utt2spk" > "data/${out_name}/spk2utt"
    utils/fix_data_dir.sh "data/${out_name}" >/dev/null
    rm -f "${flist}" "${flist}.ids" "${flist}.spk" "${flist}.utt"
}

help_message=$(cat << EOF
Usage: $0 --extra-annotations <path> [--stage <stage>] [--stop_stage <stop_stage>] [--nj <nj>] [--simulate-valid-only true]

  required argument:
    --extra-annotations: path to a directory containing extra annotations for CHiME4
                         This is required for preparing et05_simu_isolated_1ch_track.
    NOTE:
        You can download it manually from
            http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME4/download.html
        Then unzip the downloaded file to CHiME4_diff;
        You will then find the extra annotations in CHiME4_diff/CHiME3/data/annotations

  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default). Use underscore, not hyphen.
    [--nj]: number of parallel pool workers in MATLAB
    [--simulate-valid-only true]: run stage 1 only for dt05 (valid) to regenerate with random SNR; faster than full simulation. Requires existing tr05/et05 data. Must pass \"true\" as value.
EOF
)


nj=32
stage=1
stop_stage=2
extra_annotations=
simulate_valid_only=false
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ] || [ -z "${extra_annotations}" ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${CHIME3}" ]; then
    log "Fill the value of 'CHIME3' in db.sh"
    exit 1
fi

if [ ! -e "${CHIME4}" ]; then
    log "Fill the value of 'CHIME4' in db.sh"
    exit 1
fi


odir="${PWD}/local/nn-gev/data"; mkdir -p "${odir}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Simulation"

    if ! command -v matlab &> /dev/null; then
        log "You don't have matlab"
        exit 2
    fi

    # Prepare simulation data for 6ch track
    # (This takes ~10 hours with nj=10 on Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz)
    # Expected data directories to be generated (~40 GB):
    #   - ${odir}/audio/16kHz/isolated_ext/*/*.CH?.{Clean,Noise}.wav
    #   - ${odir}/audio/16kHz/isolated/*/*.CH?.wav
    # -----------------------------------------------------------------------------------------------
    # directory                   disk usage  duration      #samples
    # -----------------------------------------------------------------------------------------------
    # isolated_ext/tr05_bus_simu  4.9 GB      44h 29m 45s   1728 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/tr05_caf_simu  5.0 GB      45h 17m 23s   1794 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/tr05_ped_simu  4.9 GB      44h 58m 57s   1765 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/tr05_str_simu  5.1 GB      46h 59m 49s   1851 * 6 * 2 (6 channels, Clean & Noise)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated_ext/dt05_bus_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/dt05_caf_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/dt05_ped_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/dt05_str_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated_ext/et05_bus_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/et05_caf_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/et05_ped_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/et05_str_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # -----------------------------------------------------------------------------------------------
    # isolated/tr05_bus_simu      2.5 GB      22h 14m 52s   1728 * 6 (6 channels, Noisy)
    # isolated/tr05_caf_simu      2.5 GB      22h 38m 41s   1794 * 6 (6 channels, Noisy)
    # isolated/tr05_ped_simu      2.5 GB      22h 29m 28s   1765 * 6 (6 channels, Noisy)
    # isolated/tr05_str_simu      2.6 GB      23h 29m 54s   1851 * 6 (6 channels, Noisy)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated/dt05_bus_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # isolated/dt05_caf_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # isolated/dt05_ped_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # isolated/dt05_str_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated/et05_bus_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # isolated/et05_caf_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # isolated/et05_ped_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # isolated/et05_str_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # -----------------------------------------------------------------------------------------------

    log "Generating simulation data and storing in ${odir}"
    if "${simulate_valid_only}"; then
        log "Simulating valid (dt05) only; tr05/et05 left unchanged."
        ${train_cmd} $odir/simulation_dt05_only.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$nj,'${CHIME4}','${CHIME3}','${odir}','dt05_only');exit"
        num_dt05=$(find "$odir/audio/16kHz/isolated" -path "*dt05*simu*" -iname "*.wav" | wc -l)
        if [ "$num_dt05" -lt 1600 ]; then
            log "Error: Expected at least 1600 dt05 wav files, got $num_dt05"
            exit 1
        fi
        log "dt05 simulation done. Run stage 2 to refresh data/dt05_simu_isolated_1ch_track."
    else
        ${train_cmd} $odir/simulation.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$nj,'${CHIME4}','${CHIME3}','${odir}');exit"

        # Validate data simulation (1ch only: out_channels=3; train/valid/test as above)
        # isolated: 100188/6 = 16698 (one channel), isolated_ext: 200376/6 = 33396
        num_wavs=$(find "$odir/audio/16kHz/isolated" -iname "*.wav" | wc -l)
        if [ "$num_wavs" != "16698" ]; then
            log "Error: Expected 16698 wav files in '$odir/audio/16kHz/isolated' (1ch), but got $num_wavs"
            exit 1
        fi
        num_wavs=$(find "$odir/audio/16kHz/isolated_ext" -iname "*.wav" | wc -l)
        if [ "$num_wavs" != "33396" ]; then
            log "Error: Expected 33396 wav files in '$odir/audio/16kHz/isolated_ext' (1ch), but got $num_wavs"
            exit 1
        fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

    # preparation for original WSJ0 data:
    #  et05_orig_clean, dt05_orig_clean, tr05_orig_clean
    wsj0_data=${CHIME4}/data/WSJ0
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh

    # preparation for chime4 data:
    #  (1) tr05_real_noisy, dt05_real_noisy, et05_real_noisy
    local/real_noisy_chime4_data_prep.sh ${CHIME4}
    #  (2) tr05_simu_noisy, dt05_simu_noisy, et05_simu_noisy
    local/simu_noisy_chime4_data_prep.sh ${CHIME4}

    # prepare data for 1ch track only (run.sh uses 1ch only; 2ch/6ch prep skipped)
    #  (1) {tr05,dt05,et05}_simu_isolated_1ch_track [+ et05_simu_snr* per-SNR test sets]
    local/simu_ext_chime4_data_prep.sh --track 1 --annotations ${CHIME4}/data/annotations \
        --extra-annotations ${extra_annotations} isolated_1ch_track ${odir}/audio/16kHz
    #  (2) {tr05,dt05,et05}_real_isolated_1ch_track
    local/real_ext_chime4_data_prep.sh --track 1 --isolated_6ch_dir ${CHIME4}/data/audio/16kHz/isolated_6ch_track \
        isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track

    #  (3) Ensure no utt2category in 1ch_track dirs (enh preprocessor would need categories config; we do not use it).
    for d in tr05_simu_isolated_1ch_track dt05_simu_isolated_1ch_track et05_simu_isolated_1ch_track \
             tr05_real_isolated_1ch_track dt05_real_isolated_1ch_track et05_real_isolated_1ch_track; do
        rm -f "data/${d}/utt2category"
    done
    #  (4) real+simu combined (multi) for SE training/eval.
    utils/combine_data.sh --extra_files "spk1.scp" \
        data/tr05_multi_isolated_1ch_track data/tr05_simu_isolated_1ch_track data/tr05_real_isolated_1ch_track
    utils/combine_data.sh --extra_files "spk1.scp" \
        data/dt05_multi_isolated_1ch_track data/dt05_simu_isolated_1ch_track data/dt05_real_isolated_1ch_track
    utils/combine_data.sh --extra_files "spk1.scp" \
        data/et05_multi_isolated_1ch_track data/et05_simu_isolated_1ch_track data/et05_real_isolated_1ch_track
    for d in tr05_multi_isolated_1ch_track dt05_multi_isolated_1ch_track et05_multi_isolated_1ch_track; do
        rm -f "data/${d}/utt2category"
    done
    log "Created real+simu combined sets: tr05_multi_isolated_1ch_track, dt05_multi_isolated_1ch_track, et05_multi_isolated_1ch_track"

    # Additional single-channel-expanded sets using CH1~CH6 as independent
    # training/evaluation examples. This is still a 1ch recipe, not a true
    # multi-channel model.
    create_allch_single_set tr05 tr05_simu_allch_track
    create_allch_single_set dt05 dt05_simu_allch_track
    create_allch_single_set et05 et05_simu_allch_track
    for snr in -10 -5 0 5 10; do
        create_allch_single_snr_set "${snr}"
    done
    log "Created expanded 1ch sets: tr05/dt05/et05_simu_allch_track (+ et05_simu_snr*_allch_track when available)"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
