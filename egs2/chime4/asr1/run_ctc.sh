#!/usr/bin/env bash
# CHiME4 asr1: use the same data pipeline as enh1 (tr05_simu_isolated_1ch_track, same wav.scp, same format).
# Prerequisite: run from asr1: local/data_enh1_style.sh --extra-annotations <path>
#   (that runs enh1/local/data.sh and symlinks asr1/data -> enh1/data).
#
# Important:
#   CHiME4-only training here is much smaller than the standard WSJ+CHiME setup.
#   In practice, BPE1024 on this small corpus easily collapses into frequent pieces
#   such as "THE", "S", ".PERIOD".  Use char by default for stable CTC training.

set -e
set -u
set -o pipefail

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track
test_sets="et05_multi_isolated_1ch_track et05_simu_isolated_1ch_track"
for d in et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track; do
    [ -d "data/${d}" ] && test_sets="${test_sets} ${d}"
done

asr_config=conf/tuning/train_conformer_ctc.yaml
inference_config=conf/tuning/decode_ctc_bs1.yaml
inference_asr_model=valid.cer_ctc.ave_10best.pth
token_type="${token_type:-char}"
nbpe="${nbpe:-1024}"
bpemode="${bpemode:-unigram}"

# Use enh1 data pipeline; ASR needs single channel so we normalize paths to .CH1.wav (some disks have only CH1).
# BASH_SOURCE works when script is sourced ( . run_ctc.sh ); dirname -- avoids "-bash" as option.
_script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
enh1_dir="$(cd "${_script_dir}/../enh1" && pwd)"
enh1_data="${enh1_dir}/data"
if [ ! -f "data/${train_set}/text" ]; then
    if [ -f "${enh1_data}/${train_set}/text" ]; then
        echo "Copying enh1 data into asr1/data and fixing wav.scp (absolute paths + CH1 only)..."
        mkdir -p data
        for d in tr05_simu_isolated_1ch_track dt05_simu_isolated_1ch_track et05_simu_isolated_1ch_track \
                 et05_multi_isolated_1ch_track \
                 et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
                 et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track; do
            if [ -d "${enh1_data}/${d}" ]; then
                rm -rf "data/${d}"
                cp -r "${enh1_data}/${d}" "data/${d}"
                if [ -f "data/${d}/wav.scp" ]; then
                    # 1) paths absolute for asr1 cwd; 2) ASR single channel: .CH2–.CH6 -> .CH1 so we only open CH1
                    awk -v pre="${enh1_dir}/" 'NF>=2 && $2 !~ /^\// { $2 = pre $2 } { print }' "data/${d}/wav.scp" | \
                    sed -E 's|\.CH[2-6]\.wav|.CH1.wav|g' > "data/${d}/wav.scp.tmp"
                    mv "data/${d}/wav.scp.tmp" "data/${d}/wav.scp"
                fi
            fi
        done
    else
        echo "Error: data/${train_set}/text not found. Run first: local/data_enh1_style.sh --extra-annotations <path>"
        exit 1
    fi
fi

# nlsyms from train text
nlsyms=data/nlsyms.txt
if [ -f "data/${train_set}/text" ]; then
    cut -f 2- "data/${train_set}/text" | tr " " "\n" | sort -u | grep "^<" > "${nlsyms}" || true
    [ -s "${nlsyms}" ] || echo "<NOISE>" > "${nlsyms}"
else
    echo "Error: data/${train_set}/text not found. Run: local/data_enh1_style.sh --extra-annotations <path>"
    exit 1
fi

# Speed perturbation 0.9/1.0/1.1 -> ~3x train data (same 7k utts, 3 speeds).
speed_perturb_factors="0.9 1.0 1.1"

extra_args=()
if [ "${token_type}" = "bpe" ]; then
    extra_args+=(
        --token_type bpe
        --nbpe "${nbpe}"
        --bpemode "${bpemode}"
        --bpe_train_text "data/${train_set}/text"
    )
else
    extra_args+=(
        --token_type char
    )
fi

# Skip missing/corrupt wavs (some CH1 files may still be missing) then fix_data_dir.
./asr.sh \
    --lang en \
    --format_wav_scp_skip_bad_files true \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 8 \
    --nlsyms_txt "${nlsyms}" \
    --bpe_nlsyms "${nlsyms}" \
    --max_wav_duration 30 \
    --audio_format "flac" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    "${extra_args[@]}" \
    "$@"
