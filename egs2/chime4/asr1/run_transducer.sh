#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

channel_mode="${CHANNEL_MODE:-ch1}"
extra_annotations="${EXTRA_ANNOTATIONS:-/DB/CHiME4/data/annotations}"
simulate_stage="${SIMULATE_STAGE:-1}"
simulate_stop_stage="${SIMULATE_STOP_STAGE:-2}"
stage="${STAGE:-1}"
stop_stage="${STOP_STAGE:-13}"

. utils/parse_options.sh

if [ "${channel_mode}" = allch ] && [ -d data/tr05_simu_allch_track ]; then
    train_set=tr05_simu_allch_track
elif [ -d data/tr05_simu_noisy ]; then
    train_set=tr05_simu_noisy
elif [ -d data/tr05_simu_isolated_1ch_track ]; then
    train_set=tr05_simu_isolated_1ch_track
elif [ -d data/tr05_multi_noisy_si284 ]; then
    train_set=tr05_multi_noisy_si284
else
    train_set=tr05_simu_noisy
fi

if [ "${channel_mode}" = allch ]; then
    valid_set=dt05_simu_allch_track
    test_sets="et05_simu_allch_track "
    for dset in \
        et05_simu_snr-5_allch_track \
        et05_simu_snr0_allch_track \
        et05_simu_snr5_allch_track \
        et05_simu_snr10_allch_track \
        et05_simu_snr15_allch_track; do
        if [ -d "data/${dset}" ]; then
            test_sets="${test_sets}${dset} "
        fi
    done
else
    if [ -d data/dt05_simu_noisy ]; then
        valid_set=dt05_simu_noisy
    else
        valid_set=dt05_simu_isolated_1ch_track
    fi
    test_sets="dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track "
    for dset in \
        et05_simu_snr-5_isolated_1ch_track \
        et05_simu_snr0_isolated_1ch_track \
        et05_simu_snr5_isolated_1ch_track \
        et05_simu_snr10_isolated_1ch_track \
        et05_simu_snr15_isolated_1ch_track; do
        if [ -d "data/${dset}" ]; then
            test_sets="${test_sets}${dset} "
        fi
    done
fi

if [ -d data/dt05_real_beamformit_2mics ] && [ -d data/dt05_real_beamformit_5mics ]; then
    test_sets="${test_sets}dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics "
fi

asr_task=asr_transducer
token_type=bpe
nbpe="${NBPE:-500}"
bpemode="${BPEMODE:-unigram}"
speed_perturb_factors="0.9 1.0 1.1"
asr_config="${ASR_CONFIG:-conf/tuning/train_asr_transducer_conformer.yaml}"
inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_transducer.yaml}"
inference_asr_model="${INFERENCE_ASR_MODEL:-valid.loss.ave_10best.pth}"
asr_tag="${ASR_TAG:-train_asr_transducer_conformer_raw_en_bpe${nbpe}_sp}"

ensure_data() {
    local enh_train_src enh_valid_src
    if [ "${channel_mode}" = allch ]; then
        enh_train_src="../enh1/data/tr05_simu_allch_track"
        enh_valid_src="../enh1/data/dt05_simu_allch_track"
    else
        enh_train_src="../enh1/data/tr05_simu_isolated_1ch_track"
        enh_valid_src="../enh1/data/dt05_simu_isolated_1ch_track"
    fi

    if [ ! -f "${enh_train_src}/wav.scp" ] || [ ! -f "${enh_valid_src}/wav.scp" ]; then
        echo "[run_transducer.sh] prepare enh1 simulated data first (${extra_annotations})"
        (
            cd ../enh1
            ./local/data.sh \
                --extra-annotations "${extra_annotations}" \
                --stage "${simulate_stage}" \
                --stop_stage "${simulate_stop_stage}"
        )
    fi

    if [ ! -f "data/${train_set}/wav.scp" ] || [ ! -f "data/${valid_set}/wav.scp" ]; then
        echo "[run_transducer.sh] prepare asr1 data"
        ./local/data.sh --stage 0 --stop_stage 2
    fi
}

ensure_data

if [ "${stage}" -ge 10 ]; then
    dump_train_set="${train_set}"
    dump_valid_set="${valid_set}"
    if [ -n "${speed_perturb_factors}" ]; then
        dump_train_set="${dump_train_set}_sp"
    fi
    if [ ! -f "dump/raw/${dump_train_set}/feats_type" ]; then
        echo "[run_transducer.sh] Missing dump/raw/${dump_train_set}/feats_type"
        echo "[run_transducer.sh] Run from stage 3 first, or use a train_set with an existing dump."
        exit 1
    fi
    if [ ! -f "dump/raw/${dump_valid_set}/feats_type" ]; then
        echo "[run_transducer.sh] Missing dump/raw/${dump_valid_set}/feats_type"
        echo "[run_transducer.sh] Run from stage 3 first, or use a valid_set with an existing dump."
        exit 1
    fi
fi

./asr.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 8 \
    --format_wav_scp_skip_bad_files true \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --token_type "${token_type}" \
    --nbpe "${nbpe}" \
    --bpemode "${bpemode}" \
    --max_wav_duration 30 \
    --audio_format flac.ark \
    --feats_type raw \
    --use_lm false \
    --asr_task "${asr_task}" \
    --asr_config "${asr_config}" \
    --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
