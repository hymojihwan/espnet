#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k
channel_mode="${CHANNEL_MODE:-ch1}"
enh_config="${ENH_CONFIG:-conf/tuning/train_enh_conv_tasnet.yaml}"
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.
extra_annotations=/DB/CHiME4/data/annotations
sim_cache_dir="${PWD}/local/nn-gev/data/audio/16kHz/isolated"

if [ "${channel_mode}" = allch ]; then
    train_set=tr05_simu_allch_track
    valid_set=dt05_simu_allch_track
    test_sets="et05_simu_allch_track et05_simu_snr-5_allch_track et05_simu_snr0_allch_track et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    # 1ch: Simu-only train/valid, aligned with ASR's simulated noisy regime.
    # These dirs are the enhancement-paired version of the same simulated mixtures:
    # wav.scp = noisy speech, spk1.scp = clean reference.
    train_set=tr05_simu_isolated_1ch_track
    valid_set=dt05_simu_isolated_1ch_track
    # et: multi + per-SNR simu. If per-SNR et05 not generated, use only et05_multi_isolated_1ch_track.
    test_sets="et05_multi_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

# Multi (simu+real) train: uncomment to add real; train loss will be lower on average (real has no -10 dB type samples).
# train_set=tr05_multi_isolated_1ch_track
# valid_set=dt05_simu_isolated_1ch_track  # keep valid simu-only (real ref is not clean)

if [ -d "${sim_cache_dir}" ]; then
    local_data_opts="--extra-annotations ${extra_annotations} --stage 2 --stop_stage 2"
else
    local_data_opts="--extra-annotations ${extra_annotations} --stage 1 --stop_stage 2"
fi

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 1 \
    --ref_channel 3 \
    --format_wav_scp_skip_bad_files true \
    --local_data_opts "${local_data_opts}" \
    --enh_config "${enh_config}" \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
