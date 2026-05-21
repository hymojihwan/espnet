#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

channel_mode="${CHANNEL_MODE:-ch1}"
stage="${STAGE:-7}"
stop_stage="${STOP_STAGE:-7}"
ngpu="${NGPU:-4}"
inference_nj="${INFERENCE_NJ:-32}"
inference_model="${INFERENCE_MODEL:-valid.loss.best.pth}"
inference_tag="${INFERENCE_TAG:-enhanced_for_asr}"

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

if [ "${channel_mode}" = allch ]; then
    train_set=tr05_simu_allch_track
    valid_set=dt05_simu_allch_track
    test_sets="\
dt05_simu_allch_track \
et05_simu_allch_track \
et05_simu_snr-5_allch_track et05_simu_snr0_allch_track \
et05_simu_snr5_allch_track et05_simu_snr10_allch_track et05_simu_snr15_allch_track"
else
    train_set=tr05_simu_isolated_1ch_track
    valid_set="${train_set}"
    test_sets="\
dt05_simu_isolated_1ch_track \
et05_multi_isolated_1ch_track \
et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track \
et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track"
fi

exec ./enh.sh \
    --skip_train true \
    --skip_data_prep true \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu "${ngpu}" \
    --inference_nj "${inference_nj}" \
    --gpu_inference true \
    --ref_num 1 \
    --inf_num 1 \
    --fs 16k \
    --audio_format wav \
    --enh_config conf/tuning/train_enh_convtasnet_small.yaml \
    --inference_model "${inference_model}" \
    --inference_tag "${inference_tag}" \
    --format_wav_scp_skip_bad_files true \
    "$@"
