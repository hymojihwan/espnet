#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

export ASR_TASK="${ASR_TASK:-asr_jepa}"
export ASR_CONFIG="${ASR_CONFIG:-../se_mel_asr1/conf/tuning/train_conformer_ctc_jepa_masked_feature.yaml}"
export ASR_TAG="${ASR_TAG:-train_conformer_ctc_mel80cleanmel_char_sp_initraw_jepa}"
export TOKEN_TYPE="${TOKEN_TYPE:-char}"
export USE_LM="${USE_LM:-false}"
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-conf/tuning/decode_ctc_bs1.yaml}"
export INFERENCE_ASR_MODEL="${INFERENCE_ASR_MODEL:-valid.cer_ctc.ave_10best.pth}"

exec ./run.sh "$@"
