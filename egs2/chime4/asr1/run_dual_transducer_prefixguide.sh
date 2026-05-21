#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

ASR_CONFIG="${ASR_CONFIG:-conf/tuning/train_asr_dual_transducer_enhanced_prefixguide.yaml}" \
ASR_TAG="${ASR_TAG:-train_asr_dual_transducer_enhanced_prefixguide_en_bpe500}" \
./run_dual_transducer_enhanced.sh "$@"
