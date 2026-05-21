#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
recipe_dir="$(cd "${script_dir}/.." && pwd)"
log_file="${recipe_dir}/exp/dual_transducer_watch/launcher.log"

mkdir -p "$(dirname "${log_file}")"

cd "${recipe_dir}"

while pgrep -f "espnet2.bin.asr_transducer_inference.*asr_train_asr_transducer_conformer_enhsmall_en_bpe500_sp" >/dev/null; do
    echo "[$(date +%F_%T)] waiting for decode to finish" >> "${log_file}"
    sleep 60
done

echo "[$(date +%F_%T)] decode finished, starting 4GPU dual-transducer training" >> "${log_file}"

CHANNEL_MODE=ch1 \
NGPU=4 \
TOKEN_TYPE=bpe \
ASR_TAG=train_asr_dual_transducer_conformer_raw_en_bpe500 \
./run_dual_transducer.sh --stage 3 --stop_stage 13 --nbpe 500 \
    >> "${log_file}" 2>&1
