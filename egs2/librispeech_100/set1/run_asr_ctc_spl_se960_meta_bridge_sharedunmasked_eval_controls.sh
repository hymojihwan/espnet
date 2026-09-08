#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

source_tag=asr_ctc_spl_se960_meta_bridge_sharedunmasked_exactmaml_outerasr_adapter16_s1_k1_ilr01_main
source_exp="exp/asr_${source_tag}"
checkpoint=valid.cer_ctc.ave_10best.pth

test -f "${source_exp}/config.yaml"
test -e "${source_exp}/${checkpoint}"

prepare_control() {
    local condition=$1
    local control_exp=$2

    mkdir -p "${control_exp}"
    cp "${source_exp}/config.yaml" "${control_exp}/config.yaml"
    sed -i \
        's/^\([[:space:]]*meta_adapt_at_inference:\) true$/\1 false/' \
        "${control_exp}/config.yaml"

    if [ "${condition}" = base ]; then
        sed -i \
            's/^\([[:space:]]*meta_adapter_scale:\) 1\.0$/\1 0.0/' \
            "${control_exp}/config.yaml"
        grep -q '^[[:space:]]*meta_adapter_scale: 0.0$' \
            "${control_exp}/config.yaml"
    fi

    grep -q '^[[:space:]]*meta_adapt_at_inference: false$' \
        "${control_exp}/config.yaml"
    ln -sfn "$(readlink -f "${source_exp}/${checkpoint}")" \
        "${control_exp}/${checkpoint}"
}

k0_exp=exp/asr_${source_tag}_k0_eval
prepare_control k0 "${k0_exp}"
./run_asr_ctc_spl_se960_meta_bridge_sharedunmasked_exactmaml_outerasr_adapter16_s1_k1_ilr01.sh \
    --stage 12 \
    --stop_stage 13 \
    --asr_exp "${k0_exp}" \
    --inference_tag decode_asr_ctc_only_sharedunmasked_k0 \
    "$@"

base_exp=exp/asr_${source_tag}_base_eval
prepare_control base "${base_exp}"
./run_asr_ctc_spl_se960_meta_bridge_sharedunmasked_exactmaml_outerasr_adapter16_s1_k1_ilr01.sh \
    --stage 12 \
    --stop_stage 13 \
    --asr_exp "${base_exp}" \
    --inference_tag decode_asr_ctc_only_sharedunmasked_base \
    "$@"
