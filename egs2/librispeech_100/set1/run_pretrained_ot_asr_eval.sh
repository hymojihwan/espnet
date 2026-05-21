#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

enh_tag="enh_train_enh_se_ot_frontend_raw"

# Skip observation-ASR scoring and evaluate enhanced-only path.
printf 'skip observation ASR scoring for enhanced-only evaluation\n' > dump/raw/RESULTS_ASR.md

# Prepare wav_spk1 expected by enh.sh stage 9 from pre-generated SE outputs.
for d in exp/${enh_tag}/enhanced_*; do
    [ -d "$d" ] || continue
    mkdir -p "$d/scoring"
    cp -f "$d/spk1.scp" "$d/scoring/wav_spk1"
done

# 1) Pretrained OT-SE + pretrained ASR(no-spec)
./run_se_ot_frontend.sh \
    --stage 9 --stop_stage 10 \
    --valid_set "" \
    --enh_tag "${enh_tag#enh_}" \
    --score_with_asr true \
    --asr_exp exp/asr_asr_conformer-rnnt-nonk2-nosp-nospec_main \
    --inference_asr_model valid.loss.ave_10best.pth \
    --inference_asr_config conf/tuning/transducer/decode_transducer.yaml \
    --inference_asr_tag rnnt_nospec_pretrained_otenh \
    2>&1 | tee exp/${enh_tag}/run_pretrained_ot_nospec_stage910.log

# 2) Pretrained OT-SE + pretrained ASR(spec)
./run_se_ot_frontend.sh \
    --stage 9 --stop_stage 10 \
    --valid_set "" \
    --enh_tag "${enh_tag#enh_}" \
    --score_with_asr true \
    --asr_exp exp/asr_asr_conformer-rnnt-nonk2-nosp-spec_main_fixspec_on \
    --inference_asr_model valid.loss.ave_10best.pth \
    --inference_asr_config conf/tuning/transducer/decode_transducer.yaml \
    --inference_asr_tag rnnt_spec_pretrained_otenh \
    2>&1 | tee exp/${enh_tag}/run_pretrained_ot_spec_stage910.log
