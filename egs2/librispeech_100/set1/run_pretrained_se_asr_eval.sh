#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Skip observation-ASR scoring and evaluate enhanced-only path.
printf 'skip observation ASR scoring for enhanced-only evaluation\n' > dump/raw/RESULTS_ASR.md

# Prepare wav_spk1 expected by enh.sh stage 9 from pre-generated SE outputs.
for d in exp/enh_train_enh_convtasnet_small_librispeech_raw/enhanced_*; do
    [ -d "$d" ] || continue
    mkdir -p "$d/scoring"
    cp -f "$d/spk1.scp" "$d/scoring/wav_spk1"
done

# 1) Pretrained SE + pretrained ASR(no-spec)
./run_se.sh \
    --stage 9 --stop_stage 10 \
    --valid_set "" \
    --enh_config conf/tuning/train_enh_convtasnet_small_librispeech.yaml \
    --enh_tag train_enh_convtasnet_small_librispeech_raw \
    --score_with_asr true \
    --asr_exp exp/asr_asr_conformer-rnnt-nonk2-nosp-nospec_main \
    --inference_asr_model valid.loss.ave_10best.pth \
    --inference_asr_config conf/tuning/transducer/decode_transducer.yaml \
    --inference_asr_tag rnnt_nospec_pretrained_enh \
    2>&1 | tee exp/enh_train_enh_convtasnet_small_librispeech_raw/run_pretrained_nospec_stage910.log

# 2) Pretrained SE + pretrained ASR(spec)
./run_se.sh \
    --stage 9 --stop_stage 10 \
    --valid_set "" \
    --enh_config conf/tuning/train_enh_convtasnet_small_librispeech.yaml \
    --enh_tag train_enh_convtasnet_small_librispeech_raw \
    --score_with_asr true \
    --asr_exp exp/asr_asr_conformer-rnnt-nonk2-nosp-spec_main_fixspec_on \
    --inference_asr_model valid.loss.ave_10best.pth \
    --inference_asr_config conf/tuning/transducer/decode_transducer.yaml \
    --inference_asr_tag rnnt_spec_pretrained_enh \
    2>&1 | tee exp/enh_train_enh_convtasnet_small_librispeech_raw/run_pretrained_spec_stage910.log
