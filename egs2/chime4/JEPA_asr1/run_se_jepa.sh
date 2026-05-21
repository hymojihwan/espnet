#!/usr/bin/env bash
# CHiME4 SE-JEPA ASR: Conv-TasNet (SI-SNR) + JEPA predictive + Conformer CTC
# Flow: Noisy wav → SE → enhanced wav → log-mel → JEPA → Conformer → CTC
# Data: use enh1 data (tr05/dt05/et05_simu_isolated_1ch_track). Run enh1/local/data.sh first, then JEPA_asr1/local/data.sh.

set -e
set -u
set -o pipefail

# train/valid = multi; test = multi + per-SNR simu [-10,-5,0,5,10] dB
train_set=tr05_multi_isolated_1ch_track
valid_set=dt05_multi_isolated_1ch_track
test_sets="et05_multi_isolated_1ch_track et05_simu_snr-10_isolated_1ch_track et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track"

asr_task=asr_se_jepa
asr_config=conf/tuning/train_se_jepa_scratch.yaml
inference_config=conf/tuning/decode_jepa_ctc.yaml
inference_asr_model=valid.wer_ctc.ave_10best.pth

./asr.sh \
    --asr_task "${asr_task}" \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 8 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --asr_args "--unused_parameters true --use_amp true" \
    "$@"
