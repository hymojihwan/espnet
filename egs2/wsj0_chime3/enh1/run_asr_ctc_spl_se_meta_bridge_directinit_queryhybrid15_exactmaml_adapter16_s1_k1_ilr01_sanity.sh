#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export ESPNET_CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/user/Workspace/espnet_bridge_meta_tta:${PYTHONPATH:-}"

train_set="train"
valid_set="valid"
test_sets="test_m5 test_0 test_p5 test_p10 test_p15"

../../TEMPLATE/asr1/asr.sh \
    --expdir exp/-5to15 \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --gpu_inference false \
    --inference_nj 4 \
    --token_type bpe \
    --nbpe 2048 \
    --bpemode unigram \
    --feats_type raw \
    --feats_normalize utterance_mvn \
    --audio_format flac \
    --use_lm false \
    --use_clean_speech false \
    --nlsyms_txt data/nlsyms.txt \
    --asr_task asr_jepa \
    --asr_config conf/new_SPL/CTC/train_asr_ctc_spl_se_meta_bridge_directinit_queryhybrid15_exactmaml_adapter16_s1_k1_ilr01_sanity.yaml \
    --inference_config conf/tuning/decode_ctc_bs1.yaml \
    --inference_asr_model valid.cer_ctc.ave_10best.pth \
    --asr_tag train_asr_ctc_spl_se_meta_bridge_directinit_queryhybrid15_exactmaml_adapter16_s1_k1_ilr01_wsj0_chime3_sanity \
    --speed_perturb_factors "" \
    --stage 11 \
    --stop_stage 11 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"
