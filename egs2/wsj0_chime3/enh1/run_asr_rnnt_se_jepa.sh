#!/usr/bin/env bash
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test_m5 test_0 test_p5 test_p10 test_p15"

token_type="${TOKEN_TYPE:-bpe}"
nbpe="${NBPE:-2048}"
bpemode="${BPEMODE:-unigram}"
asr_task="${ASR_TASK:-asr_transducer}"
asr_config="${ASR_CONFIG:-conf/tuning/SPL/train_asr_rnnt_se_jepa_small.yaml}"
inference_config="${INFERENCE_CONFIG:-conf/tuning/SPL/decode_transducer_rnnt_frozen_se.yaml}"
inference_asr_model="${INFERENCE_ASR_MODEL:-valid.loss.ave_10best.pth}"
asr_tag="${ASR_TAG:-train_asr_rnnt_se_jepa_small_raw_bpe2048}"
expdir="${EXPDIR:-exp/-5to15}"

./../../TEMPLATE/asr1/asr.sh \
    --expdir "${expdir}" \
    --lang en \
    --ngpu 4 \
    --token_type "${token_type}" \
    --nbpe "${nbpe}" \
    --bpemode "${bpemode}" \
    --feats_type raw \
    --audio_format flac \
    --use_lm false \
    --nlsyms_txt data/nlsyms.txt \
    --asr_task "${asr_task}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --asr_tag "${asr_tag}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"
