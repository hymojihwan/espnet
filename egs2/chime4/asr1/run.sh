#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

channel_mode="${CHANNEL_MODE:-ch1}"

# Default to simu-only train/valid so the noisy distribution matches the
# controlled random-SNR simulation used in enhancement (-5~15 dB after the
# patched CHiME4 simulation step).
if [ "${channel_mode}" = allch ] && [ -d data/tr05_simu_allch_track ]; then
    train_set=tr05_simu_allch_track
elif [ -d data/tr05_simu_isolated_1ch_track ]; then
    train_set=tr05_simu_isolated_1ch_track
elif [ -d data/tr05_multi_noisy_si284 ]; then
    train_set=tr05_multi_noisy_si284
else
    train_set=tr05_simu_noisy
fi
if [ "${channel_mode}" = allch ]; then
    valid_set=dt05_simu_allch_track
    test_sets="et05_simu_allch_track "
    for dset in \
        et05_simu_snr-5_allch_track \
        et05_simu_snr0_allch_track \
        et05_simu_snr5_allch_track \
        et05_simu_snr10_allch_track \
        et05_simu_snr15_allch_track; do
        if [ -d "data/${dset}" ]; then
            test_sets="${test_sets}${dset} "
        fi
    done
else
    valid_set=dt05_simu_isolated_1ch_track
    test_sets="\
dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
"
    for dset in \
        et05_simu_snr-5_isolated_1ch_track \
        et05_simu_snr0_isolated_1ch_track \
        et05_simu_snr5_isolated_1ch_track \
        et05_simu_snr10_isolated_1ch_track \
        et05_simu_snr15_isolated_1ch_track; do
        if [ -d "data/${dset}" ]; then
            test_sets="${test_sets}${dset} "
        fi
    done
fi
if [ -d data/dt05_real_beamformit_2mics ] && [ -d data/dt05_real_beamformit_5mics ]; then
    test_sets="${test_sets} \
dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
"
fi

token_type="${TOKEN_TYPE:-bpe}"
nbpe="${NBPE:-500}"
bpemode="${BPEMODE:-unigram}"
use_lm="${USE_LM:-true}"
lm_config="${LM_CONFIG:-conf/train_lm_transformer.yaml}"
if [ "${token_type}" = bpe ]; then
    asr_config="${ASR_CONFIG:-conf/tuning/train_conformer_hybrid.yaml}"
    inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_asr_bs10_lm03.yaml}"
    inference_asr_model=valid.acc.ave_10best.pth
else
    asr_config="${ASR_CONFIG:-conf/tuning/train_conformer_ctc.yaml}"
    inference_config="${INFERENCE_CONFIG:-conf/tuning/decode_ctc_bs1_lm03.yaml}"
    inference_asr_model=valid.cer_ctc.ave_10best.pth
fi

speed_perturb_factors="0.9 1.0 1.1"
if [ "${token_type}" = bpe ]; then
    token_tag="bpe${nbpe}"
else
    token_tag="${token_type}"
fi
if [ -n "${speed_perturb_factors}" ]; then
    asr_tag="train_conformer_ctc_raw_en_${token_tag}_sp"
else
    asr_tag="train_conformer_ctc_raw_en_${token_tag}"
fi
lm_exp="${LM_EXP:-exp/lm_train_lm_transformer_en_${token_tag}}"

use_word_lm=false
word_vocab_size=65000

./asr.sh                                   \
    --lang en \
    --format_wav_scp_skip_bad_files true   \
    --ngpu 4                               \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type "${token_type}"           \
    --nbpe "${nbpe}"                       \
    --bpemode "${bpemode}"                 \
    --feats_type raw                       \
    --audio_format flac.ark                \
    --asr_config "${asr_config}"           \
    --asr_tag "${asr_tag}"                \
    --lm_config "${lm_config}"             \
    --lm_exp "${lm_exp}"                   \
    --inference_config "${inference_config}"     \
    --inference_asr_model "${inference_asr_model}" \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --use_lm "${use_lm}"                   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
