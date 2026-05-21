#!/usr/bin/env bash
# CHiME4 enh+ASR: same *data* and *tokenization* defaults as ../asr1/run_ctc.sh (BPE1024, speed perturb, flac, no LM).
# Stage 1 uses local/data.sh (copy from enh1 + wav/spk fixes + nlsyms).
set -e
set -u
set -o pipefail

_script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${_script_dir}"

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track

# Same order as ../asr1/run_ctc.sh: copy/fix first so per-SNR dirs exist, then build test_sets.
if [ ! -f "data/${train_set}/text" ]; then
    # Optional: LOCAL_DATA_OPTS='--extra-annotations /path/to/annotations' ./run_from_enh1.sh
    # shellcheck disable=SC2086
    ./local/data.sh ${LOCAL_DATA_OPTS:-}
fi

test_sets="et05_multi_isolated_1ch_track et05_simu_isolated_1ch_track"
for d in et05_simu_snr-5_isolated_1ch_track et05_simu_snr0_isolated_1ch_track et05_simu_snr5_isolated_1ch_track et05_simu_snr10_isolated_1ch_track et05_simu_snr15_isolated_1ch_track; do
    [ -d "data/${d}" ] && test_sets="${test_sets} ${d}"
done

# Match chime4 enh_asR published ConvTasNet+Transformer recipe (fbank frontend); change if you use another yaml.
enh_asr_config=conf/tuning/train_enh_asr_convtasnet_si_snr_fbank_transformer_lr2e-3_accum2_warmup20k_specaug.yaml
inference_config=conf/decode_asr_transformer.yaml

# After local/data.sh, nlsyms exists (same as run_ctc.sh).
nlsyms=data/nlsyms.txt
# enh_asr.sh expects --bpe_nlsyms as comma-separated (asr.sh accepts a file and converts)
bpe_nlsyms_list=
if [ -f "${nlsyms}" ]; then
    bpe_nlsyms_list="$(awk '{print $1}' "${nlsyms}" | paste -sd, -)"
fi

speed_perturb_factors="0.9 1.0 1.1"

./enh_asr.sh \
    --lang en \
    --spk_num 1 \
    --ref_channel 3 \
    --token_type bpe \
    --nbpe 1024 \
    --bpemode unigram \
    --nlsyms_txt "${nlsyms}" \
    --bpe_nlsyms "${bpe_nlsyms_list}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --max_wav_duration 30 \
    --audio_format flac \
    --feats_type raw \
    --feats_normalize global_mvn \
    --use_lm false \
    --use_word_lm false \
    --local_data_opts "" \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text_spk1" \
    --lm_train_text "data/${train_set}/text_spk1" \
    "$@"

# Note: bpe/lm text uses the *base* train dir (no _sp), same as ../asr1/run_ctc.sh; speed-perturbed
# training uses data/${train_set}_sp combined in enh_asr stage 2.
