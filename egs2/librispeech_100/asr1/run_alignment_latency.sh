#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Configuration
exp_dir="exp/asr_conformer-rnnt-streaming_raw_en_bpe2048_sp"
model_file="valid.loss.ave_10best.pth"
test_set="test_clean"
output_dir="${exp_dir}/decode_transducer_asr_model_${model_file}/${test_set}_alignment_latency"

# Create alignment directory if it doesn't exist
mkdir -p data/alignment

# Run inference with alignment-based latency measurement
python -m espnet2.bin.asr_transducer_inference \
    --output_dir "${output_dir}" \
    --batch_size 1 \
    --ngpu 1 \
    --dtype "float32" \
    --seed 0 \
    --num_workers 0 \
    --log_level "INFO" \
    --data_path_and_name_and_type "data/${test_set}/wav.scp,speech,sound" \
    --key_file "${output_dir}/keys.1.scp" \
    --asr_train_config "${exp_dir}/config.yaml" \
    --asr_model_file "${exp_dir}/${model_file}" \
    --beam_size 5 \
    --nbest 1 \
    --streaming True \
    --decoding_window 640 \
    --left_context 32 \
    --display_hypotheses False \
    --min_duration 0.1 \
    --max_duration 30.0 \
    --use_alignment_latency True \
    --alignment_dir "data/alignment"

echo "Alignment-based latency measurement completed."
echo "Results saved in: ${output_dir}" 