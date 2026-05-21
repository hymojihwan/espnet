#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# 1) prepare noisy 960 data (no speed perturbation)
# ./local/data_noisy_spl.sh --stage 1 --stop_stage 3

# 2) run four SPL models with minimal stats recomputation
# - ASR-only and SpecAug+ASR share the same stats
# - SE+ASR and SE+JEPA+ASR each need their own stats

# A) ASR-only: compute stats + train/decode
# ./run_asr_ctc_spl_asr_main.sh \
#   --expdir exp_spl_asr \
#   --asr_stats_dir exp_spl_asr/asr_stats_shared_asr_specaug \
#   --asr_tag train_asr_ctc_spl_asr_raw_en_bpe2048 \
#   "$@"

# B) SpecAug+ASR: reuse ASR-only stats (skip stage10)
./run_asr_ctc_spl_specaug_asr_main.sh \
  --expdir exp_spl_asr \
  --asr_stats_dir exp_spl_asr/asr_stats_shared_asr_specaug \
  --asr_tag train_asr_ctc_spl_specaug_asr_raw_en_bpe2048 \
  --stage 11 \
  "$@"

# C) SE+ASR: own stats + train/decode
./run_asr_ctc_spl_se_asr_main.sh \
  --expdir exp_spl_se_asr \
  --asr_stats_dir exp_spl_se_asr/asr_stats_se_asr \
  --asr_tag train_asr_ctc_spl_se_asr_raw_en_bpe2048 \
  "$@"

# D) SE+JEPA+ASR: own stats + train/decode
./run_asr_ctc_spl_se_jepa_asr_main.sh \
  --expdir exp_spl_se_jepa_asr \
  --asr_stats_dir exp_spl_se_jepa_asr/asr_stats_se_jepa_asr \
  --asr_tag train_asr_ctc_spl_se_jepa_asr_raw_en_bpe2048 \
  "$@"
