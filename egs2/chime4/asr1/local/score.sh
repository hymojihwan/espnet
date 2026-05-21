#!/usr/bin/env bash
# CHiME4 refs are uppercase (WSJ); model output is often lowercase.
# Re-score WER/CER with lowercased ref and hyp for case-insensitive results.
# Usage: from asr1 dir, run:  local/score.sh exp/asr_train_conformer_ctc_raw_en_char
# Then:  scripts/utils/show_asr_result.sh exp/asr_train_conformer_ctc_raw_en_char > exp/.../RESULTS.md

set -euo pipefail
[ -f path.sh ] && . ./path.sh

if [ $# -lt 1 ]; then
    echo "Usage: $0 <asr_exp_dir>"
    exit 1
fi
exp=$1

for dir in "${exp}"/*/*/score_wer "${exp}"/*/*/score_cer; do
    [ -d "$dir" ] || continue
    [ -f "${dir}/ref.trn" ] && [ -f "${dir}/hyp.trn" ] || continue
    # Lowercase text only (keep last field = utt id unchanged)
    awk '{for(i=1;i<NF;i++) $i=tolower($i); print}' "${dir}/ref.trn" > "${dir}/ref_lc.trn"
    awk '{for(i=1;i<NF;i++) $i=tolower($i); print}' "${dir}/hyp.trn" > "${dir}/hyp_lc.trn"
    sclite -r "${dir}/ref_lc.trn" trn -h "${dir}/hyp_lc.trn" trn -i rm -o all stdout > "${dir}/result.txt"
    echo "Re-scored (lowercase): $dir"
done
