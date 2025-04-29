#!/usr/bin/env bash

# CTC segmentation example recipe

# Copyright 2017, 2020 Johns Hopkins University (Shinji Watanabe, Xuankai Chang)
# 2020, Technische Universität München, Authors: Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# general configuration
python=python3

asr_train_config=exp/asr_train_conformer_ctc_raw_en_bpe2048_sp/config.yaml
asr_model_file=exp/asr_train_conformer_ctc_raw_en_bpe2048_sp/valid.cer_ctc.ave_10best.pth
# wav_scp=data/train_clean_100/wav.scp
# text_file=data/train_clean_100/text
# output=data/train_clean_100/forced_alignments

wav_scp=data/dev/wav.scp
text_file=data/dev/text
output=data/dev/forced_alignments

# audio=test/test_clean_1.flac
# text=test/test_clean_1.txt
# output=test/aligned_segments

# python -m espnet2.bin.asr_align \
#     --asr_train_config "${asr_train_config}" \
#     --asr_model_file "${asr_model_file}" \
#     --audio "${audio}" \
#     --text "${text}" \
#     --output "${output}"

python -m espnet2.bin.asr_align \
    --asr_train_config "${asr_train_config}" \
    --asr_model_file "${asr_model_file}" \
    --wav_scp "${wav_scp}" \
    --text_file "${text_file}" \
    --output "${output}"
