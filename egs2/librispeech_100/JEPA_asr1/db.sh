# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

# Clean LibriSpeech path
LIBRISPEECH=/DB

# Noisy LibriSpeech path
NOISY_LIBRISPEECH="/DB/noisy_librispeech"

# For CMU TIR environment
if [[ "$(hostname)" == tir* ]]; then
    LIBRISPEECH=/projects/tir5/data/speech_corpora/LibriSpeech
fi

# For JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    LIBRISPEECH=/DB
fi
