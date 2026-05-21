# JEPA Frontend for ASR on LibriSpeech-100

This directory contains the recipe for training ASR models with a JEPA (Joint Embedding Predictive Architecture) based frontend for noise reduction.

## Overview

This recipe implements a novel noise reduction frontend based on Joint Embedding Predictive Architecture (JEPA) for Automatic Speech Recognition (ASR). Unlike traditional speech enhancement approaches, JEPA uses joint embedding learning to extract noise-robust features directly, eliminating the need for explicit speech enhancement preprocessing.

## Architecture

The JEPA frontend architecture consists of:

1. **Joint Embedding Module**: Learns joint representations of noisy and clean speech
2. **Predictive Module**: Predicts clean embeddings from noisy input
3. **Noise Reduction**: Implicit noise reduction through embedding space projection

```
Noisy Speech → JEPA Frontend → Clean Embeddings → Pre-encoder → Encoder → Decoder → Text
```

## Key Features

- **No Speech Enhancement Required**: The JEPA frontend performs noise reduction implicitly through joint embedding learning
- **End-to-End Trainable**: The entire pipeline can be trained end-to-end with the ASR model
- **Robust to Noise**: Designed to learn noise-invariant representations

## Directory Structure

```
JEPA_asr1/
├── conf/
│   └── tuning/
│       ├── train_asr_jepa_frontend.yaml  # Training configuration with JEPA frontend
│       └── decode_jepa_transducer.yaml   # Decoding configuration
├── local/
│   ├── data.sh              # Data preparation script
│   ├── data_prep.sh         # LibriSpeech data preparation
│   └── download_and_untar.sh # Download utility
├── run.sh                   # Main execution script
├── asr.sh                   # ASR training script (from ESPnet template)
├── db.sh                    # Database paths configuration
├── cmd.sh                   # Command execution configuration
└── path.sh                  # Environment paths
```

## Usage

### 1. Data Preparation

```bash
# Prepare LibriSpeech-100 data
./local/data.sh
```

### 2. Training

```bash
# Train ASR model with JEPA frontend
./run.sh
```

### 3. Decoding

The decoding is automatically performed after training. To run decoding separately:

```bash
./run.sh --stage 12
```

## Configuration

### JEPA Frontend Parameters

The JEPA frontend configuration can be modified in `conf/tuning/train_asr_jepa_frontend.yaml`:

- `embedding_dim`: Dimension of the joint embedding space (default: 512)
- `predictor_dim`: Dimension of the predictor network (default: 256)
- `num_predictor_layers`: Number of layers in the predictor (default: 2)
- `noise_reduction_weight`: Weight for noise reduction loss (default: 1.0)
- `embedding_loss_weight`: Weight for embedding learning loss (default: 0.5)

## Implementation Notes

**Note**: The actual JEPA frontend implementation needs to be added to ESPnet2. The configuration file assumes a frontend type `jepa` will be available. To implement:

1. Create `espnet2/asr/frontend/jepa.py` implementing `AbsFrontend`
2. Register it in `espnet2/tasks/asr.py` in the `frontend_choices`
3. Implement the joint embedding and predictive architecture according to JEPA principles

## Differences from Traditional Enhancement-Based Approaches

- **No Explicit Enhancement**: Unlike enhancement-based approaches that first enhance speech and then do ASR, JEPA learns noise-robust features directly
- **Joint Learning**: The embedding space is learned jointly with the ASR task
- **Predictive Architecture**: Uses predictive coding principles to learn robust representations

## References

- Joint Embedding Predictive Architecture (JEPA) for self-supervised learning
- LibriSpeech dataset: http://www.openslr.org/12/

## License

This recipe follows the same license as ESPnet (Apache 2.0).
