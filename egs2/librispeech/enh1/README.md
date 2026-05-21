# LibriSpeech-100 Enhancement Recipe

This recipe is for speech enhancement only training on LibriSpeech-100 with added noise.

## Usage

### 1. Data Preparation

First, prepare the noisy training data:

```bash
# If you don't have noisy training data, create it with noise
./local/prepare_noisy_train.sh --min_snr 0 --max_snr 20

# Or if you have a noise dataset
./local/prepare_noisy_train.sh --noise_dir /path/to/noise --min_snr 0 --max_snr 20
```

### 2. Enhancement Training

Run the enhancement training:

```bash
./run.sh --stage 1 --stop_stage 10
```

### 3. Inference

```bash
./run.sh --stage 11 --stop_stage 11
```

## Configuration

- Training config: `conf/tuning/train_enh_conv_tasnet.yaml`
- Uses ConvTasNet for speech enhancement
- Single-speaker enhancement with SI-SNR loss

## Data Format

The data should have:
- `wav.scp`: Noisy speech
- `spk1.scp`: Clean reference speech
- `text`, `utt2spk`, `spk2utt`: Standard Kaldi format

## Notes

- Training data: train_clean_100_noisy
- Validation data: dev_clean_noisy
- Test data: test_clean_noisy, test_other_noisy

