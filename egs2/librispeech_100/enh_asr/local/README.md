# Local Scripts for Enhancement + ASR Transducer

이 디렉토리에는 Enhancement + ASR Transducer 학습을 위한 로컬 스크립트들이 포함되어 있습니다.

## 스크립트 목록

### 1. `download_musan.sh`
MUSAN 데이터셋을 다운로드하는 스크립트입니다.

**사용법:**
```bash
bash local/download_musan.sh
```

**기능:**
- OpenSLR에서 MUSAN 데이터셋을 다운로드
- `data/musan/` 디렉토리에 압축 해제
- 이미 존재하는 경우 스킵

### 2. `prepare_noisy_data.sh`
LibriSpeech 데이터에 MUSAN noise를 섞어서 noisy 데이터를 생성하는 스크립트입니다.

**사용법:**
```bash
bash local/prepare_noisy_data.sh \
    --clean_data_dir "data" \
    --noisy_data_dir "data_noisy" \
    --musan_dir "data/musan" \
    --snr_range "5:15" \
    --noise_apply_prob 1.0
```

**옵션:**
- `--clean_data_dir`: 깨끗한 LibriSpeech 데이터 디렉토리 (기본값: data)
- `--noisy_data_dir`: noisy 데이터를 저장할 디렉토리 (기본값: data_noisy)
- `--musan_dir`: MUSAN noise 데이터 디렉토리 (기본값: data/musan)
- `--snr_range`: SNR 범위 (기본값: 5:15)
- `--noise_apply_prob`: noise 적용 확률 (기본값: 1.0)

**기능:**
- LibriSpeech의 모든 데이터셋에 대해 noisy 버전 생성
- MUSAN의 noise, music, speech를 랜덤하게 선택하여 추가
- sox를 사용하여 audio mixing 수행
- text, utt2spk, spk2utt 파일은 그대로 복사

## 데이터 구조

### 입력 데이터 구조
```
data/
├── train_clean_100/
│   ├── wav.scp
│   ├── text
│   ├── utt2spk
│   └── spk2utt
├── dev/
├── test_clean/
└── test_other/
```

### 출력 데이터 구조
```
data_noisy/
├── train_clean_100/
│   ├── wav.scp          # Noisy audio paths
│   ├── text             # Same as clean
│   ├── utt2spk          # Same as clean
│   ├── spk2utt          # Same as clean
│   └── wav/             # Noisy audio files
├── dev/
├── test_clean/
└── test_other/
```

## 요구사항

- **sox**: Audio processing을 위해 필요
- **wget 또는 curl**: MUSAN 다운로드를 위해 필요
- **Python 3**: Noisy 데이터 생성 스크립트 실행을 위해 필요

## 설치

```bash
# sox 설치 (Ubuntu/Debian)
sudo apt-get install sox

# sox 설치 (CentOS/RHEL)
sudo yum install sox

# sox 설치 (macOS)
brew install sox
```

## 사용 예제

### 1. MUSAN 다운로드
```bash
bash local/download_musan.sh
```

### 2. Noisy 데이터 생성
```bash
bash local/prepare_noisy_data.sh \
    --snr_range "0:10" \
    --noise_apply_prob 0.8
```

### 3. 전체 파이프라인 실행
```bash
# stage 0부터 시작 (데이터 준비 포함)
bash run.sh --stage 0 --enh_asr_config conf/tuning/enh_asr_transducer/conv_tasnet_se_only.yaml
``` 