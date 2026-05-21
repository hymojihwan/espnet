# OT-based Feature Alignment for Enhancement + ASR Transducer

## Overview

이 모델은 **Optimal Transport (OT)**를 사용하여 noisy features를 enhanced features로 정렬하는 새로운 방법론입니다.

## 기존 방법론과의 차이

### 1. **Baseline (기존 방법)**
- Enhanced speech만 사용
- Noisy 정보 손실

### 2. **Feature Fusion (기존 방법)**
- Enhanced features + Noisy features를 concatenate
- Learnable fusion layer로 결합
- 파일: `espnet_enh_fusion_transducer_model.py`

### 3. **OT-based Feature Alignment (새로운 방법)** ⭐
- Noisy features를 Enhanced features로 OT 정렬
- Fusion 없이 변환된 특징만 사용
- 파일: `espnet_enh_ot_align_transducer_model.py`

## 아키텍처

```
Noisy Speech
    ↓
ConvTasNet Enhancement
    ↓
Enhanced Speech
    
    ┌─────────────────┬─────────────────┐
    ↓                 ↓                 
Enhanced Features   Noisy Features
                      ↓
                  OT Alignment
                  (noisy → enhanced)
                      ↓
                 Aligned Features
                      ↓
                 ASR Encoder
                      ↓
                Recognition Result
```

## OT Alignment 방법

1. **Enhanced features**와 **Noisy features**를 추출
2. **Sinkhorn-Knopp algorithm**을 사용하여 OT plan 계산
   - Cost matrix: **Cosine similarity** 기반 (1 - cosine_similarity)
   - Entropic regularization: `epsilon` (default: 0.1)
   - Maximum iterations: `max_iter` (default: 5)
3. OT plan을 적용하여 noisy features를 enhanced features 분포로 변환
4. 변환된 특징을 ASR encoder에 입력

## 학습 방법

### 설정 파일
- **Config**: `conf/tuning/enh_asr_transducer/conv_tasnet_ot_alignment.yaml`
- **Run script**: `run_ot_alignment.sh`

### 실행 명령어

```bash
cd /home/user/Workspace/espnet/egs2/librispeech_100/enh_asr
bash run_ot_alignment.sh
```

### 주요 설정

```yaml
model_conf:
  # OT alignment 모델 활성화
  use_ot_align: true
  
  # Loss weights
  enh_weight: 0.2
  transducer_weight: 1.0
  
  # OT parameters
  epsilon: 0.1  # Sinkhorn entropic regularization
  max_iter: 5   # Maximum iterations for Sinkhorn algorithm
```

## 파일 구조

### 모델 파일
- `espnet2/asr_transducer/espnet_enh_ot_align_transducer_model.py`
  - OT-based feature alignment 모델 구현

### 설정 파일
- `conf/tuning/enh_asr_transducer/conv_tasnet_ot_alignment.yaml`
  - 학습 설정

### 실행 스크립트
- `run_ot_alignment.sh`
  - 학습 실행 스크립트

## 주요 차이점 요약

| 방법 | 파일명 | 특징 결합 방식 | 학습 가능한 파라미터 |
|------|--------|----------------|---------------------|
| Fusion | `espnet_enh_fusion_transducer_model.py` | Concat + Learnable Layer | ✅ (Fusion layer) |
| OT Align | `espnet_enh_ot_align_transducer_model.py` | OT Alignment | ❌ (OT는 비학습) |

## 장점

1. **명시적인 정렬**: OT를 사용하여 noisy features를 enhanced features 분포로 명시적으로 정렬
2. **학습 파라미터 없음**: Fusion layer 없이 OT만으로 정렬 수행
3. **이론적 근거**: Optimal Transport의 수학적 기반
4. **유연성**: `epsilon`과 `max_iter` 파라미터로 정렬 강도 조절 가능
5. **효율성**: PyTorch native 구현으로 GPU 가속 지원
6. **의미적 정렬**: Cosine similarity 기반 cost matrix로 특징의 방향성 고려

## 의존성

- 외부 라이브러리 불필요 (PyTorch만 사용)
- 이전 버전과 달리 POT 라이브러리 없이도 동작

## 참고 사항

- OT alignment는 GPU에서 계산됨 (PyTorch 기반)
- Cosine similarity 기반 cost matrix 사용
- Batch 내 각 샘플을 독립적으로 처리
- Enhanced features는 target distribution으로 사용
- Noisy features가 enhanced distribution으로 변환됨

