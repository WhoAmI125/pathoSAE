# architecture.md
> Opus 작성 · Codex 참조 · `v1.0.0` · 2026-02-20

---

## System Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║  PathoSAEv3 — Pathology SAE for Interpretable Feature Learning     ║
║                                                                    ║
║  ┌────────────┐    ┌────────────┐    ┌────────────┐               ║
║  │  Extract    │───▶│   Train     │───▶│  Evaluate   │              ║
║  │            │    │            │    │            │              ║
║  │ ViT → Acts │    │ SAE 학습   │    │ 메트릭/시각화│              ║
║  └────────────┘    └────────────┘    └────────────┘              ║
║       │                 │                  │                      ║
║       ▼                 ▼                  ▼                      ║
║  data/activations/ models/checkpoints/ results/                   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Module Definitions

### Module 1: Extract (`src/extract/`)
- **책임**: EXAONEPath ViT에서 activation 추출
- **입력**: WSI 타일 이미지 (224x224 PNG)
- **출력**: `data/activations/spatial_folder_XX.pt` — `{vectors: [N, 196, 768], paths: [...]}`
- **의존**: EXAONEPath 체크포인트, torchstain (Macenko)
- **제약**: GPU 메모리, 기존 추출 데이터 재사용 가능 (symlink)

### Module 2: Models (`src/models/`)
- **책임**: SAE 모델 정의 (4가지 variant)
- **핵심**: 공통 `BaseSAE` → `VanillaSAE`, `GatedSAE`, `TopKSAE`, `JumpReLUSAE`
- **제약**: 모든 variant가 동일 인터페이스 (`forward() → sae_out, feature_acts, loss_dict`)

### Module 3: Train (`src/training/`)
- **책임**: 통합 학습 루프
- **입력**: Config + Activations
- **출력**: 체크포인트 (`models/checkpoints/{run_name}/`)
- **핵심**: `SAETrainer` 클래스 — 모든 variant 통합 학습
- **제약**: 동일 hyperparameter로 모든 variant 학습 (논문 비교용)

### Module 4: Evaluate (`src/evaluation/`)
- **책임**: 메트릭 계산, 시각화, 비교 분석
- **입력**: 학습된 체크포인트 + 데이터
- **출력**: `results/{run_name}/` — metrics.json, plots, feature_data
- **핵심**: variant 간 공정 비교 지표

### Module 5: Config (`src/config.py`)
- **책임**: 모든 hyperparameter 중앙 관리
- **핵심**: 단일 dataclass, CLI override 지원
- **제약**: variant별 차이는 최소화, 공통 파라미터 통일

---

## Data Flow

```
[WSI Tiles]
    │
    ▼ extract (1회만, symlink 가능)
[data/activations/spatial_folder_XX.pt]  ─── {vectors: [N, 196, 768]}
    │
    ▼ flatten
[N*196, 768] patches
    │
    ▼ train --model {vanilla|gated|topk|jumprelu}
[models/checkpoints/{run_name}/best.pt]
    │
    ▼ evaluate
[results/{run_name}/metrics.json + plots/]
    │
    ▼ compare (all variants)
[results/comparison/]  ─── 논문용 표/그래프
```

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Lang | Python 3.10 | pathoSAEv2 호환, PyTorch 2.x 지원 |
| DL | PyTorch 2.2.0 + CUDA 11.8 | 기존 체크포인트 호환 |
| ViT | EXAONEPath (ViT-B/16, 768d) | 병리 도메인 특화 |
| Stain | torchstain (Macenko) | 기존 파이프라인 유지 |
| Logging | wandb | 실험 추적 |
| CLI | argparse | 단순, 의존성 없음 |
| Env | conda (pathosae3) | 격리된 환경 |

---

## Directory Structure

```
pathoSAEv3/
├── src/
│   ├── __init__.py
│   ├── config.py                    # 통합 Config dataclass
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_sae.py              # BaseSAE (공통 인터페이스)
│   │   ├── vanilla_sae.py           # ReLU + L1
│   │   ├── gated_sae.py             # Gate + ReLU + L1
│   │   ├── topk_sae.py              # TopK activation (MSAE 통합)
│   │   └── jumprelu_sae.py          # JumpReLU + L0
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # SAETrainer (통합 학습 루프)
│   │   ├── losses.py                # 손실 함수 모음
│   │   └── scheduler.py             # LR 스케줄러
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # MSE, FVU, L0, sparsity 등
│   │   ├── feature_analysis.py      # feature별 통계, top-k 이미지
│   │   ├── visualize.py             # 시각화 유틸
│   │   └── compare.py               # variant 간 비교 표/그래프
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── vit_encoder.py           # EXAONEPath 로딩 + hook
│   │   ├── activation_store.py      # activation 로딩/스트리밍
│   │   └── macenko.py               # stain normalization
│   └── utils.py                     # 공통 유틸 (GPU 탐색, seed 등)
├── tasks/
│   ├── train.py                     # 통합 학습 엔트리포인트
│   ├── extract.py                   # activation 추출 엔트리포인트
│   ├── evaluate.py                  # 평가 엔트리포인트
│   └── compare.py                   # variant 비교 엔트리포인트
├── scripts/
│   ├── train_all.sh                 # 4 variant 순차 학습
│   └── evaluate_all.sh              # 4 variant 순차 평가
├── data/
│   └── activations/                 # symlink → pathoSAEv2/data/encode_spatial
├── models/
│   ├── backbone/
│   │   └── EXAONEPath.ckpt          # symlink → 기존 체크포인트
│   └── checkpoints/                 # 학습된 SAE 체크포인트
├── results/                         # 평가 결과
├── configs/                         # 데이터셋별 클래스명 매핑
│   └── classnames/
├── .pipeline/                       # Opus-Codex 파이프라인 문서
├── .claude/                         # Claude Code 스킬
├── CLAUDE.md
├── environment.yaml                 # conda 환경 정의
└── requirements.txt
```

---

## Error Strategy

```
정상:  Load Config → Load Data → Init Model → Train → Save → Evaluate
에러:  GPU OOM → batch_size 자동 감소 or 에러 로그
       Data Missing → 명확한 에러 메시지 + 경로 안내
       Checkpoint Missing → skip (evaluate 시)
```

---

> NOTE: **CODEX**: 이 구조를 임의 변경 금지. 변경 필요 시 `codex_notes.md`에 기록.
