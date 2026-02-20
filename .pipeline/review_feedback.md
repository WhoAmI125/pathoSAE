# review_feedback.md
> Opus 작성 · Codex 반영

---

## Review #1 — P0 (T-001~T-006) 구현 리뷰

**날짜**: 2026-02-20
**상태**: **APPROVED** (minor fix applied by Opus)

### 리뷰 결과

| 파일 | 판정 | 비고 |
|------|------|------|
| `src/config.py` | OK | contracts 완벽 일치 |
| `src/models/base_sae.py` | OK | encode/activate/decode/forward, unit norm, grad projection |
| `src/models/__init__.py` | OK | MODEL_REGISTRY + create_sae factory |
| `src/models/vanilla_sae.py` | OK | ReLU + L1, warmup scale |
| `src/models/gated_sae.py` | OK | b_gate, gated activation |
| `src/models/topk_sae.py` | FIXED | ReLU 누락 → Opus가 직접 수정 |
| `src/models/jumprelu_sae.py` | OK | Custom autograd (JumpReLU + StepFunction) |
| `src/extract/activation_store.py` | OK | IterableDataset, file-based split, compute_train_mean |
| `src/training/trainer.py` | OK | QD-001/002 (decoder norm + grad projection) 매 step |
| `src/training/losses.py` | OK | 유틸리티 함수 |
| `src/training/scheduler.py` | OK | warmup + cosine decay |
| `tasks/train.py` | OK | CLI 통합, QD-003 (b_dec mean init) |

### 수정 사항 (Opus 직접 수정)
1. `topk_sae.py:24` — `out.scatter_(-1, indices, values)` → `out.scatter_(-1, indices, torch.relu(values))`
   - contracts에 명시된 대로 음수 값을 ReLU로 제거

### Smoke Test 결과
```
vanilla: out=[8, 768], acts=[8, 24576], loss=2.23, l0=12282.0
gated:   out=[8, 768], acts=[8, 24576], loss=2.30, l0=12288.1
topk:    out=[8, 768], acts=[8, 24576], loss=1.04, l0=64.0  ← k 정확
jumprelu: out=[8, 768], acts=[8, 24576], loss=2.99, l0=10313.0
```

### ADR/QD 준수 확인
- [x] ADR-001: 모든 variant 동일 Config
- [x] ADR-002: BaseSAE + Factory
- [x] ADR-003: 단일 tasks/train.py --model
- [x] ADR-005: TopK 단일 k=64
- [x] ADR-006: JumpReLU custom autograd
- [x] QD-001: set_decoder_norm_to_unit_norm() 매 step
- [x] QD-002: remove_gradient_parallel_to_decoder_directions() 매 step
- [x] QD-003: b_dec = mean(train_data) 초기화
- [x] QD-004: float32 고정 (AMP 없음)
- [x] QD-005: 파일 단위 val split
