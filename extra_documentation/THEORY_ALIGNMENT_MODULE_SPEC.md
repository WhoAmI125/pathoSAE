# PathoSAEv3 Theory Alignment + Module Handoff Spec (v1)

## 1) Goal and Scope

- Goal: lock a single decision-complete handoff document for the next implementation phase.
- Scope: theory alignment between PathoSAEv3 codebase and base papers, plus module-based ownership split.
- Out of scope: code/CLI/checkpoint format changes in this phase.

Assumptions:
- Working tree state is used as-is.
- Paper interpretation prioritizes values/definitions in:
  - `/home/kimhj/projects/cytoSAE/cytosae.pdf`
  - `/home/kimhj/projects/cytoSAE/plutosae.pdf`

---

## 2) Theory Alignment (Papers -> PathoSAEv3)

### 2.1 Common SAE Objective Axis

Base objective used across both papers:

\[
\mathcal{L} = \mathcal{L}_{recon}(x, \hat{x}) + \lambda \cdot \mathcal{L}_{sparsity}(z)
\]

- Reconstruction term: MSE.
- Sparsity term: L1/L0/TopK-style constraints by variant.

Code anchors:
- Base interface and forward contract:
  - [`src/models/base_sae.py:38`](/home/kimhj/projects/pathoSAEv3/src/models/base_sae.py:38)
  - [`src/models/base_sae.py:55`](/home/kimhj/projects/pathoSAEv3/src/models/base_sae.py:55)
- Trainer loop uses `loss/mse_loss/sparsity_loss/l0`:
  - [`src/training/trainer.py:112`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:112)

### 2.2 Variant-by-Variant Mapping

| Variant | Theory intent | PathoSAEv3 implementation anchor |
|---|---|---|
| Vanilla | `ReLU + L1` sparse dictionary baseline | [`src/models/vanilla_sae.py:15`](/home/kimhj/projects/pathoSAEv3/src/models/vanilla_sae.py:15), [`src/models/vanilla_sae.py:18`](/home/kimhj/projects/pathoSAEv3/src/models/vanilla_sae.py:18) |
| Gated | gate/magnitude separation, sparsity on gate path | [`src/models/gated_sae.py:31`](/home/kimhj/projects/pathoSAEv3/src/models/gated_sae.py:31), [`src/models/gated_sae.py:37`](/home/kimhj/projects/pathoSAEv3/src/models/gated_sae.py:37) |
| TopK | hard-k activation instead of L1 penalty | [`src/models/topk_sae.py:17`](/home/kimhj/projects/pathoSAEv3/src/models/topk_sae.py:17), [`src/models/topk_sae.py:27`](/home/kimhj/projects/pathoSAEv3/src/models/topk_sae.py:27) |
| JumpReLU | learned threshold + L0 proxy | [`src/models/jumprelu_sae.py:107`](/home/kimhj/projects/pathoSAEv3/src/models/jumprelu_sae.py:107), [`src/models/jumprelu_sae.py:110`](/home/kimhj/projects/pathoSAEv3/src/models/jumprelu_sae.py:110) |
| MSAE | multi-scale TopK with weighted multi-level MSE | [`src/models/msae.py:63`](/home/kimhj/projects/pathoSAEv3/src/models/msae.py:63), [`src/models/msae.py:95`](/home/kimhj/projects/pathoSAEv3/src/models/msae.py:95) |
| DriftPrior | distribution matching to sparse prior in latent space | [`src/models/driftprior_sae.py:29`](/home/kimhj/projects/pathoSAEv3/src/models/driftprior_sae.py:29), [`src/models/driftprior_sae.py:117`](/home/kimhj/projects/pathoSAEv3/src/models/driftprior_sae.py:117) |

Registry/CLI coverage:
- Registry has 6 variants: [`src/models/__init__.py:12`](/home/kimhj/projects/pathoSAEv3/src/models/__init__.py:12)
- Train CLI allows same 6 variants: [`tasks/train.py:57`](/home/kimhj/projects/pathoSAEv3/tasks/train.py:57)

### 2.3 Interpretability Axis

Paper concepts to keep:
- Entropy/monosemanticity (`p_i = |rho_i| / sum |rho_j|`, low entropy -> more monosemantic).
- Sparse probe utility (`k`-sparse linear probe, R2/F1 style downstream validation).
- Barcode hierarchy (patch -> image -> patient/group).

PathoSAEv3 anchors:
- Entropy proxy from top patches + optional metadata:
  - [`src/evaluation/feature_figures.py:44`](/home/kimhj/projects/pathoSAEv3/src/evaluation/feature_figures.py:44)
  - [`src/evaluation/feature_figures.py:60`](/home/kimhj/projects/pathoSAEv3/src/evaluation/feature_figures.py:60)
- Group/patient aggregation:
  - [`src/evaluation/group_analysis.py:157`](/home/kimhj/projects/pathoSAEv3/src/evaluation/group_analysis.py:157)
- Monosemanticity-oriented MS score pipeline:
  - [`src/evaluation/ms_score.py:163`](/home/kimhj/projects/pathoSAEv3/src/evaluation/ms_score.py:163)
  - [`tasks/eval_ms_score.py:37`](/home/kimhj/projects/pathoSAEv3/tasks/eval_ms_score.py:37)

---

## 3) Module-Based Handoff Spec

This section is implementation ownership ready. One engineer can pick one module and execute independently with defined I/O.

### Module A: Extract

Responsibility:
- `PNG tile -> normalized tensor -> spatial token activation -> .pt file`.

Entrypoints:
- [`tasks/extract.py:142`](/home/kimhj/projects/pathoSAEv3/tasks/extract.py:142)
- [`src/extract/vit_encoder.py:423`](/home/kimhj/projects/pathoSAEv3/src/extract/vit_encoder.py:423)

Input contract:
- Folder of PNG tiles.
- Encoder choice `{exaonepath, uni}`.

Output contract:
- Per-folder activation file:
  - key `vectors`: shape `[N, 196, D]`
  - key `paths`: list length `N`
  - save logic: [`tasks/extract.py:193`](/home/kimhj/projects/pathoSAEv3/tasks/extract.py:193)

Dependencies:
- `macenko`, encoder backbone checkpoint, GPU selection.

Definition of done:
- Files are readable by `ActivationStore` and feature/group analyzers.

### Module B: Train

Responsibility:
- Build config/model/store, initialize decoder bias, run unified trainer, save checkpoints.

Entrypoints:
- [`tasks/train.py:126`](/home/kimhj/projects/pathoSAEv3/tasks/train.py:126)
- [`src/training/trainer.py:23`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:23)

Input contract:
- Activation files under `data/activations/spatial_folder_*.pt`.
- CLI selects variant and training hyperparameters.

Output contract:
- Checkpoint folder with `best.pt`, `final.pt`, `config.json`.
- Checkpoint payload includes `state_dict`, `config`, optimizer state:
  - [`src/training/trainer.py:79`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:79)

Dependencies:
- `create_sae` registry, `ActivationStore`, scheduler, optional wandb.

Definition of done:
- Variant trains end-to-end and can be loaded by `BaseSAE.load`.

### Module C: Eval/Analysis

Responsibility:
- Reconstruction metrics, feature stats, group-level matrices, optional MS score.

Entrypoints:
- Reconstruction eval: [`tasks/evaluate.py:84`](/home/kimhj/projects/pathoSAEv3/tasks/evaluate.py:84)
- Compare: [`src/evaluation/compare.py:131`](/home/kimhj/projects/pathoSAEv3/src/evaluation/compare.py:131)
- Group analysis: [`src/evaluation/group_analysis.py:156`](/home/kimhj/projects/pathoSAEv3/src/evaluation/group_analysis.py:156)
- MS score: [`tasks/eval_ms_score.py:87`](/home/kimhj/projects/pathoSAEv3/tasks/eval_ms_score.py:87)

Input contract:
- Trained checkpoint(s).
- Activation directory.
- Optional metadata CSV (`path,class,patient`).

Output contract:
- `results/{run_name}/metrics.json` and plots.
- `feature_analysis.json`, `feature_summary.csv`, group matrices/heatmaps.
- `results/comparison/comparison_table.csv`, Pareto plot.

Dependencies:
- Consistent activation path-key consumption across features/group/ms-score.

Definition of done:
- All generated artifacts are internally consistent and traceable to input checkpoints.

### Module D: Report

Responsibility:
- Aggregate all experiment artifacts into one final markdown report.

Entrypoint:
- [`tasks/generate_experiment_report.py:151`](/home/kimhj/projects/pathoSAEv3/tasks/generate_experiment_report.py:151)

Input contract:
- Existing `results/`, `logs/`, optional comparison csv.

Output contract:
- `results/summary/experiment_design_full_report.md`.

Dependencies:
- Stable output filenames from Eval module.

Definition of done:
- Report can be regenerated without manual edits.

---

## 4) Interface Lock Decisions (to apply in next implementation phase)

These are now fixed defaults for implementation planning.

### Lock-1: Activation key compatibility (`paths` vs `patch_paths`)

Decision:
- Canonical write key: `paths` for extract outputs.
- Canonical read behavior: accept both `paths` and `patch_paths` everywhere.

Rationale:
- Extract currently writes `paths` ([`tasks/extract.py:196`](/home/kimhj/projects/pathoSAEv3/tasks/extract.py:196)).
- Some consumers still hard-require `patch_paths`:
  - [`tasks/feature_viz.py:202`](/home/kimhj/projects/pathoSAEv3/tasks/feature_viz.py:202)
  - [`src/evaluation/ms_score.py:123`](/home/kimhj/projects/pathoSAEv3/src/evaluation/ms_score.py:123)

Acceptance for implementation:
- No consumer directly assumes only one key.

### Lock-2: `evaluate --split` behavior for reconstruction metrics

Decision:
- `--split` must control both:
  - feature analysis split, and
  - reconstruction metric split.
- Default remains `val`.

Rationale:
- Current mismatch:
  - split parsed in CLI: [`tasks/evaluate.py:65`](/home/kimhj/projects/pathoSAEv3/tasks/evaluate.py:65)
  - reconstruction path always uses val loader:
    [`src/evaluation/metrics.py:209`](/home/kimhj/projects/pathoSAEv3/src/evaluation/metrics.py:209)

Acceptance for implementation:
- Same `--split` value yields same data partition across all eval artifacts.

Status (2026-03-29):
- Implemented in code:
  - [`tasks/evaluate.py`](/home/kimhj/projects/pathoSAEv3/tasks/evaluate.py)
  - [`src/evaluation/metrics.py`](/home/kimhj/projects/pathoSAEv3/src/evaluation/metrics.py)
  - [`src/extract/activation_store.py`](/home/kimhj/projects/pathoSAEv3/src/extract/activation_store.py)

### Lock-3: Variant set policy (4/5/6 split)

Decision:
- `Core set`: `vanilla,gated,topk,jumprelu,msae`.
- `Experimental set`: `driftprior` (opt-in only).

Rationale:
- Registry/CLI expose 6 variants:
  - [`src/models/__init__.py:12`](/home/kimhj/projects/pathoSAEv3/src/models/__init__.py:12)
  - [`tasks/train.py:57`](/home/kimhj/projects/pathoSAEv3/tasks/train.py:57)
- Automation scripts are inconsistent:
  - run.sh has 5 variants: [`run.sh:32`](/home/kimhj/projects/pathoSAEv3/run.sh:32)
  - train/evaluate_all have 4 variants: [`scripts/train_all.sh:4`](/home/kimhj/projects/pathoSAEv3/scripts/train_all.sh:4), [`scripts/evaluate_all.sh:4`](/home/kimhj/projects/pathoSAEv3/scripts/evaluate_all.sh:4)

Acceptance for implementation:
- One explicit source of truth for model sets, reused by scripts/CLI docs/report.

---

## 5) Traceability and Scenario Checks

### 5.1 Theory-to-Code Traceability

Minimum requirement:
- Each claim below has at least one code anchor.

| Claim | Anchor |
|---|---|
| SAE uses encode-activate-decode contract | [`src/models/base_sae.py:38`](/home/kimhj/projects/pathoSAEv3/src/models/base_sae.py:38) |
| MSE + sparsity terms are tracked in training | [`src/training/trainer.py:119`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:119) |
| L1/L0/TopK variant behaviors exist distinctly | model files in Section 2.2 |
| Entropy/feature interpretability artifacts exist | [`src/evaluation/feature_figures.py:60`](/home/kimhj/projects/pathoSAEv3/src/evaluation/feature_figures.py:60) |
| Group/patient aggregation exists | [`src/evaluation/group_analysis.py:157`](/home/kimhj/projects/pathoSAEv3/src/evaluation/group_analysis.py:157) |

### 5.2 Pipeline Scenario Checks

Scenario-1: Extract schema consistency
- Verify all downstream readers accept extract output keys.
- Risk currently open due to mixed key expectations.

Scenario-2: Split consistency
- Same split should be used by metrics and feature/group analysis.
- Risk currently open (`evaluate_checkpoint` val-only).

Scenario-3: Parallel ownership feasibility
- Extract/Train/Eval/Report have disjoint primary outputs.
- Dependencies are linear and explicit, so independent assignment is feasible.

---

## 6) Execution-Ready Work Packages (Module Split)

WP-A Extract
- Own files: `tasks/extract.py`, `src/extract/*`.
- Deliverables: stable activation schema + extraction reproducibility notes.
- Depends on: backbone checkpoints and image tiles.

WP-B Train
- Own files: `tasks/train.py`, `src/models/*`, `src/training/*`, `src/config.py`.
- Deliverables: reproducible checkpoints for core model set.
- Depends on: WP-A outputs.

WP-C Eval
- Own files: `tasks/evaluate.py`, `tasks/compare.py`, `tasks/group_analysis.py`, `tasks/eval_ms_score.py`, `src/evaluation/*`.
- Deliverables: metrics/features/group/ms artifacts + comparison table.
- Depends on: WP-B outputs and optional metadata.

WP-D Report
- Own files: `tasks/generate_experiment_report.py`.
- Deliverables: single final markdown report for experiment run.
- Depends on: WP-C outputs.

---

## 7) Acceptance Criteria

Document is accepted only if:

1. A new contributor can pick one module and start implementation with no extra decision requests.
2. The three interface locks in Section 4 are treated as fixed defaults.
3. All high-level theory claims in Section 2 are traceable to concrete code anchors.
4. No code changes are required to understand ownership boundaries and I/O contracts.
