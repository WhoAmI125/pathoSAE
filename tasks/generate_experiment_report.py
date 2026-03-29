from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MODELS = ["vanilla", "gated", "topk", "jumprelu", "msae"]


@dataclass
class FeatureSummaryStats:
    path: Path
    row_count: int
    finite_entropy_count: int
    unique_class_coverage: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PathoSAEv3 full experiment summary report")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument(
        "--output",
        type=str,
        default="results/summary/experiment_design_full_report.md",
        help="Output markdown path",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_feature_summary_stats(path: Path) -> FeatureSummaryStats | None:
    rows = read_csv_rows(path)
    if not rows:
        return None

    finite_entropy = 0
    coverage_values: set[int] = set()

    for row in rows:
        entropy = row.get("entropy", "")
        if entropy not in {"", "nan", "NaN", "None"}:
            try:
                float(entropy)
                finite_entropy += 1
            except ValueError:
                pass

        cov = row.get("class_coverage", "")
        try:
            coverage_values.add(int(float(cov)))
        except Exception:
            pass

    return FeatureSummaryStats(
        path=path,
        row_count=len(rows),
        finite_entropy_count=finite_entropy,
        unique_class_coverage=len(coverage_values),
    )


def parse_monosemanticity_from_log(log_path: Path) -> dict[str, str] | None:
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")

    def _extract(pattern: str) -> str | None:
        matches = re.findall(pattern, txt)
        return matches[-1].strip() if matches else None

    vg = _extract(r"Vinje-Gallant Sparseness\s*:\s*([0-9.]+)")
    active = _extract(r"Active features\s*:\s*([0-9]+/[0-9]+\s*\([^)]+\))")
    selective = _extract(r"Selective features\s*:\s*([0-9.]+%)")
    saved = _extract(r"저장됨:\s*(.+top10_features_z\.png)")

    if not any([vg, active, selective, saved]):
        return None
    return {
        "vg": vg or "-",
        "active": active or "-",
        "selective": selective or "-",
        "saved": saved or "-",
    }


def latest_feature_viz_full_log(log_dir: Path, model: str) -> Path | None:
    logs = sorted(log_dir.glob(f"feature_viz_{model}_full_*.log"))
    return logs[-1] if logs else None


def suspicious_similarity_checks(project_root: Path) -> list[str]:
    warnings: list[str] = []
    vanilla = project_root / "results" / "vanilla_e32_ep10"
    gated = project_root / "results" / "gated_e32_ep10"

    checks = [
        ("feature_summary.csv", vanilla / "feature_summary.csv", gated / "feature_summary.csv"),
        (
            "group/class_feature_matrix.csv",
            vanilla / "group" / "class_feature_matrix.csv",
            gated / "group" / "class_feature_matrix.csv",
        ),
        (
            "group/patient_feature_matrix.csv",
            vanilla / "group" / "patient_feature_matrix.csv",
            gated / "group" / "patient_feature_matrix.csv",
        ),
    ]

    for label, p1, p2 in checks:
        h1 = hash_file(p1)
        h2 = hash_file(p2)
        if h1 is None or h2 is None:
            continue
        if h1 == h2:
            warnings.append(f"[WARNING] vanilla and gated are byte-identical for `{label}`")

    return warnings


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    results_root = project_root / "results"
    logs_root = project_root / "logs"

    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Core metrics table
    comparison_csv = results_root / "comparison" / "comparison_table.csv"
    comparison_rows = read_csv_rows(comparison_csv)

    lines: list[str] = []
    lines.append("# PathoSAEv3 Experiment Design Full Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Project root: `{project_root}`")
    lines.append("")

    lines.append("## 1) Reconstruction Metrics (comparison_table.csv)")
    lines.append("")
    if comparison_rows:
        lines.append("| run_name | model_type | mse | fvu | l0 | sparsity | cosine_sim | active | dead | total |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in comparison_rows:
            lines.append(
                "| {run_name} | {model_type} | {mse} | {fvu} | {l0} | {sparsity} | {cosine_sim} | {active_neurons} | {dead_neurons} | {total_neurons} |".format(
                    **row
                )
            )
    else:
        lines.append("- `results/comparison/comparison_table.csv` not found or empty.")
    lines.append("")

    lines.append("## 2) Feature Entropy / Coverage Summary")
    lines.append("")
    lines.append("| model | feature_summary.csv | rows | finite_entropy_rows | unique_class_coverage_values |")
    lines.append("|---|---|---:|---:|---:|")
    for model in MODELS:
        run_dir = results_root / f"{model}_e32_ep10"
        fs = run_dir / "feature_summary.csv"
        stats = collect_feature_summary_stats(fs)
        if stats is None:
            lines.append(f"| {model} | missing | 0 | 0 | 0 |")
        else:
            lines.append(
                f"| {model} | present | {stats.row_count} | {stats.finite_entropy_count} | {stats.unique_class_coverage} |"
            )
    lines.append("")

    lines.append("## 3) Group Matrix / Heatmap Summary")
    lines.append("")
    lines.append("| model | class_matrix_rows | patient_matrix_rows | class_heatmap | patient_heatmap |")
    lines.append("|---|---:|---:|---|---|")
    for model in MODELS:
        run_dir = results_root / f"{model}_e32_ep10" / "group"
        class_rows = len(read_csv_rows(run_dir / "class_feature_matrix.csv"))
        patient_rows = len(read_csv_rows(run_dir / "patient_feature_matrix.csv"))
        class_heatmap = "yes" if (run_dir / "class_feature_heatmap.png").exists() else "no"
        patient_heatmap = "yes" if (run_dir / "patient_feature_heatmap.png").exists() else "no"
        lines.append(
            f"| {model} | {class_rows} | {patient_rows} | {class_heatmap} | {patient_heatmap} |"
        )
    lines.append("")

    lines.append("## 4) Monosemantic Proxy (from feature_viz full logs)")
    lines.append("")
    lines.append("| model | VG sparseness | active features | selective features |")
    lines.append("|---|---:|---|---:|")
    for model in MODELS:
        log_path = latest_feature_viz_full_log(logs_root, model)
        parsed = parse_monosemanticity_from_log(log_path) if log_path else None
        if parsed is None:
            lines.append(f"| {model} | - | - | - |")
        else:
            lines.append(
                f"| {model} | {parsed['vg']} | {parsed['active']} | {parsed['selective']} |"
            )
    lines.append("")

    lines.append("## 5) Observations and Risks")
    lines.append("")
    warnings = suspicious_similarity_checks(project_root)
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- No byte-identical vanilla-vs-gated artifacts detected in checked files.")

    lines.append("- If any model has missing `feature_summary.csv` or group outputs, rerun evaluate/group steps for that model.")
    lines.append("- If class rows are too small, check `meta_csv` quality and `min_group_samples` threshold.")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
