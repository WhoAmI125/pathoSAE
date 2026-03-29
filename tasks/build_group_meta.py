from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import SAEConfig
from src.evaluation.feature_analysis import resolve_activation_files_for_split


@dataclass
class MetaBuildStats:
    total_seen_rows: int = 0
    unique_paths: int = 0
    clinical_matched: int = 0
    unknown_class: int = 0
    duplicate_paths: int = 0
    conflicting_duplicates: int = 0


def normalize_path(path: str) -> str:
    return os.path.normpath(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build group meta CSV(path,class,patient) from activation patch paths"
    )
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--split", type=str, choices=["train", "val", "full"], default="full")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="/home/kimhj/projects/past_try/images/data/tcga_brca_survival/clinical_data(labels).csv",
    )
    parser.add_argument("--patient_col", type=str, default="bcr_patient_barcode")
    parser.add_argument("--class_col", type=str, default="vital_status")
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def load_clinical_map(clinical_csv: Path, patient_col: str, class_col: str) -> dict[str, str]:
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")

    mapping: dict[str, str] = {}
    with clinical_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Clinical CSV has no header: {clinical_csv}")

        missing = [c for c in [patient_col, class_col] if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Clinical CSV missing required columns: {missing}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            patient = str(row.get(patient_col, "")).strip()
            cls = str(row.get(class_col, "")).strip()
            if not patient:
                continue
            if patient not in mapping and cls:
                mapping[patient] = cls

    if not mapping:
        raise ValueError(f"No valid patient rows found in clinical CSV: {clinical_csv}")
    return mapping


def get_patch_paths(payload: dict, file_path: Path) -> list[str]:
    if "patch_paths" in payload:
        return [str(x) for x in payload["patch_paths"]]
    if "paths" in payload:
        return [str(x) for x in payload["paths"]]
    raise KeyError(f"Missing 'patch_paths' or 'paths' in activation file: {file_path}")


def format_class(class_col: str, class_raw: str | None) -> str:
    if class_raw is None or str(class_raw).strip() == "":
        return "unknown"
    return f"{class_col}_{str(class_raw).strip()}"


def main() -> None:
    args = parse_args()

    cfg = SAEConfig()
    cfg.data_dir = args.data_dir
    files = resolve_activation_files_for_split(cfg, split=args.split, max_files=args.max_files)

    clinical_map = load_clinical_map(
        clinical_csv=Path(args.clinical_csv),
        patient_col=args.patient_col,
        class_col=args.class_col,
    )

    meta_records: dict[str, tuple[str, str]] = {}
    stats = MetaBuildStats()

    iterator = tqdm(files, desc="Build meta", unit="file")
    for file_path in iterator:
        payload = torch.load(file_path, map_location="cpu")
        patch_paths = get_patch_paths(payload, file_path)

        for raw_path in patch_paths:
            stats.total_seen_rows += 1
            path_norm = normalize_path(raw_path)

            patient = Path(path_norm).parent.name
            cls_raw = clinical_map.get(patient)
            cls = format_class(args.class_col, cls_raw)

            if cls_raw is not None:
                stats.clinical_matched += 1
            else:
                stats.unknown_class += 1

            if path_norm in meta_records:
                stats.duplicate_paths += 1
                prev_cls, prev_patient = meta_records[path_norm]
                if prev_cls != cls or prev_patient != patient:
                    stats.conflicting_duplicates += 1
                continue

            meta_records[path_norm] = (cls, patient)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "patient"])
        for path_norm in sorted(meta_records.keys()):
            cls, patient = meta_records[path_norm]
            writer.writerow([path_norm, cls, patient])

    stats.unique_paths = len(meta_records)

    classes = sorted({v[0] for v in meta_records.values()})
    known_classes = [c for c in classes if c != "unknown"]
    match_rate = (stats.clinical_matched / max(1, stats.total_seen_rows)) * 100.0

    print("==================================================")
    print("Meta build completed")
    print("==================================================")
    print(f"Output CSV              : {output_csv}")
    print(f"Split / files           : {args.split} / {len(files)}")
    print(f"Rows seen               : {stats.total_seen_rows}")
    print(f"Unique paths            : {stats.unique_paths}")
    print(f"Clinical matched        : {stats.clinical_matched}")
    print(f"Unknown class           : {stats.unknown_class}")
    print(f"Patient matching rate   : {match_rate:.2f}%")
    print(f"Duplicate paths         : {stats.duplicate_paths}")
    print(f"Conflicting duplicates  : {stats.conflicting_duplicates}")
    print(f"Class count (known)     : {len(known_classes)}")
    print(f"Classes (known)         : {known_classes[:10]}")
    if len(known_classes) < 2:
        print("WARNING: known class count < 2; class-wise analysis may be weak.")
    if match_rate < 95.0:
        print("WARNING: patient matching rate is below 95%.")
    print("==================================================")


if __name__ == "__main__":
    main()
