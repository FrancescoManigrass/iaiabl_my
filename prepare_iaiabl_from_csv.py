#!/usr/bin/env python3
"""Prepare IAIA-BL dataset folders (.npy by class/split) from a CSV and PNG patches.

This script builds the directory layout expected by IAIA-BL training scripts:

    out_root/
      train/{Circumscribed,Indistinct,Spiculated}
      test/{Circumscribed,Indistinct,Spiculated}
      push/{Circumscribed,Indistinct,Spiculated}
      finer/{Circumscribed,Indistinct,Spiculated}

Input rows are filtered to replicate the IAIA-BL mass-margin setting:
- only masses
- only margins in {CIRCUMSCRIBED, INDISTINCT, SPICULATED}
- rows with inconsistent one-hot margin flags are skipped

PNG patches are converted to grayscale .npy arrays. The repository's dataloader uses
np.load and skimage.resize, then Image.fromarray for augmentation. To remain compatible,
we save uint8 arrays in [0,255] (single channel).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


CLASSES = ("Circumscribed", "Indistinct", "Spiculated")
SPLITS = ("train", "test", "push", "finer")


@dataclass
class SelectedRow:
    csv_index: int
    patient_id: str
    split: str
    class_name: str
    patch_filename: str
    source_png: Path
    unique_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV+PNG patches to IAIA-BL ready class folders with .npy files."
    )
    parser.add_argument("--csv_path", required=True, help="Path to input CSV.")
    parser.add_argument(
        "--images_root",
        required=True,
        help="Root folder to prepend to patch_filename for reading PNG patches.",
    )
    parser.add_argument("--out_root", required=True, help="Output dataset root.")
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="copy: place real .npy files in split folders; symlink: link from _npy_cache.",
    )
    parser.add_argument(
        "--val_as",
        choices=("test", "train", "skip"),
        default="test",
        help="How to map CSV split=Val.",
    )
    parser.add_argument(
        "--allow_leakage",
        action="store_true",
        help="Allow patient leakage between train and test (otherwise fail).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print report; do not create files.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional limit on processed CSV rows (debug).",
    )
    return parser.parse_args()


def norm_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_intish(value: object) -> Optional[int]:
    text = norm_text(value)
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "t", "yes", "y"}:
        return 1
    if lowered in {"false", "f", "no", "n"}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return None


def is_mass(row: Dict[str, str]) -> bool:
    has_mass = parse_intish(row.get("has_mass"))
    lesion_type = norm_text(row.get("type_of_lesion")).lower()
    return (has_mass == 1) or (lesion_type == "mass")


def map_split(raw_split: str, val_as: str) -> Optional[str]:
    s = norm_text(raw_split).lower()
    if s == "train":
        return "train"
    if s == "test":
        return "test"
    if s in {"val", "valid", "validation"}:
        if val_as == "skip":
            return None
        return val_as
    return None


def determine_class(row: Dict[str, str]) -> Tuple[Optional[str], str]:
    flags = {
        "Circumscribed": parse_intish(row.get("mass_margin_CIRCUMSCRIBED")) == 1,
        "Indistinct": parse_intish(row.get("mass_margin_INDISTINCT")) == 1,
        "Spiculated": parse_intish(row.get("mass_margin_SPICULATED")) == 1,
    }
    positives = [k for k, v in flags.items() if v]
    if len(positives) != 1:
        return None, "inconsistent_margin_flags"

    # Exclude margins outside the 3-target setting (if explicitly labeled).
    excluded_margin_flags = [
        parse_intish(row.get("mass_margin_OBSCURED")) == 1,
        parse_intish(row.get("mass_margin_MICROLOBULATED")) == 1,
    ]
    if any(excluded_margin_flags):
        return None, "excluded_margin"

    return positives[0], "ok"


def sanitize_stem(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "sample"


def build_unique_stem(patient_id: str, patch_filename: str) -> str:
    patch_stem = Path(patch_filename).stem
    base = f"{patient_id}__{patch_stem}" if patient_id else patch_stem
    base = sanitize_stem(base)
    digest = hashlib.sha1(f"{patient_id}|{patch_filename}".encode("utf-8")).hexdigest()[:10]
    return f"{base}__{digest}"


def ensure_layout(out_root: Path, dry_run: bool) -> None:
    for split in SPLITS:
        for cls in CLASSES:
            target = out_root / split / cls
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        (out_root / "_npy_cache").mkdir(parents=True, exist_ok=True)


def png_to_uint8_npy(source_png: Path) -> np.ndarray:
    with Image.open(source_png) as img:
        gray = img.convert("L")
        arr = np.array(gray)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def write_or_link(src_npy: Path, dst_npy: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst_npy.parent.mkdir(parents=True, exist_ok=True)

    if dst_npy.exists() or dst_npy.is_symlink():
        dst_npy.unlink()

    if mode == "copy":
        shutil.copy2(src_npy, dst_npy)
    else:
        rel = os.path.relpath(src_npy, start=dst_npy.parent)
        os.symlink(rel, dst_npy)


def write_paths_file(out_root: Path, dry_run: bool) -> None:
    content = (
        f"train_dir={out_root / 'train'}\n"
        f"test_dir={out_root / 'test'}\n"
        f"push_dir={out_root / 'push'}\n"
        f"finer_dir={out_root / 'finer'}\n"
    )
    if dry_run:
        print("\n[DRY RUN] paths_for_train_sh.txt would contain:\n" + content)
        return
    (out_root / "paths_for_train_sh.txt").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv_path)
    images_root = Path(args.images_root)
    out_root = Path(args.out_root)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2
    if not images_root.exists():
        print(f"ERROR: images_root not found: {images_root}", file=sys.stderr)
        return 2

    ensure_layout(out_root, args.dry_run)

    selected: List[SelectedRow] = []
    skipped = Counter()
    warnings: List[str] = []
    train_patients = set()
    test_patients = set()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "patch_filename",
            "split",
            "patient_id",
            "type_of_lesion",
            "has_mass",
            "mass_margin_CIRCUMSCRIBED",
            "mass_margin_INDISTINCT",
            "mass_margin_SPICULATED",
            "mass_margin_OBSCURED",
            "mass_margin_MICROLOBULATED",
        }
        missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing:
            print(f"ERROR: CSV missing required columns: {missing}", file=sys.stderr)
            return 2

        for idx, row in enumerate(reader, start=1):
            if args.max_rows is not None and idx > args.max_rows:
                break

            patch_filename = norm_text(row.get("patch_filename"))
            patient_id = norm_text(row.get("patient_id")) or "unknown_patient"
            raw_split = norm_text(row.get("split"))

            if not patch_filename:
                skipped["missing_patch_filename"] += 1
                continue

            if not is_mass(row):
                skipped["non_mass"] += 1
                continue

            class_name, class_reason = determine_class(row)
            if class_name is None:
                skipped[class_reason] += 1
                if class_reason == "inconsistent_margin_flags":
                    warnings.append(
                        f"Row {idx}: inconsistent target margin flags for patch '{patch_filename}'"
                    )
                continue

            split = map_split(raw_split, args.val_as)
            if split is None:
                skipped["unsupported_split"] += 1
                continue

            source_png = images_root / patch_filename
            if not source_png.exists():
                skipped["missing_png"] += 1
                continue

            unique_stem = build_unique_stem(patient_id, patch_filename)
            selected.append(
                SelectedRow(
                    csv_index=idx,
                    patient_id=patient_id,
                    split=split,
                    class_name=class_name,
                    patch_filename=patch_filename,
                    source_png=source_png,
                    unique_stem=unique_stem,
                )
            )

            if split == "train":
                train_patients.add(patient_id)
            elif split == "test":
                test_patients.add(patient_id)

    leakage = sorted(train_patients.intersection(test_patients))
    if leakage and not args.allow_leakage:
        print("ERROR: patient leakage detected between train and test splits.", file=sys.stderr)
        print("Leaking patient_ids (up to 50 shown):", file=sys.stderr)
        for pid in leakage[:50]:
            print(f"  - {pid}", file=sys.stderr)
        print("Use --allow_leakage to continue anyway.", file=sys.stderr)
        return 1

    usage_counts = defaultdict(Counter)
    examples = defaultdict(list)

    for row in selected:
        cache_npy = out_root / "_npy_cache" / f"{row.unique_stem}.npy"

        if not args.dry_run:
            if not cache_npy.exists():
                arr = png_to_uint8_npy(row.source_png)
                np.save(cache_npy, arr)

        split_target = out_root / row.split / row.class_name / f"{row.unique_stem}.npy"
        write_or_link(cache_npy, split_target, args.mode, args.dry_run)
        usage_counts[row.split][row.class_name] += 1
        if len(examples[(row.split, row.class_name)]) < 5:
            examples[(row.split, row.class_name)].append(str(split_target))

        if row.split == "train":
            push_target = out_root / "push" / row.class_name / f"{row.unique_stem}.npy"
            write_or_link(cache_npy, push_target, args.mode, args.dry_run)
            usage_counts["push"][row.class_name] += 1
            if len(examples[("push", row.class_name)]) < 5:
                examples[("push", row.class_name)].append(str(push_target))

    write_paths_file(out_root, args.dry_run)

    print("\n=== IAIA-BL dataset preparation report ===")
    print(f"CSV rows considered: {sum(skipped.values()) + len(selected)}")
    print(f"Rows selected: {len(selected)}")
    print("\nSkipped rows by reason:")
    if skipped:
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}")
    else:
        print("  - none")

    if warnings:
        print("\nWarnings (first 20):")
        for msg in warnings[:20]:
            print(f"  - {msg}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings)-20} more")

    print("\nUsed samples by split/class:")
    for split in ("train", "test", "push"):
        print(f"  {split}:")
        for cls in CLASSES:
            print(f"    - {cls}: {usage_counts[split][cls]}")

    print("\nOutput examples (up to 5 per split/class):")
    for split in ("train", "test", "push"):
        for cls in CLASSES:
            key = (split, cls)
            print(f"  {split}/{cls}:")
            if examples[key]:
                for path in examples[key]:
                    print(f"    - {path}")
            else:
                print("    - (none)")

    if leakage and args.allow_leakage:
        print(f"\nWARNING: leakage allowed; overlapping patient_ids: {len(leakage)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Example usage:
# python prepare_iaiabl_from_csv.py --csv_path validation_set.csv --images_root /data/patches --out_root /data/iaiabl_ready
