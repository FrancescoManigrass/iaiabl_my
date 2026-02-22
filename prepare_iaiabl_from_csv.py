#!/usr/bin/env python3
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


CLASSES = ("Circumscribed", "Indistinct", "Spiculated")
SPLITS = ("train", "test", "validation", "push", "finer")


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
    p = argparse.ArgumentParser(
        description="Convert CSV+PNG patches to IAIA-BL ready class folders with uint16 .npy files."
    )
    p.add_argument("--csv_path", required=True)
    p.add_argument("--images_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    p.add_argument("--val_as", choices=("test", "train", "validation", "skip"), default="test")
    p.add_argument("--allow_leakage", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_rows", type=int, default=None)

    # NEW: cache control
    p.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Force regeneration of _npy_cache even if files already exist.",
    )
    return p.parse_args()


def norm_text(v: object) -> str:
    return "" if v is None else str(v).strip()


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
    if s in {"train", "test"}:
        return s
    if s in {"val", "valid", "validation"}:
        return None if val_as == "skip" else val_as
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

    excluded = [
        parse_intish(row.get("mass_margin_OBSCURED")) == 1,
        parse_intish(row.get("mass_margin_MICROLOBULATED")) == 1,
    ]
    if any(excluded):
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


def extract_patient_id_from_patch_filename(patch_filename: str) -> Optional[str]:
    stem = Path(patch_filename).name
    m = re.match(r"^(P_\d+)(?:_|$)", stem)
    return m.group(1) if m else None


def ensure_layout(out_root: Path, dry_run: bool) -> None:
    for split in SPLITS:
        for cls in CLASSES:
            d = out_root / split / cls
            if not dry_run:
                d.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        (out_root / "_npy_cache").mkdir(parents=True, exist_ok=True)


def png_to_uint16_npy(source_png: Path) -> np.ndarray:
    """
    Read PNG patch preserving 16-bit grayscale when present.
    Output: np.uint16 array in [0, 65535].
    """
    with Image.open(source_png) as img:
        # If it is already 16-bit grayscale, keep it
        if img.mode in ("I;16", "I;16B", "I;16L"):
            arr = np.array(img)
            # Ensure dtype exactly uint16
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16)
            return arr

        # 8-bit grayscale -> upscale to 16-bit
        if img.mode == "L":
            arr8 = np.array(img, dtype=np.uint8)
            return (arr8.astype(np.uint16) * 257)  # 0..255 -> 0..65535

        # Other (RGB/RGBA/etc.) -> convert to 8-bit gray then upscale
        gray8 = img.convert("L")
        arr8 = np.array(gray8, dtype=np.uint8)
        return (arr8.astype(np.uint16) * 257)


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
        f"validation_dir={out_root / 'validation'}\n"
    )
    if dry_run:
        print("\n[DRY RUN] paths_for_train_sh.txt would contain:\n" + content)
        return
    (out_root / "paths_for_train_sh.txt").write_text(content, encoding="utf-8")


def normalize_out_root(out_root_arg: str) -> Path:
    text = norm_text(out_root_arg)
    return Path(text.split()[0]) if text else Path(text)


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv_path)
    images_root = Path(args.images_root)
    out_root = normalize_out_root(args.out_root)

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
    eval_patients = set()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "patch_filename",
            "split",
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
            if not patch_filename:
                skipped["missing_patch_filename"] += 1
                continue

            patient_id = norm_text(row.get("patient_id"))
            if not patient_id:
                patient_id = extract_patient_id_from_patch_filename(patch_filename) or "unknown_patient"

            if not is_mass(row):
                skipped["non_mass"] += 1
                continue

            class_name, class_reason = determine_class(row)
            if class_name is None:
                skipped[class_reason] += 1
                if class_reason == "inconsistent_margin_flags":
                    warnings.append(f"Row {idx}: inconsistent margin flags for '{patch_filename}'")
                continue

            split = map_split(norm_text(row.get("split")), args.val_as)
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
            elif split in {"test", "validation"}:
                eval_patients.add(patient_id)

    leakage = sorted(train_patients.intersection(eval_patients))
    if leakage and not args.allow_leakage:
        print("ERROR: patient leakage between train and eval splits.", file=sys.stderr)
        for pid in leakage[:50]:
            print(f"  - {pid}", file=sys.stderr)
        print("Use --allow_leakage to continue.", file=sys.stderr)
        return 1

    usage_counts = defaultdict(Counter)

    # --- MAIN WRITE LOOP ---
    for row in selected:
        cache_npy = out_root / "_npy_cache" / f"{row.unique_stem}.npy"

        if not args.dry_run:
            need_regen = args.overwrite_cache or (not cache_npy.exists())
            if (not need_regen) and cache_npy.exists():
                # If cache exists but is not uint16, regenerate
                try:
                    prev = np.load(cache_npy, mmap_mode="r")
                    if prev.dtype != np.uint16:
                        need_regen = True
                except Exception:
                    need_regen = True

            if need_regen:
                arr16 = png_to_uint16_npy(row.source_png)
                # enforce exact dtype
                if arr16.dtype != np.uint16:
                    arr16 = arr16.astype(np.uint16)
                np.save(cache_npy, arr16)

        split_target = out_root / row.split / row.class_name / f"{row.unique_stem}.npy"
        write_or_link(cache_npy, split_target, args.mode, args.dry_run)
        usage_counts[row.split][row.class_name] += 1

        if row.split == "train":
            push_target = out_root / "push" / row.class_name / f"{row.unique_stem}.npy"
            write_or_link(cache_npy, push_target, args.mode, args.dry_run)
            usage_counts["push"][row.class_name] += 1

    write_paths_file(out_root, args.dry_run)

    print("\n=== IAIA-BL dataset preparation report (uint16) ===")
    print(f"Rows selected: {len(selected)}")
    print("Skipped rows by reason:")
    for reason, count in sorted(skipped.items()):
        print(f"  - {reason}: {count}")

    print("\nUsed samples by split/class:")
    for split in ("train", "test", "validation", "push"):
        print(f"  {split}:")
        for cls in CLASSES:
            print(f"    - {cls}: {usage_counts[split][cls]}")

    if leakage and args.allow_leakage:
        print(f"\nWARNING: leakage allowed; overlapping patient_ids: {len(leakage)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())