#!/usr/bin/env python3
"""
Build a clean map: release ID (Dataverse / scratch name) -> /raid DICOM path.

The release ID is the numeric name shared by:
  - metadata/<id>.json
  - images/<id>.npz
  - DICOM file .../DXm.2.25.<id>   (strip "DXm.2.25." from the whitelist basename)

Inputs: omama/loaders/final_2d_dataset.txt (one /raid path per line).
Outputs:
  - release_to_dicom_mapping.pkl   dict: release_id -> dicom_path
  - release_to_dicom_mapping.csv   two columns only

Read-only on /raid (only reads path strings from the list file in the repo).

Run:
  python build_release_to_dicom_mapping.py
  python build_release_to_dicom_mapping.py --verify-scratch
  python build_release_to_dicom_mapping.py --limit 100
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it

# ---------- paths ----------
HERE = Path(__file__).resolve().parent
FINAL_2D_LIST = HERE.parent.parent / "omama" / "loaders" / "final_2d_dataset.txt"
SCRATCH_METADATA_DIR = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/2d_resized_512/metadata"
)
DICOM_ID_PREFIX = "DXm.2.25."

OUT_MAPPING_PKL = HERE / "release_to_dicom_mapping.pkl"
OUT_MAPPING_CSV = HERE / "release_to_dicom_mapping.csv"


def dicom_path_to_release_id(dicom_path):
    # type: (str) -> str
    """.../DXm.2.25.100002819... -> 100002819... (release / Dataverse name)."""
    basename = dicom_path.strip().split("/")[-1]
    if basename.startswith(DICOM_ID_PREFIX):
        return basename[len(DICOM_ID_PREFIX) :]
    return basename


def build_mapping_from_list(list_file, limit=0):
    # type: (Path, int) -> Dict[str, str]
    """
    release_id -> full /raid DICOM path.
    Update final_2d_dataset.txt (or --dicom-list) to refresh.
    """
    lines = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
    if limit > 0:
        lines = lines[:limit]

    mapping = {}  # type: Dict[str, str]
    duplicates = []  # type: List[Tuple[str, str, str]]

    for path in lines:
        if not path.startswith("/raid/"):
            raise ValueError(f"Path not under /raid/: {path}")

        release_id = dicom_path_to_release_id(path)
        if release_id in mapping:
            duplicates.append((release_id, mapping[release_id], path))
            continue
        mapping[release_id] = path

    if duplicates:
        print(f"ERROR: {len(duplicates)} duplicate release IDs.", file=sys.stderr)
        for rid, p1, p2 in duplicates[:5]:
            print(f"  {rid}:\n    {p1}\n    {p2}", file=sys.stderr)
        sys.exit(1)

    return mapping


def verify_scratch_metadata(mapping, metadata_dir, limit=0):
    # type: (Dict[str, str], Path, int) -> Tuple[bool, List[str]]
    """Optional: every scratch JSON stem exists in the mapping."""
    issues = []  # type: List[str]
    if not metadata_dir.is_dir():
        issues.append(f"Missing: {metadata_dir}")
        return False, issues

    json_files = sorted(metadata_dir.glob("*.json"))
    if limit > 0:
        json_files = json_files[:limit]

    missing = [p.stem for p in json_files if p.stem not in mapping]
    extra = set(mapping) - {p.stem for p in json_files}

    if missing:
        issues.append(f"{len(missing)} scratch JSON IDs not in whitelist mapping")
    if extra and limit == 0:
        issues.append(f"{len(extra)} whitelist IDs not in scratch metadata")

    return len(issues) == 0, issues


def save_results(mapping):
    # type: (Dict[str, str]) -> None
    with open(OUT_MAPPING_PKL, "wb") as f:
        pickle.dump(mapping, f)

    with open(OUT_MAPPING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release_id", "dicom_raid_path"])
        w.writeheader()
        for release_id in sorted(mapping):
            w.writerow(
                {"release_id": release_id, "dicom_raid_path": mapping[release_id]}
            )

    print(f"\nSaved: {OUT_MAPPING_PKL}")
    print(f"Saved: {OUT_MAPPING_CSV}")


def print_summary(mapping):
    # type: (Dict[str, str]) -> None
    print("\n=== Summary ===")
    print(f"Mapped: {len(mapping):,}")
    print("Columns: release_id -> dicom_raid_path")
    print("(release_id = Dataverse JSON / .npz name without extension)")


def main():
    parser = argparse.ArgumentParser(
        description="Map release ID to /raid DICOM path (from final_2d_dataset.txt)"
    )
    parser.add_argument(
        "--dicom-list",
        type=Path,
        default=FINAL_2D_LIST,
        help="Whitelist: one /raid DICOM path per line",
    )
    parser.add_argument(
        "--verify-scratch",
        action="store_true",
        help="Check scratch metadata/ IDs match the mapping",
    )
    parser.add_argument("--limit", type=int, default=0, help="Test on first N lines only")
    args = parser.parse_args()

    if not args.dicom_list.exists():
        sys.exit(f"Missing DICOM list: {args.dicom_list}")

    print("=== Build release_id -> /raid DICOM path ===")
    print(f"  List: {args.dicom_list}")
    mapping = build_mapping_from_list(args.dicom_list, limit=args.limit)
    print(f"  {len(mapping):,} entries (strip {DICOM_ID_PREFIX!r})")

    if args.verify_scratch:
        print("\n=== Verify scratch metadata (optional) ===")
        ok, issues = verify_scratch_metadata(
            mapping, SCRATCH_METADATA_DIR, limit=args.limit
        )
        if ok:
            print("  Scratch JSON IDs match mapping.")
        else:
            for msg in issues:
                print(f"  WARNING: {msg}")

    print("\n=== Save ===")
    save_results(mapping)
    print_summary(mapping)


if __name__ == "__main__":
    main()
