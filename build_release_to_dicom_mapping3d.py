#!/usr/bin/env python3
"""
Map 3D release ID -> /raid DICOM path.

Input:  /raid/mpsych/OMAMA/DATA/whitelists/3D_whitelist_final.txt
Rule:   BT.2.25.<id>  becomes  2.25.<id>  (same name as mpsych JSON / NPZ)

Outputs (in this folder):
  release_to_dicom_mapping_3d.pkl
  release_to_dicom_mapping_3d.csv

Run:
  python build_release_to_dicom_mapping3d.py
  python build_release_to_dicom_mapping3d.py --verify-metadata --verify-raid
  python build_release_to_dicom_mapping3d.py --limit 100
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHITELIST = Path("/raid/mpsych/OMAMA/DATA/whitelists/3D_whitelist_final.txt")
METADATA_DIR = Path("/raid/mpsych/OMAMA/DATA/data/3d/metadata")
OUT_PKL = HERE / "release_to_dicom_mapping_3d.pkl"
OUT_CSV = HERE / "release_to_dicom_mapping_3d.csv"


def path_to_release_id(dicom_path):
    """BT.2.25.xxx -> 2.25.xxx"""
    name = dicom_path.strip().split("/")[-1]
    if name.startswith("BT."):
        return name[3:]
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-list", type=Path, default=WHITELIST)
    parser.add_argument("--verify-metadata", action="store_true")
    parser.add_argument("--verify-raid", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.dicom_list.exists():
        sys.exit(f"Missing file: {args.dicom_list}")

    # --- 1. Read whitelist, build dict ---
    lines = [ln.strip() for ln in args.dicom_list.read_text().splitlines() if ln.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]

    mapping = {}
    for path in lines:
        release_id = path_to_release_id(path)
        if release_id in mapping:
            print(f"ERROR: duplicate ID {release_id}", file=sys.stderr)
            print(f"  {mapping[release_id]}", file=sys.stderr)
            print(f"  {path}", file=sys.stderr)
            sys.exit(1)
        mapping[release_id] = path

    print(f"Built mapping: {len(mapping):,} entries")

    # --- 2. Optional checks ---
    if args.verify_metadata:
        if not METADATA_DIR.is_dir():
            print(f"WARNING: metadata folder not found: {METADATA_DIR}")
        else:
            json_ids = {p.stem for p in METADATA_DIR.glob("*.json")}
            not_in_mapping = json_ids - set(mapping)
            not_in_metadata = set(mapping) - json_ids
            print(f"Metadata JSON not in mapping: {len(not_in_mapping)}")
            if args.limit == 0:
                print(f"Mapping IDs not in metadata: {len(not_in_metadata)}")

    if args.verify_raid:
        paths = list(mapping.values())
        if args.limit > 0:
            paths = paths[: args.limit]
        missing = [p for p in paths if not os.path.isfile(p)]
        print(f"Missing on raid: {len(missing):,} / {len(paths):,}")
        if missing:
            print(f"Example missing: {missing[0]}")

    # --- 3. Save pkl + csv ---
    with open(OUT_PKL, "wb") as f:
        pickle.dump(mapping, f)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["release_id", "dicom_raid_path"])
        for release_id in sorted(mapping):
            writer.writerow([release_id, mapping[release_id]])

    print(f"Saved: {OUT_PKL}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
