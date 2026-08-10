#!/usr/bin/env python3
"""
HQ study-level DICOM → NPZ + JSON (original resolution).

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

VIEWS = ["l-cc", "r-cc", "l-mlo", "r-mlo"]

DEFAULT_MANIFEST = Path(
    "/home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/"
    "manifests/csv/final_curated_list.csv"
)
DEFAULT_OUT = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_Data/original_study_level"
)


def convert_to_serializable(value):
    """DICOM MultiValue → plain list for json.dump."""
    if isinstance(value, pydicom.multival.MultiValue):
        return list(value)
    return value


def extract_dicom_metadata(ds):
    """Same 2D fields as Omama process_dicoms.py (no coords/score)."""
    view = (
        getattr(ds.ViewCodeSequence[0], "CodeMeaning", "Unknown")
        if hasattr(ds, "ViewCodeSequence") and len(ds.ViewCodeSequence) > 0
        else "Unknown"
    )
    return {
        "PatientID": convert_to_serializable(getattr(ds, "PatientID", "Unknown")),
        "View": convert_to_serializable(view),
        "WindowCenter": convert_to_serializable(getattr(ds, "WindowCenter", "Unknown")),
        "WindowWidth": convert_to_serializable(getattr(ds, "WindowWidth", "Unknown")),
        "WindowCenterWidthExplanation": convert_to_serializable(
            getattr(ds, "WindowCenterWidthExplanation", "Unknown")
        ),
        "ImagerPixelSpacing": convert_to_serializable(
            getattr(ds, "ImagerPixelSpacing", "Unknown")
        ),
        "ImageLaterality": convert_to_serializable(
            getattr(ds, "ImageLaterality", "Unknown")
        ),
    }


def uid_basename(dicom_path):
    """DXm.2.25.xxx → 2.25.xxx"""
    name = Path(dicom_path).name
    if name.startswith("DXm."):
        return name[4:]
    return name


def load_jobs(manifest, limit=0):
    """final_curated_list.csv → list of (study_id, dicom_path, label)."""
    df = pd.read_csv(manifest)
    if limit > 0:
        df = df.head(limit)

    jobs = []
    for _, row in df.iterrows():
        study_id = str(row["study_id"])
        label = "IndexCancer" if row["label"] == "cancer" else "NonCancer"
        for view in VIEWS:
            jobs.append((study_id, str(row[view]), label))
    return jobs


def worker(study_id, dicom_path, label, images_root, metadata_root):
    """
    One DICOM → one NPZ + one JSON under study folders.
    Returns: "ok" | "skip" | "fail"
    """
    basename = uid_basename(dicom_path)

    img_dir = images_root / study_id
    meta_dir = metadata_root / study_id
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    npz_path = img_dir / f"{basename}.npz"
    json_path = meta_dir / f"{basename}.json"

    if npz_path.exists() and json_path.exists():
        return "skip"

    try:
        ds = pydicom.dcmread(dicom_path, force=True)

        if not hasattr(ds, "PixelData"):
            print(f"{dicom_path} has no PixelData. Skipping.")
            return "fail"

        # original resolution — same as Omama process_2d_image (identity)
        np.savez_compressed(npz_path, data=ds.pixel_array)

        metadata = extract_dicom_metadata(ds)
        metadata["label"] = label

        with open(json_path, "w") as f:
            json.dump(metadata, f)

        return "ok"
    except Exception as e:
        print(f"FAIL {dicom_path}: {e}")
        return "fail"


def run(manifest, out_dir, limit=0):
    images_root = out_dir / "images"
    metadata_root = out_dir / "metadata"
    images_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(manifest, limit)
    print(f"total images: {len(jobs)}  studies: {len(jobs) // 4}")
    print(f"output: {out_dir}")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for i, (study_id, dicom_path, label) in enumerate(jobs, 1):
        status = worker(study_id, dicom_path, label, images_root, metadata_root)
        counts[status] += 1
        if i % 50 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] {counts}", flush=True)

    print("done", counts)


def main():
    parser = argparse.ArgumentParser(
        description="Process HQ DICOMs → study-level NPZ + JSON (original resolution)."
    )
    parser.add_argument("-i", "--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max studies (0=all). e.g. 5 for smoke test",
    )
    args = parser.parse_args()
    run(args.manifest, args.output, args.limit)


if __name__ == "__main__":
    main()
