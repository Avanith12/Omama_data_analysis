#!/usr/bin/env python3
"""
HQ 4-view curated subset 
Writes local pkl + csv manifests
"""

import random
import pickle
import time
from pathlib import Path

import pandas as pd
import pydicom

NEED = {"l-cc", "r-cc", "l-mlo", "r-mlo"}
OUT_DIR = Path(__file__).resolve().parent
OUT_PKL = OUT_DIR / "hq_4view_manifest.pkl"
OUT_CSV = OUT_DIR / "hq_4view_manifest.csv"

PAIRS = [
    (
        "/raid/data01/deephealth/dh_dcm_ast",
        "/raid/data01/deephealth/labels/dh_dcm_ast_labels.csv",
    ),
    (
        "/raid/data01/deephealth/dh_dh0new",
        "/raid/data01/deephealth/labels/dh_dh0new_labels.csv",
    ),
    (
        "/raid/data01/deephealth/dh_dh2",
        "/raid/data01/deephealth/labels/dh_dh2_labels.csv",
    ),
]


# ---------- helpers ----------
def get_view(ds):
    v = str(ds.get("ViewPosition") or "").lower()
    seq = getattr(ds, "ViewCodeSequence", None) or []
    if not v and len(seq) > 0:
        v = str(seq[0].CodeMeaning).lower()
    if "cc" in v or "cranio" in v:
        return "cc"
    if "mlo" in v or "oblique" in v or "medio" in v:
        return "mlo"
    return ""


def read_header(path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def log_loss(step, before, after):
    print(f"  {step}: {before} -> {after}  (lost {before - after})")


def progress(tag, i, total, t0, kept):
    if i % 1000 and i != total:
        return
    elapsed = time.time() - t0
    rate = i / elapsed if elapsed else 0
    eta = (total - i) / rate if rate else 0
    print(
        f"  [{tag}] {i}/{total}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  kept={kept}",
        flush=True,
    )


# ---------- step 1: get study ids from csv ----------
def get_study_ids(label_csv, cancer=True):
    df = pd.read_csv(label_csv)
    if cancer:
        mask = df["Label"].isin(["IndexCancer", "PreIndexCancer"])
    else:
        mask = df["Label"] == "NonCancer"
    return set(df.loc[mask, "StudyInstanceUID"].astype(str))


# ---------- step 2: find studies with all 4 views ----------
def get_four_view_studies(folder, study_ids, tag=""):
    result = {}
    ids = list(study_ids)
    t0 = time.time()
    for i, sid in enumerate(ids, 1):
        views = {}
        for f in Path(folder, sid).glob("DXm.*"):
            ds = read_header(f)
            if ds is None:
                continue
            lat = str(ds.get("ImageLaterality", "")).lower()
            key = f"{lat}-{get_view(ds)}"
            if key in NEED and key not in views:
                views[key] = str(f)
        if NEED <= set(views):
            result[sid] = views
        progress(tag, i, len(ids), t0, len(result))
    return result


# ---------- step 3: filter size + scanner (drop whole study if fail) ----------
def study_ok(paths):
    for p in paths.values():
        ds = read_header(p)
        if ds is None:
            return False
        if int(ds.get("Rows") or 0) < 1024 or int(ds.get("Columns") or 0) < 1024:
            return False
        manu = str(ds.get("Manufacturer") or "").upper()
        if ("GE" not in manu) and ("HOLOGIC" not in manu) and ("LORAD" not in manu):
            return False
    return True


def filter_studies(four_view_dict, tag=""):
    result = {}
    items = list(four_view_dict.items())
    t0 = time.time()
    for i, (sid, paths) in enumerate(items, 1):
        if study_ok(paths):
            result[sid] = paths
        progress(tag, i, len(items), t0, len(result))
    return result


# ---------- step 4: balance (all cancer, match non-cancer) ----------
def balance(cancer_dict, noncancer_dict):
    n = len(cancer_dict)
    if n > len(noncancer_dict):
        raise ValueError(
            f"Not enough noncancer studies: cancer={n}, noncancer={len(noncancer_dict)}"
        )
    return list(cancer_dict), random.sample(list(noncancer_dict), n)


# ---------- step 5: save pkl + csv ----------
def save_manifest(cancer_ids, noncancer_ids, cancer_dict, noncancer_dict, out_pkl, out_csv):
    manifest = []
    rows = []
    for sid, label, d in (
        [(s, "cancer", cancer_dict) for s in cancer_ids]
        + [(s, "noncancer", noncancer_dict) for s in noncancer_ids]
    ):
        paths = d[sid]
        manifest.append({"study_id": sid, "label": label, "paths": paths})
        rows.append(
            {
                "study_id": sid,
                "label": label,
                "l-cc": paths["l-cc"],
                "r-cc": paths["r-cc"],
                "l-mlo": paths["l-mlo"],
                "r-mlo": paths["r-mlo"],
            }
        )
    with open(out_pkl, "wb") as f:
        pickle.dump(manifest, f)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return manifest


# ---------- main ----------
def main():
    t_all = time.time()
    cancer_clean = {}
    noncancer_clean = {}

    for folder, csv in PAIRS:
        name = Path(folder).name
        print("\n=== folder:", name, "===")
        t0 = time.time()

        c_ids = get_study_ids(csv, cancer=True)
        n_ids = get_study_ids(csv, cancer=False)
        print(f"  csv cancer ids: {len(c_ids)}")
        print(f"  csv noncancer ids: {len(n_ids)}")

        c4 = get_four_view_studies(folder, c_ids, tag=f"{name}/cancer")
        log_loss("cancer keep 4-view", len(c_ids), len(c4))

        n4 = get_four_view_studies(folder, n_ids, tag=f"{name}/noncancer")
        log_loss("noncancer keep 4-view", len(n_ids), len(n4))

        c_clean = filter_studies(c4, tag=f"{name}/cancer-filter")
        log_loss("cancer size+scanner", len(c4), len(c_clean))

        n_clean = filter_studies(n4, tag=f"{name}/noncancer-filter")
        log_loss("noncancer size+scanner", len(n4), len(n_clean))

        print(f"  folder time: {(time.time() - t0)/60:.1f}m")
        cancer_clean.update(c_clean)
        noncancer_clean.update(n_clean)

    print("\n=== TOTALS ===")
    print("cancer clean:", len(cancer_clean))
    print("noncancer clean:", len(noncancer_clean))

    final_c, final_nc = balance(cancer_clean, noncancer_clean)
    print(f"balance: keep all cancer={len(final_c)}, sample noncancer={len(final_nc)}")

    manifest = save_manifest(
        final_c, final_nc, cancer_clean, noncancer_clean, OUT_PKL, OUT_CSV
    )
    print("saved:", OUT_PKL)
    print("saved:", OUT_CSV, "total studies:", len(manifest))
    print(f"TOTAL time: {(time.time() - t_all)/60:.1f}m")


if __name__ == "__main__":
    main()
