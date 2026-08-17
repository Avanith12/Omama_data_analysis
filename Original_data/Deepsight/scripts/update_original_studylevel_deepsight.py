#!/usr/bin/env python3
"""Original study-level: add coords/score, drop incomplete studies, check vs v2 CSV."""

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/original_studylevel"
)
CACHE = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/"
    "deepsight/predictions_cache_merged.json"
)
CSV = Path(
    "/home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/"
    "deepsight/manifests/final_curated_list_v2_deepsight.csv"
)

merged = json.loads(CACHE.read_text())
keep = {row["study_id"] for row in csv.DictReader(open(CSV))}

# 1) add coords + score only when DeepSight produced a box
n_write = 0
for jp in (ROOT / "metadata").rglob("*.json"):
    pred = merged.get(jp.stem, {})
    if not pred.get("coords"):
        continue
    meta = json.loads(jp.read_text())
    meta["coords"] = pred["coords"]
    meta["score"] = pred["score"]
    jp.write_text(json.dumps(meta))
    n_write += 1
print("updated jsons", n_write)

# 2) delete studies not in the v2 csv
n_del = 0
for study_dir in list((ROOT / "metadata").iterdir()):
    if not study_dir.is_dir() or study_dir.name in keep:
        continue
    shutil.rmtree(study_dir)
    img = ROOT / "images" / study_dir.name
    if img.exists():
        shutil.rmtree(img)
    n_del += 1
print("deleted studies", n_del)

# 3) check original folders match v2 csv
meta_ids = {p.name for p in (ROOT / "metadata").iterdir() if p.is_dir()}
img_ids = {p.name for p in (ROOT / "images").iterdir() if p.is_dir()}
missing_fields = 0
bad_count = 0
for sid in keep:
    jsons = list((ROOT / "metadata" / sid).glob("*.json"))
    npzs = list((ROOT / "images" / sid).glob("*.npz"))
    if len(jsons) != 4 or len(npzs) != 4:
        bad_count += 1
        continue
    for jp in jsons:
        meta = json.loads(jp.read_text())
        if not meta.get("coords") or "score" not in meta:
            missing_fields += 1

ok = meta_ids == keep and img_ids == keep and missing_fields == 0 and bad_count == 0
print("v2 studies", len(keep), "folders", len(meta_ids))
print("jsons missing coords/score", missing_fields, "studies not 4+4", bad_count)
print("OK" if ok else "FAIL")
