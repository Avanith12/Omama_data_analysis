#!/usr/bin/env python3
"""Drop HQ studies where any of the 4 views is missing DeepSight coords."""

import json
from pathlib import Path

import pandas as pd

CSV_IN = Path(
    "/home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/"
    "manifests/csv/final_curated_list.csv"
)
CSV_OUT = Path(
    "/home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/"
    "deepsight/manifests/final_curated_list_v2_deepsight.csv"
)
CACHE = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/"
    "deepsight/predictions_cache_merged.json"
)
VIEWS = ["l-cc", "r-cc", "l-mlo", "r-mlo"]


def sop_from_path(p):
    # .../DXm.2.25.xxx  ->  2.25.xxx
    name = Path(p).name
    return name[4:] if name.startswith("DXm.") else name


merged = json.loads(CACHE.read_text())
df = pd.read_csv(CSV_IN)


def study_ok(row):
    # keep only if all 4 views have nonempty coords
    for v in VIEWS:
        pred = merged.get(sop_from_path(row[v]), {})
        if not pred.get("coords"):
            return False
    return True


keep = df[df.apply(study_ok, axis=1)]
keep.to_csv(CSV_OUT, index=False)
print(f"in={len(df)} keep={len(keep)} drop={len(df) - len(keep)} -> {CSV_OUT}")
