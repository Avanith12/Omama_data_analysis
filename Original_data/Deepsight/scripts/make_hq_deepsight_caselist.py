#!/usr/bin/env python3
"""Build HQ DeepSight caselist (one absolute DICOM path per line)."""

import pandas as pd
from pathlib import Path

CSV = Path(
    "/home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/"
    "manifests/csv/final_curated_list.csv"
)
OUT = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/"
    "deepsight/hq_caselist.txt"
)
VIEWS = ["l-cc", "r-cc", "l-mlo", "r-mlo"]

df = pd.read_csv(CSV)
paths = [str(row[v]) for _, row in df.iterrows() for v in VIEWS]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(paths) + "\n")
print(f"{len(df)} studies, {len(paths)} paths -> {OUT}")
