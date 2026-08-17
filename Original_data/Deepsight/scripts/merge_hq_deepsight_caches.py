#!/usr/bin/env python3
"""Merge HQ DeepSight per-task caches into one file."""

import json
from pathlib import Path

# folder with predictions_cache_task1.json ... task12.json
CACHE_DIR = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/deepsight"
)

# one dict: SOP -> {coords, score, errors}
merged = {}
for p in sorted(CACHE_DIR.glob("predictions_cache_task*.json")):
    merged.update(json.loads(p.read_text() or "{}"))

# write the combined cache
out = CACHE_DIR / "predictions_cache_merged.json"
out.write_text(json.dumps(merged))
print(len(merged), "->", out)
