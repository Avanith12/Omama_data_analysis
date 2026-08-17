#!/usr/bin/env python3
"""Set JSON label from DeepHealth (IndexCancer / PreIndexCancer / NonCancer)
on original + 256 + 512 + 1024 study-level metadata.

Run AFTER all three resize jobs finish.
"""

import csv
import json
from collections import Counter
from pathlib import Path

CSV = Path(
    "/home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/"
    "deepsight/manifests/final_curated_list_v2_deepsight.csv"
)
LABEL_CSVS = [
    Path("/home/a.kanamarlapudi001/projects/omama-proj/_EXPERIMENTS/CS438/labels/dh_dcm_ast_labels.csv"),
    Path("/home/a.kanamarlapudi001/projects/omama-proj/_EXPERIMENTS/CS438/labels/dh_dh0new_labels.csv"),
    Path("/home/a.kanamarlapudi001/projects/omama-proj/_EXPERIMENTS/CS438/labels/dh_dh2_labels.csv"),
]
ROOTS = [
    Path("/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/original_studylevel"),
    Path("/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/256x256_studylevel"),
    Path("/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/512x512_studylevel"),
    Path("/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/1024x1024_studylevel"),
]

# StudyInstanceUID -> IndexCancer / PreIndexCancer / NonCancer
uid_to_label = {}
for p in LABEL_CSVS:
    with open(p) as f:
        for row in csv.DictReader(f):
            uid_to_label[str(row["StudyInstanceUID"])] = row["Label"]

keep = [row["study_id"] for row in csv.DictReader(open(CSV))]

n_write = 0
missing = []
for sid in keep:
    lab = uid_to_label.get(sid)
    if lab is None:
        missing.append(sid)
        continue
    for root in ROOTS:
        mdir = root / "metadata" / sid
        if not mdir.is_dir():
            missing.append(f"{root.name}/{sid}")
            continue
        for jp in mdir.glob("*.json"):
            meta = json.loads(jp.read_text())
            meta["label"] = lab
            tmp = jp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta))
            tmp.replace(jp)
            n_write += 1

print("updated jsons", n_write)
print("missing", len(missing))
if missing[:10]:
    print("examples", missing[:10])
print("OK" if not missing and n_write == len(keep) * 4 * 4 else "CHECK")

# study-level counts (one label per study)
counts = Counter(uid_to_label[s] for s in keep if s in uid_to_label)
print("studies", len(keep), dict(counts))
