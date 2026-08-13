#!/usr/bin/env python3
"""Run DeepSight on one SLURM array slice of the HQ caselist (study-complete)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/a.kanamarlapudi001/projects/omama-proj")
import omama as O

CASELIST = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/"
    "deepsight/hq_caselist.txt"
)
OUT_DIR = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/deepsight/out_v2"
)
CACHE_DIR = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/deepsight"
)
N_TASKS = 12
PATHS_PER_STUDY = 4


def main(task_id: int):
    paths = [ln.strip() for ln in CASELIST.read_text().splitlines() if ln.strip()]
    n_studies = len(paths) // PATHS_PER_STUDY
    per = (n_studies + N_TASKS - 1) // N_TASKS
    start = (task_id - 1) * per
    end = min(start + per, n_studies)
    subset = paths[start * PATHS_PER_STUDY : end * PATHS_PER_STUDY]

    cache_path = CACHE_DIR / f"predictions_cache_task{task_id}.json"
    if not cache_path.exists():
        cache_path.write_text("{}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"task={task_id}/{N_TASKS} studies={start}:{end} "
        f"({end - start} studies, {len(subset)} paths)"
    )
    if not subset:
        print("empty slice, nothing to do")
        return

    pred = O.DeepSight.run(
        subset,
        output_dir=str(OUT_DIR) + "/",
        pred_cache_path=str(cache_path),
        task_num=str(task_id),
        timing=True,
    )
    print(f"task={task_id} predictions={len(pred)} cache={cache_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <SLURM_ARRAY_TASK_ID>")
        sys.exit(1)
    main(int(sys.argv[1]))
