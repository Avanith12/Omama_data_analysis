#!/usr/bin/env python
"""Run Mammo-CLIP on full balanced OMAMA dataset."""

import subprocess

BASE_CMD = [
    "python",
    "/home/a.kanamarlapudi001/projects/ml_cv_projects/Mammo-CLIP/src/codebase/train_classifier.py",
    "--data-dir", "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/data",
    "--img-dir", "omama_balanced/images_png",
    "--csv-file", "omama_balanced/csv/omama_trainval.csv",
    "--clip_chk_pt_path", "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/checkpoints/Pre-trained-checkpoints/b5-model-best-epoch-7.tar",
    "--dataset", "RSNA",
    "--label", "cancer",
    "--arch", "upmc_breast_clip_det_b5_period_n_ft",
    "--epochs", "5",
    "--batch-size", "8",
    "--n_folds", "2",
    "--tensorboard-path", "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/logs",
    "--checkpoints", "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/checkpoints",
    "--output_path", "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/outputs",
]


def main() -> None:
    print("Running Mammo-CLIP full training…")
    print("Command:", " ".join(BASE_CMD))
    subprocess.run(BASE_CMD, check=True)
    print("Done! Check outputs under /hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/outputs")


if __name__ == "__main__":
    main()
