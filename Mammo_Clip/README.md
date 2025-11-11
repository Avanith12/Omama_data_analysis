## Mammo-CLIP OMAMA Experiments

This folder is where I kept every helper notebook and script while getting Mammo-CLIP to run on the balanced OMAMA dataset. The heavy assets—PNGs, CSV exports, checkpoints, logs—live on scratch at `/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip`.

### How things are set up
- **Environment**: the `mammo_clip` conda env comes from the repo `environment.yml`. I also pointed `CONDA_PKGS_DIRS` to scratch so the installs don’t eat the home quota.
- **Weights**: the published EfficientNet-B5 checkpoint (`b5-model-best-epoch-7.tar`) is parked in `scratch/.../checkpoints/Pre-trained-checkpoints/`.

### Data prep workflow
- `dataset_conversion.ipynb` handles a 100-image smoke test. It converts NPZ files to PNG, builds a tiny CSV, and shows NPZ vs. PNG side-by-side so we can eyeball quality.
- `dataset_conversion_full.ipynb` repeats the same routine for the entire dataset under `scratch/.../test_balanced/`, producing the full PNG tree plus the train/val/test CSV manifests (including the `fold` labels used later).

### Training and inference helpers
- `run_mammo_clip_pilot.py` launches the smoke test run: 2 folds, no interactive sampling, just to make sure the plumbing works.
- `run_mammo_clip_full.py` kicks off the real fine-tuning job. It calls the repo’s `train_classifier.py` with the full CSV, 5 epochs, batch size 8, and the UPMC Mammo-CLIP B5 architecture.
- `run_mammo_clip_test_eval.py` reloads the best checkpoints, fixes the tensor shape that tripped us earlier, ensembles the predictions on `omama_test.csv`, and writes `seed_10_test_eval.csv`.

All three scripts just build the long command line for `src/codebase/train_classifier.py`, which is exactly how the Mammo-CLIP authors run their experiments.

### Reading back the results
- `mammo_clip_results_review.ipynb` reads the CSVs from training and test inference, groups predictions by patient + laterality, and recomputes the usual metrics (ROC AUC, accuracy, sensitivity, specificity, F1). That gives us the numbers in the table below.

### Final patient-level metrics

| Split | Fold    | AUC   | Accuracy | Sensitivity | Specificity | F1    |
|-------|---------|-------|----------|-------------|-------------|-------|
| Val   | 0       | 0.846 | 0.782    | 0.692       | 0.863       | 0.752 |
| Val   | 1       | 0.904 | 0.841    | 0.750       | 0.894       | 0.778 |
| Val   | Overall | 0.903 | 0.841    | 0.751       | 0.891       | 0.770 |
| Test  | Overall | 0.841 | 0.761    | 0.681       | 0.839       | 0.739 |

*I pulled these numbers from the CSVs in `/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/outputs/RSNA/zz/Classifier/upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_5_weighted_BCE_n_cancer_data_frac_1.0/`. Each row in those files has a probability per image; I average them per patient and side, threshold at 0.5, and then compute the metrics so we’re looking at patient-level performance.*

