#!/usr/bin/env python
"""Run Mammo-CLIP checkpoints on the OMAMA balanced test split."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "codebase"
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

from Classifiers.models.breast_clip_classifier import BreastClipClassifier
from Datasets.dataset_concepts import MammoDataset, collator_mammo_dataset_w_concepts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    data_dir: Path = Path("/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/data")
    img_dir: str = "omama_balanced/images_png"
    csv_path: Path = Path(
        "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/data/omama_balanced/csv/omama_test.csv"
    )
    pretrain_ckpt: Path = Path(
        "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/checkpoints/Pre-trained-checkpoints/"
        "b5-model-best-epoch-7.tar"
    )
    finetuned_ckpts: List[Path] = field(
        default_factory=lambda: [
            Path(
                "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/checkpoints/RSNA/Classifier/"
                "upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_5_weighted_BCE_n_cancer_data_frac_1.0/"
                "upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_aucroc_ver084.pth"
            ),
            Path(
                "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/checkpoints/RSNA/Classifier/"
                "upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_5_weighted_BCE_n_cancer_data_frac_1.0/"
                "upmc_breast_clip_det_b5_period_n_ft_seed_10_fold1_best_aucroc_ver084.pth"
            ),
        ]
    )
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    arch: str = "upmc_breast_clip_det_b5_period_n_ft"
    label: str = "cancer"
    mean: float = 0.3089279
    std: float = 0.25053555408335154
    output_csv: Path = Path(
        "/hpcstor6/scratch01/a/a.kanamarlapudi001/mammo-clip/outputs/RSNA/zz/Classifier/"
        "upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_5_weighted_BCE_n_cancer_data_frac_1.0/"
        "seed_10_test_eval.csv"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_args(cfg: EvalConfig):
    """Create a lightweight args namespace used by dataset/model classes."""

    class _Args:
        pass

    args = _Args()
    args.data_dir = cfg.data_dir
    args.img_dir = cfg.img_dir
    args.dataset = "RSNA"
    args.label = cfg.label
    args.arch = cfg.arch
    args.mean = cfg.mean
    args.std = cfg.std
    args.image_encoder_type = None
    args.model_type = "Classifier"
    args.alpha = 10.0
    args.sigma = 15.0
    args.p = 1.0
    args.img_size = [1520, 912]
    args.num_workers = cfg.num_workers
    args.batch_size = cfg.batch_size
    return args


def load_model(args, cfg: EvalConfig, finetuned_ckpt: Path, device: torch.device):
    """Initialise Mammo-CLIP classifier and load the fine-tuned checkpoint."""

    print(f"Loading pretrain checkpoint from {cfg.pretrain_ckpt}")
    pretrain = torch.load(cfg.pretrain_ckpt, map_location="cpu")
    args.image_encoder_type = pretrain["config"]["model"]["image_encoder"]["model_type"]

    model = BreastClipClassifier(args, ckpt=pretrain, n_class=1)
    state = torch.load(finetuned_ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def make_dataloader(args, cfg: EvalConfig):
    df = pd.read_csv(cfg.csv_path)
    df[cfg.label] = df[cfg.label].astype(int)
    dataset = MammoDataset(args=args, df=df, transform=None)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator_mammo_dataset_w_concepts,
    )
    return df, loader


def ensemble_predict(models, loader, device):
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["x"]
            if images.dim() == 5:  # (B, 1, H, W, C)
                images = images.squeeze(1)
            if images.dim() == 4 and images.shape[-1] in {3, 4}:  # channel-last -> channel-first
                images = images.permute(0, 3, 1, 2)
            images = images.to(device)
            logits_per_model = []
            for model in models:
                logits = model(images).detach().cpu().numpy()
                logits_per_model.append(logits)
            stacked = np.stack(logits_per_model, axis=0)  # [n_models, batch, 1]
            all_logits.append(stacked)
    logits_concat = np.concatenate(all_logits, axis=1)  # [n_models, N, 1]
    probs = torch.sigmoid(torch.from_numpy(logits_concat)).numpy()
    mean_probs = probs.mean(axis=0).squeeze()
    return mean_probs


def compute_metrics(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5):
    df = df.copy()
    df["prediction"] = probs
    df["prediction_bin"] = (df["prediction"] >= threshold).astype(int)

    patient_agg = (
        df[["patient_id", "laterality", "cancer", "prediction"]]
        .groupby(["patient_id", "laterality"], as_index=False)
        .mean()
    )

    auc = roc_auc_score(patient_agg["cancer"], patient_agg["prediction"])
    acc = accuracy_score(patient_agg["cancer"], (patient_agg["prediction"] >= threshold).astype(int))
    sens = (
        ((patient_agg["cancer"] == 1) & (patient_agg["prediction"] >= threshold)).sum()
        / (patient_agg["cancer"] == 1).sum()
    )
    spec = (
        ((patient_agg["cancer"] == 0) & (patient_agg["prediction"] < threshold)).sum()
        / (patient_agg["cancer"] == 0).sum()
    )
    f1 = f1_score(patient_agg["cancer"], (patient_agg["prediction"] >= threshold).astype(int))

    return df, {
        "auc": auc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "n_patients": len(patient_agg),
    }


def main():
    cfg = EvalConfig()
    args = build_args(cfg)
    device = torch.device(cfg.device)

    df, loader = make_dataloader(args, cfg)

    models = [load_model(build_args(cfg), cfg, ckpt, device) for ckpt in cfg.finetuned_ckpts]

    probs = ensemble_predict(models, loader, device)
    df_out, metrics = compute_metrics(df, probs)

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg.output_csv, index=False)

    print("Saved test predictions to", cfg.output_csv)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
