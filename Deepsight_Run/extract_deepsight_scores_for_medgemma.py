#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve
)

EDWARD_DATASET_PATH = "/hpcstor6/scratch01/e/edward.gaibor001/omamadata256/hf_arrow_balanced"
METADATA_DIR = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/2d_resized_512/metadata"
OUTPUT_DIR = "/home/a.kanamarlapudi001/projects/omama-proj/avanith/deepsightv"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "deepsight_scores.json")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "deepsight_scores.csv")

def load_dataset_filenames(dataset_path):
    print("\n[1/6] Loading dataset...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"  Loaded: {len(dataset['train'])} train + {len(dataset['validation'])} val")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, None
    
    print("\n[2/6] Extracting filenames...")
    all_filenames = []
    all_labels = []
    all_splits = []
    
    for split_name in ['train', 'validation']:
        split_data = dataset[split_name]
        for i in range(len(split_data)):
            filename = split_data[i]['filename']
            label_idx = split_data[i]['label']
            label_name = dataset['train'].features['label'].int2str(label_idx)
            
            all_filenames.append(filename)
            all_labels.append(label_name)
            all_splits.append(split_name)
    
    print(f"  Got {len(all_filenames)} filenames (Train: {all_splits.count('train')}, Val: {all_splits.count('validation')})")
    return all_filenames, all_labels, all_splits


def extract_metadata_for_files(filenames, labels, splits, metadata_dir):
    print("\n[3/6] Extracting DeepSight scores from metadata...")
    results = []
    missing_metadata = []
    missing_scores = []
    
    for i, filename in enumerate(tqdm(filenames, desc="Processing")):
        json_filename = filename.replace('.npz', '.json')
        metadata_path = os.path.join(metadata_dir, json_filename)
        
        if not os.path.exists(metadata_path):
            missing_metadata.append(filename)
            continue
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            deepsight_score = metadata.get('score', None)
            coords = metadata.get('coords', None)
            
            if deepsight_score is None:
                missing_scores.append(filename)
            
            results.append({
                'filename': filename,
                'split': splits[i],
                'ground_truth_label': labels[i],
                'deepsight_score': deepsight_score,
                'deepsight_coords': coords,
                'sop_instance_uid': metadata.get('SOPInstanceUID', None),
                'patient_id': metadata.get('PatientID', None),
                'view': metadata.get('View', None),
                'image_laterality': metadata.get('ImageLaterality', None),
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            missing_metadata.append(filename)
    
    print(f"  Matched {len(results)} files")
    if missing_metadata:
        print(f"  Missing metadata: {len(missing_metadata)}")
    if missing_scores:
        print(f"  Missing scores: {len(missing_scores)}")
    
    return results, missing_metadata, missing_scores


def calculate_metrics_at_threshold(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def calculate_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def calculate_deepsight_performance(df):
    print("\n[4/6] Calculating performance metrics...")
    print("  Using validation set only")
    
    df_val = df[(df['split'] == 'validation') & (df['deepsight_score'].notna())].copy()
    
    if len(df_val) == 0:
        print("  No validation samples found")
        return {}
    
    print(f"  Evaluating on {len(df_val)} validation samples")
    
    df_val['ground_truth_binary'] = (df_val['ground_truth_label'] == 'Cancer').astype(int)
    
    y_true = df_val['ground_truth_binary'].values
    y_scores = df_val['deepsight_score'].values
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"  AUC-ROC: {auc:.4f}")
    
    optimal_threshold = calculate_optimal_threshold(y_true, y_scores)
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    
    thresholds_to_test = [0.3, 0.4, 0.5, optimal_threshold, 0.6, 0.7]
    thresholds_to_test = sorted(set([round(t, 4) for t in thresholds_to_test]))
    
    metrics_by_threshold = {}
    optimal_metrics = None
    
    for threshold in thresholds_to_test:
        metrics = calculate_metrics_at_threshold(y_true, y_scores, threshold)
        metrics_by_threshold[threshold] = metrics
        
        if abs(threshold - optimal_threshold) < 0.0001:
            optimal_metrics = metrics
        
        if threshold == optimal_threshold or threshold == 0.5:
            print(f"\n  Threshold {threshold:.4f}:")
            print(f"    Accuracy:    {metrics['accuracy']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
            print(f"    F1:          {metrics['f1_score']:.4f}")
            print(f"    Confusion:   TP={metrics['true_positives']}, FP={metrics['false_positives']}, "
                  f"TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")
    
    metrics_at_05 = metrics_by_threshold.get(0.5)
    
    return {
        'auc': auc,
        'threshold_used': 0.5,
        'optimal_threshold': optimal_threshold,
        'metrics_at_05': metrics_at_05,
        'optimal_metrics': optimal_metrics,
        'metrics_by_threshold': metrics_by_threshold,
        'n_samples_validation': len(df_val),
        'evaluation_set': 'validation_only'
    }


def save_results(df, results, missing_metadata, missing_scores, performance, output_dir):
    print("\n[5/6] Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV: {OUTPUT_CSV}")
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {OUTPUT_JSON}")
    
    if performance:
        metrics_file = os.path.join(output_dir, "deepsight_performance_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(performance, f, indent=2)
        print(f"  Metrics: {metrics_file}")
    
    if missing_metadata or missing_scores:
        missing_report = os.path.join(output_dir, "missing_files_report.txt")
        with open(missing_report, 'w') as f:
            f.write("MISSING METADATA FILES:\n")
            f.write("="*80 + "\n")
            for fname in missing_metadata:
                f.write(f"{fname}\n")
            f.write("\n\nMISSING DEEPSIGHT SCORES:\n")
            f.write("="*80 + "\n")
            for fname in missing_scores:
                f.write(f"{fname}\n")
        print(f"  Missing files report: {missing_report}")


def print_summary_statistics(df):
    print("\n[6/6] Summary stats...")
    
    print("\n  Breakdown by split and label:")
    summary = df.groupby(['split', 'ground_truth_label']).agg({
        'filename': 'count',
        'deepsight_score': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(summary)
    
    print("\n  DeepSight score distribution:")
    print(f"    Samples with scores: {df['deepsight_score'].notna().sum()}")
    print(f"    Mean: {df['deepsight_score'].mean():.4f}")
    print(f"    Std:  {df['deepsight_score'].std():.4f}")
    print(f"    Range: [{df['deepsight_score'].min():.4f}, {df['deepsight_score'].max():.4f}]")
    
    print("\n  Scores by ground truth:")
    for label in ['Cancer', 'NonCancer']:
        if label in df['ground_truth_label'].values:
            subset = df[df['ground_truth_label'] == label]['deepsight_score']
            print(f"    {label}: mean={subset.mean():.4f}, std={subset.std():.4f}")


def save_results_summary(performance, output_dir):
    output_file = os.path.join(output_dir, "deepsight_results_summary.txt")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEPSIGHT PERFORMANCE RESULTS\n")
        f.write("="*80 + "\n\n")
        
        if performance and performance.get('metrics_at_05'):
            metrics = performance['metrics_at_05']
            
            f.write(f"Dataset: Edward's balanced dataset (validation set only)\n")
            f.write(f"Samples: {performance['n_samples_validation']:,}\n")
            f.write(f"AUC-ROC: {performance['auc']:.4f}\n")
            f.write(f"Threshold Used: {performance['threshold_used']:.1f} (DeepSight standard)\n\n")
            
            f.write("="*80 + "\n")
            f.write("METRICS FOR COMPARISON WITH MEDGEMMA (Threshold = 0.5)\n")
            f.write("="*80 + "\n")
            f.write(f"Accuracy:    {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Sensitivity: {metrics['sensitivity']*100:.2f}%\n")
            f.write(f"Specificity: {metrics['specificity']*100:.2f}%\n")
            f.write(f"F1 Score:    {metrics['f1_score']*100:.2f}%\n\n")
            
            f.write("="*80 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("="*80 + "\n")
            f.write(f"True Positives:  {metrics['true_positives']}\n")
            f.write(f"False Positives: {metrics['false_positives']}\n")
            f.write(f"True Negatives:  {metrics['true_negatives']}\n")
            f.write(f"False Negatives: {metrics['false_negatives']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("NOTE\n")
            f.write("="*80 + "\n")
            f.write(f"Optimal threshold (Youden's J): {performance['optimal_threshold']:.4f}\n")
            f.write(f"But using 0.5 for fair comparison (DeepSight's standard threshold)\n\n")
            
            f.write("="*80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*80 + "\n")
            f.write(f"{OUTPUT_CSV}\n")
            f.write(f"{OUTPUT_JSON}\n")
            f.write(f"{os.path.join(output_dir, 'deepsight_performance_metrics.json')}\n")
            f.write("="*80 + "\n")
    
    print(f"\n  Results summary saved: {output_file}")
    return output_file


def main():
    print("="*80)
    print("DEEPSIGHT vs MEDGEMMA COMPARISON")
    print("="*80)
    print("\nPlan:")
    print("  1. Load Edward's dataset")
    print(f"     {EDWARD_DATASET_PATH}")
    print("  2. Extract DeepSight scores from metadata")
    print(f"     {METADATA_DIR}")
    print("  3. Calculate metrics on validation set")
    print("  4. Save results")
    print("="*80)
    
    filenames, labels, splits = load_dataset_filenames(EDWARD_DATASET_PATH)
    if filenames is None:
        return
    
    results, missing_metadata, missing_scores = extract_metadata_for_files(
        filenames, labels, splits, METADATA_DIR
    )
    
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No data found")
        return
    
    performance = calculate_deepsight_performance(df)
    
    print_summary_statistics(df)
    
    save_results(df, results, missing_metadata, missing_scores, performance, OUTPUT_DIR)
    
    save_results_summary(performance, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    
    if performance and performance.get('metrics_at_05'):
        metrics = performance['metrics_at_05']
        
        print("\nDEEPSIGHT RESULTS (Validation Set):")
        print(f"\n  Samples:   {performance['n_samples_validation']:,}")
        print(f"  AUC-ROC:   {performance['auc']:.4f}")
        print(f"  Threshold: {performance['threshold_used']:.1f} (DeepSight standard)")
        
        print("\n  Metrics for comparison with MedGemma:")
        print(f"    Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"    Sensitivity: {metrics['sensitivity']*100:.2f}%")
        print(f"    Specificity: {metrics['specificity']*100:.2f}%")
        print(f"    F1 Score:    {metrics['f1_score']*100:.2f}%")
        
        print(f"\n  Confusion: TP={metrics['true_positives']}, FP={metrics['false_positives']}, "
              f"TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")
    
    print("\n Output files:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")
    print(f"  {os.path.join(OUTPUT_DIR, 'deepsight_performance_metrics.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'deepsight_results_summary.txt')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

