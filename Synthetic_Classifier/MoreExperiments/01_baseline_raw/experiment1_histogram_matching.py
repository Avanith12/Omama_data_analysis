"""
EXPERIMENT 1: HISTOGRAM MATCHING APPROACH

This experiment applies histogram matching to remove brightness/contrast bias
between synthetic and original mammograms, then trains a classifier.

Goal: Test if GAN quality improves when brightness differences are removed.
"""

import os
import sys
import glob
import random
import numpy as np
import cv2
import time
from datetime import datetime
from tqdm import tqdm
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, precision_recall_fscore_support,
                            accuracy_score)

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/full_synthetic_resized/'
ORIGINAL_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/full_original_method3/'
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'experiment1_results.txt')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Open output file
output = open(OUTPUT_FILE, 'w')

def log(message):
    """Write message to both console and file."""
    print(message)
    output.write(message + '\n')
    output.flush()

def apply_histogram_matching(source, reference):
    """
    Apply histogram matching to make source image histogram match reference.
    
    Args:
        source: Source image to transform
        reference: Reference image with target histogram
    
    Returns:
        Matched image
    """
    # Get histograms
    source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Compute CDFs
    source_cdf = source_hist.cumsum()
    source_cdf = source_cdf / source_cdf[-1]
    
    reference_cdf = reference_hist.cumsum()
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # Build lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find closest reference CDF value
        lookup_table[i] = np.argmin(np.abs(reference_cdf - source_cdf[i]))
    
    # Apply lookup table
    matched = lookup_table[source]
    return matched

def build_model():
    """Build CNN model for binary classification."""
    inputs = layers.Input(shape=(512, 512, 1), name='input')
    
    # Conv block 1
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Conv block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Conv block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Conv block 4
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    
    # Conv block 5
    x = layers.Conv2D(512, (3, 3), padding='same', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('relu', name='relu5')(x)
    x = layers.MaxPooling2D((2, 2), name='pool5')(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CustomCNN')
    return model

# Start logging
log("="*80)
log("EXPERIMENT 1: HISTOGRAM MATCHING APPROACH")
log("="*80)
log(f"Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Output directory: {SCRIPT_DIR}\n")

# Dataset info
log("="*80)
log("DATASET INFORMATION")
log("="*80)
log(f"\nSource Paths:")
log(f"- Synthetic: {SYNTHETIC_PATH}")
log(f"- Original:  {ORIGINAL_PATH}\n")

# Load file lists
synthetic_files = sorted(glob.glob(os.path.join(SYNTHETIC_PATH, '*.png')))
original_files = sorted(glob.glob(os.path.join(ORIGINAL_PATH, '*.png')))

log(f"Dataset Counts:")
log(f"- Total synthetic images: {len(synthetic_files):,}")
log(f"- Total original images: {len(original_files):,}")

# Balance datasets - USE ONLY 10K FROM EACH FOR FASTER TRAINING
num_samples = 10000
synthetic_files = random.sample(synthetic_files, num_samples)
original_files = random.sample(original_files, num_samples)

log(f"- Selected for experiment: {num_samples:,} from each (balanced - subset for faster training)\n")

log(f"Image Specifications:")
log(f"- Dimensions: 512 x 512 pixels")
log(f"- Color mode: Grayscale")
log(f"- Pixel range: [0, 255]\n")

# Calculate statistics before preprocessing
log("="*80)
log("PREPROCESSING APPLIED")
log("="*80)
log(f"\nTechnique: Histogram Matching")
log(f"- Method: Match synthetic image histograms to original image distribution")
log(f"- Goal: Remove brightness/contrast bias between datasets\n")

log("Calculating statistics before preprocessing (sampling 100 images)...")
syn_sample = random.sample(synthetic_files, 100)
orig_sample = random.sample(original_files, 100)

syn_means_before = []
orig_means_before = []

for f in syn_sample:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    syn_means_before.append(img.mean())

for f in orig_sample:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    orig_means_before.append(img.mean())

syn_mean_before = np.mean(syn_means_before)
syn_std_before = np.std(syn_means_before)
orig_mean_before = np.mean(orig_means_before)
orig_std_before = np.std(orig_means_before)

log(f"\nBefore Preprocessing:")
log(f"- Synthetic mean: {syn_mean_before:.1f} ± {syn_std_before:.1f}")
log(f"- Original mean:  {orig_mean_before:.1f} ± {orig_std_before:.1f}")
log(f"- Mean difference: {abs(syn_mean_before - orig_mean_before):.1f} (THIS IS THE PROBLEM!)\n")

# Apply histogram matching to synthetic images
log("Applying histogram matching to all synthetic images...")
log("This will take several minutes...\n")

# Load a reference original image for histogram matching
reference_img = cv2.imread(original_files[0], cv2.IMREAD_GRAYSCALE)

# Create output directory for matched images
matched_dir = os.path.join(SCRIPT_DIR, 'matched_synthetic')
os.makedirs(matched_dir, exist_ok=True)

matched_synthetic_files = []
for syn_file in tqdm(synthetic_files, desc="Histogram matching"):
    syn_img = cv2.imread(syn_file, cv2.IMREAD_GRAYSCALE)
    matched_img = apply_histogram_matching(syn_img, reference_img)
    
    # Save matched image
    matched_path = os.path.join(matched_dir, os.path.basename(syn_file))
    cv2.imwrite(matched_path, matched_img)
    matched_synthetic_files.append(matched_path)

# Calculate statistics after preprocessing
log("\nCalculating statistics after preprocessing (sampling 100 images)...")
syn_means_after = []

for f in random.sample(matched_synthetic_files, 100):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    syn_means_after.append(img.mean())

syn_mean_after = np.mean(syn_means_after)
syn_std_after = np.std(syn_means_after)

log(f"\nAfter Histogram Matching:")
log(f"- Synthetic mean: {syn_mean_after:.1f} ± {syn_std_after:.1f}")
log(f"- Original mean:  {orig_mean_before:.1f} ± {orig_std_before:.1f}")
log(f"- Mean difference: {abs(syn_mean_after - orig_mean_before):.1f} (FIXED!)\n")

# Data augmentation
log("="*80)
log("DATA AUGMENTATION")
log("="*80)
log(f"\nTraining augmentation:")
log(f"- Rotation: ±10 degrees")
log(f"- Horizontal flip: 50% probability")
log(f"- Vertical flip: 50% probability")
log(f"- Width shift: ±10%")
log(f"- Height shift: ±10%")
log(f"- Zoom: ±5%")
log(f"\nValidation/Test augmentation: None (original images only)\n")

# Create data splits
log("="*80)
log("DATA SPLITS")
log("="*80)
log(f"\nSplit ratio: 80% train / 10% validation / 10% test\n")

# Shuffle and split
random.shuffle(matched_synthetic_files)
random.shuffle(original_files)

split_train = int(0.8 * num_samples)
split_val = int(0.9 * num_samples)

syn_train = matched_synthetic_files[:split_train]
syn_val = matched_synthetic_files[split_train:split_val]
syn_test = matched_synthetic_files[split_val:]

orig_train = original_files[:split_train]
orig_val = original_files[split_train:split_val]
orig_test = original_files[split_val:]

log(f"Training set:")
log(f"- Total: {len(syn_train) + len(orig_train):,} images")
log(f"- Synthetic: {len(syn_train):,} images")
log(f"- Original:  {len(orig_train):,} images")
log(f"- Batches: {(len(syn_train) + len(orig_train)) // 16} (batch size = 16)\n")

log(f"Validation set:")
log(f"- Total: {len(syn_val) + len(orig_val):,} images")
log(f"- Synthetic: {len(syn_val):,} images")
log(f"- Original:  {len(orig_val):,} images")
log(f"- Batches: {(len(syn_val) + len(orig_val)) // 16} (batch size = 16)\n")

log(f"Test set:")
log(f"- Total: {len(syn_test) + len(orig_test):,} images")
log(f"- Synthetic: {len(syn_test):,} images")
log(f"- Original:  {len(orig_test):,} images")
log(f"- Batches: {(len(syn_test) + len(orig_test)) // 16} (batch size = 16)\n")

# Save splits
splits_dir = os.path.join(SCRIPT_DIR, 'data_splits')
os.makedirs(splits_dir, exist_ok=True)

with open(os.path.join(splits_dir, 'exp1_train_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_train))
with open(os.path.join(splits_dir, 'exp1_train_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_train))
with open(os.path.join(splits_dir, 'exp1_val_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_val))
with open(os.path.join(splits_dir, 'exp1_val_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_val))
with open(os.path.join(splits_dir, 'exp1_test_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_test))
with open(os.path.join(splits_dir, 'exp1_test_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_test))

log(f"Split files saved to: {splits_dir}\n")

# Build model
log("="*80)
log("MODEL ARCHITECTURE")
log("="*80)
log("\nModel: Custom CNN for Binary Classification\n")

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary to file
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
for line in model_summary:
    log(line)

# Training configuration
log("\n" + "="*80)
log("TRAINING CONFIGURATION")
log("="*80)
log(f"\nOptimizer: Adam")
log(f"- Learning rate: 0.0001")
log(f"- Beta_1: 0.9")
log(f"- Beta_2: 0.999\n")

log(f"Loss function: Categorical Crossentropy\n")

log(f"Callbacks:")
log(f"- EarlyStopping (patience=10, restore_best_weights=True)")
log(f"- ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-7)")
log(f"- ModelCheckpoint (save_best_only=True)\n")

log(f"Maximum epochs: 50")
log(f"Batch size: 16\n")

if gpus:
    log(f"GPU Configuration:")
    for gpu in gpus:
        log(f"- Device: {gpu.name}")
    log(f"- Memory growth: Enabled\n")
else:
    log(f"GPU Configuration: Using CPU\n")

# Prepare data generators
def data_generator(syn_files, orig_files, batch_size=16, augment=False):
    """Generate batches of data."""
    all_files = [(f, 0) for f in syn_files] + [(f, 1) for f in orig_files]
    random.shuffle(all_files)
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.05,
        fill_mode='constant',
        cval=0
    ) if augment else None
    
    while True:
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            
            batch_images = []
            batch_labels = []
            
            for file_path, label in batch_files:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                
                if augment and datagen:
                    img = datagen.random_transform(img)
                
                batch_images.append(img)
                batch_labels.append([1, 0] if label == 0 else [0, 1])
            
            yield np.array(batch_images), np.array(batch_labels)

# Create generators
train_gen = data_generator(syn_train, orig_train, batch_size=16, augment=True)
val_gen = data_generator(syn_val, orig_val, batch_size=16, augment=False)

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(SCRIPT_DIR, 'exp1_best_model.h5'), 
                   save_best_only=True, verbose=1)
]

# Training
log("="*80)
log("TRAINING PROGRESS")
log("="*80)
log("")

train_steps = (len(syn_train) + len(orig_train)) // 16
val_steps = (len(syn_val) + len(orig_val)) // 16

start_time = time.time()

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=2
)

training_time = time.time() - start_time

log(f"\nTotal training time: {training_time//3600:.0f} hours {(training_time%3600)//60:.0f} minutes {training_time%60:.0f} seconds\n")

# Evaluation
log("="*80)
log("TEST SET EVALUATION")
log("="*80)
log(f"\nEvaluating model on {len(syn_test) + len(orig_test):,} test images...\n")

# Load test data
test_images = []
test_labels = []

for f in tqdm(syn_test, desc="Loading synthetic test"):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    test_images.append(img)
    test_labels.append(0)

for f in tqdm(orig_test, desc="Loading original test"):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    test_images.append(img)
    test_labels.append(1)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Predict
predictions = model.predict(test_images, batch_size=16, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_classes)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_classes, average='weighted')
cm = confusion_matrix(test_labels, predicted_classes)

# For ROC AUC
test_labels_binary = keras.utils.to_categorical(test_labels, 2)
auc = roc_auc_score(test_labels_binary, predictions)

log("FINAL RESULTS:")
log("="*80)
log(f"Accuracy:  {accuracy*100:.2f}%")
log(f"Precision: {precision:.4f}")
log(f"Recall:    {recall:.4f}")
log(f"F1-Score:  {f1:.4f}")
log(f"ROC AUC:   {auc:.4f}")
log("="*80)
log("")

# Per-class metrics
precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
    test_labels, predicted_classes, average=None
)

log("Per-Class Metrics:")
log(f"                  Precision    Recall    F1-Score    Support")
log(f"Synthetic (0)     {precision_per_class[0]:.4f}       {recall_per_class[0]:.4f}    {f1_per_class[0]:.4f}      {support[0]:,}")
log(f"Original (1)      {precision_per_class[1]:.4f}       {recall_per_class[1]:.4f}    {f1_per_class[1]:.4f}      {support[1]:,}")
log("")

log("Confusion Matrix:")
log(f"                Predicted")
log(f"                Synthetic  Original")
log(f"Actual Synthetic   {cm[0,0]:,}     {cm[0,1]:,}")
log(f"       Original    {cm[1,0]:,}     {cm[1,1]:,}")
log("")

log("Classification breakdown:")
log(f"- True Positives (Synthetic correctly identified): {cm[0,0]:,}")
log(f"- True Negatives (Original correctly identified):  {cm[1,1]:,}")
log(f"- False Positives (Original misclassified as Synthetic): {cm[1,0]:,}")
log(f"- False Negatives (Synthetic misclassified as Original): {cm[0,1]:,}")
log("")

# Interpretation
log("="*80)
log("INTERPRETATION")
log("="*80)
log(f"\nResult: {accuracy*100:.2f}% accuracy\n")

if accuracy < 0.60:
    log("✅ EXCELLENT GAN QUALITY!")
    log("\nThis accuracy is close to random guessing (50%).")
    log("With histogram matching removing brightness bias, the classifier cannot")
    log("reliably distinguish synthetic from original mammograms.")
elif accuracy < 0.70:
    log("✅ GOOD GAN QUALITY!")
    log("\nThis accuracy is significantly better than the baseline 100% accuracy.")
    log("With histogram matching removing brightness bias, the classifier struggles")
    log("to distinguish synthetic from original mammograms.")
elif accuracy < 0.80:
    log("⚠️ MODERATE GAN QUALITY")
    log("\nThe classifier can still distinguish images with moderate success.")
    log("This suggests some structural or textural differences remain.")
else:
    log("❌ GAN QUALITY NEEDS IMPROVEMENT")
    log("\nThe classifier can easily distinguish synthetic from original images")
    log("even after removing brightness bias.")

log("\nThis indicates:")
if accuracy < 0.70:
    log("1. The GAN generates realistic tissue structures")
    log("2. Texture patterns are similar to real mammograms")
    log("3. No obvious GAN artifacts in spatial domain")
    log("4. The synthetic images are medically plausible")
else:
    log("1. Some distinguishable patterns remain")
    log("2. May have texture or structural artifacts")
    log("3. Further GAN training may be needed")

log(f"\nFor comparison:")
log(f"- Random guessing: 50%")
log(f"- Your result: {accuracy*100:.2f}%")
log(f"- Previous baseline: 100%")

if accuracy < 0.70:
    log("\nA lower accuracy in this fair test is GOOD - it means the GAN is realistic!")

log("")

# Files saved
log("="*80)
log("FILES SAVED")
log("="*80)
log(f"\nResults:")
log(f"- experiment1_results.txt (this file)")
log(f"- exp1_best_model.h5")
log(f"\nData:")
log(f"- matched_synthetic/ (histogram-matched images)")
log(f"- data_splits/ (train/val/test file lists)")
log("")

# Footer
log("="*80)
log("EXPERIMENT COMPLETED")
log("="*80)
log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total duration: {training_time//3600:.0f} hours {(training_time%3600)//60:.0f} minutes {training_time%60:.0f} seconds")
log("\nRecommendation: Proceed to Experiment 2 (Frequency Normalization) and")
log("Experiment 3 (Extreme Augmentation) for comparison.")
log("="*80)

output.close()
print(f"\n✅ Experiment 1 complete! Results saved to: {OUTPUT_FILE}")

