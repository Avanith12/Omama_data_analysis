"""
EXPERIMENT 2: FREQUENCY DOMAIN NORMALIZATION

This experiment normalizes images in the frequency domain (FFT) to remove
potential GAN artifacts that may appear in frequency components.

Goal: Test if GAN has frequency-domain signatures that make it distinguishable.
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
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'experiment2_results.txt')

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

def frequency_normalize(image):
    """
    Normalize image in frequency domain using FFT.
    
    This removes low-frequency components that may be GAN artifacts
    and normalizes the magnitude spectrum.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Frequency-normalized image
    """
    # Apply FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get magnitude and phase
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)
    
    # Normalize magnitude spectrum
    magnitude_normalized = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
    magnitude_normalized = magnitude_normalized * 30 + 100  # Rescale
    
    # Reconstruct
    f_shift_normalized = magnitude_normalized * np.exp(1j * phase)
    f_inverse_shift = np.fft.ifftshift(f_shift_normalized)
    img_back = np.fft.ifft2(f_inverse_shift)
    img_back = np.abs(img_back)
    
    # Normalize to [0, 255]
    img_back = np.clip(img_back, 0, 255)
    img_back = img_back.astype(np.uint8)
    
    return img_back

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
log("EXPERIMENT 2: FREQUENCY DOMAIN NORMALIZATION")
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

# Preprocessing
log("="*80)
log("PREPROCESSING APPLIED")
log("="*80)
log(f"\nTechnique: Frequency Domain Normalization (FFT)")
log(f"- Method: Transform to frequency domain, normalize magnitude spectrum")
log(f"- Goal: Remove GAN artifacts in frequency components\n")

log("Processing details:")
log("1. Apply 2D Fast Fourier Transform (FFT)")
log("2. Separate magnitude and phase components")
log("3. Normalize magnitude spectrum (mean=0, std=1)")
log("4. Reconstruct image from normalized magnitude + original phase")
log("5. Inverse FFT to return to spatial domain")
log("\nThis removes systematic frequency patterns that GANs may introduce.\n")

# Apply frequency normalization
log("Applying frequency normalization to ALL images...")
log("This will take several minutes...\n")

# Create output directories
freq_syn_dir = os.path.join(SCRIPT_DIR, 'freq_normalized_synthetic')
freq_orig_dir = os.path.join(SCRIPT_DIR, 'freq_normalized_original')
os.makedirs(freq_syn_dir, exist_ok=True)
os.makedirs(freq_orig_dir, exist_ok=True)

freq_synthetic_files = []
for syn_file in tqdm(synthetic_files, desc="Normalizing synthetic"):
    syn_img = cv2.imread(syn_file, cv2.IMREAD_GRAYSCALE)
    freq_img = frequency_normalize(syn_img)
    
    freq_path = os.path.join(freq_syn_dir, os.path.basename(syn_file))
    cv2.imwrite(freq_path, freq_img)
    freq_synthetic_files.append(freq_path)

freq_original_files = []
for orig_file in tqdm(original_files, desc="Normalizing original"):
    orig_img = cv2.imread(orig_file, cv2.IMREAD_GRAYSCALE)
    freq_img = frequency_normalize(orig_img)
    
    freq_path = os.path.join(freq_orig_dir, os.path.basename(orig_file))
    cv2.imwrite(freq_path, freq_img)
    freq_original_files.append(freq_path)

log("\nFrequency normalization complete!")
log(f"- Processed synthetic images saved to: {freq_syn_dir}")
log(f"- Processed original images saved to: {freq_orig_dir}\n")

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
random.shuffle(freq_synthetic_files)
random.shuffle(freq_original_files)

split_train = int(0.8 * num_samples)
split_val = int(0.9 * num_samples)

syn_train = freq_synthetic_files[:split_train]
syn_val = freq_synthetic_files[split_train:split_val]
syn_test = freq_synthetic_files[split_val:]

orig_train = freq_original_files[:split_train]
orig_val = freq_original_files[split_train:split_val]
orig_test = freq_original_files[split_val:]

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

with open(os.path.join(splits_dir, 'exp2_train_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_train))
with open(os.path.join(splits_dir, 'exp2_train_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_train))
with open(os.path.join(splits_dir, 'exp2_val_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_val))
with open(os.path.join(splits_dir, 'exp2_val_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_val))
with open(os.path.join(splits_dir, 'exp2_test_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_test))
with open(os.path.join(splits_dir, 'exp2_test_orig.txt'), 'w') as f:
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
log(f"- Learning rate: 0.0001\n")

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
    ModelCheckpoint(os.path.join(SCRIPT_DIR, 'exp2_best_model.h5'), 
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

# Interpretation
log("="*80)
log("INTERPRETATION")
log("="*80)
log(f"\nResult: {accuracy*100:.2f}% accuracy\n")

if accuracy < 0.60:
    log("✅ EXCELLENT - No frequency domain artifacts!")
    log("\nThe GAN does not have detectable frequency-domain signatures.")
elif accuracy < 0.70:
    log("✅ GOOD - Minimal frequency artifacts")
    log("\nThe classifier struggles even when looking at frequency components.")
elif accuracy < 0.80:
    log("⚠️ MODERATE - Some frequency patterns detectable")
    log("\nThe GAN may have some frequency-domain artifacts.")
else:
    log("❌ GAN has strong frequency signatures")
    log("\nThe classifier can easily detect GAN patterns in frequency domain.")

log(f"\nComparison:")
log(f"- Random guessing: 50%")
log(f"- Your result: {accuracy*100:.2f}%")
log(f"- Previous baseline: 100%")
log("")

# Files saved
log("="*80)
log("FILES SAVED")
log("="*80)
log(f"\nResults:")
log(f"- experiment2_results.txt (this file)")
log(f"- exp2_best_model.h5")
log(f"\nData:")
log(f"- freq_normalized_synthetic/ (FFT-normalized synthetic images)")
log(f"- freq_normalized_original/ (FFT-normalized original images)")
log(f"- data_splits/ (train/val/test file lists)")
log("")

# Footer
log("="*80)
log("EXPERIMENT COMPLETED")
log("="*80)
log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total duration: {training_time//3600:.0f} hours {(training_time%3600)//60:.0f} minutes {training_time%60:.0f} seconds")
log("\nRecommendation: Compare with Experiment 1 and 3 results.")
log("="*80)

output.close()
print(f"\n✅ Experiment 2 complete! Results saved to: {OUTPUT_FILE}")

