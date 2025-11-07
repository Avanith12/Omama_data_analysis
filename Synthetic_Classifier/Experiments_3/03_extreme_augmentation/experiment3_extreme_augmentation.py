"""
EXPERIMENT 3: EXTREME DATA AUGMENTATION

This experiment uses raw data but applies extreme augmentation to force the
classifier to ignore brightness/contrast differences and focus on structure.

Goal: Test if aggressive augmentation can make the classifier rely on
      structural features rather than simple brightness patterns.
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

from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, precision_recall_fscore_support,
                            accuracy_score)

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/full_synthetic_resized/'
ORIGINAL_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/full_original_method3/'
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'experiment3_results.txt')

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

def extreme_augment(image):
    """
    Apply extreme augmentation to an image.
    
    Transformations:
    - Random brightness: ±40%
    - Random contrast: ±40%
    - Gaussian noise
    - Random gamma correction
    - Random cutout/erasing
    
    Args:
        image: Input image (0-1 range, single channel)
    
    Returns:
        Augmented image
    """
    img = image.copy()
    
    # Random brightness (±40%)
    brightness_factor = np.random.uniform(0.6, 1.4)
    img = img * brightness_factor
    
    # Random contrast (±40%)
    mean = img.mean()
    contrast_factor = np.random.uniform(0.6, 1.4)
    img = (img - mean) * contrast_factor + mean
    
    # Gaussian noise
    noise = np.random.normal(0, 0.02, img.shape)
    img = img + noise
    
    # Random gamma correction
    gamma = np.random.uniform(0.7, 1.3)
    img = np.power(np.clip(img, 0, 1), gamma)
    
    # Random cutout (10% of image)
    if np.random.rand() > 0.5:
        h, w = img.shape[:2]
        cutout_size = int(min(h, w) * 0.1)
        y = np.random.randint(0, h - cutout_size)
        x = np.random.randint(0, w - cutout_size)
        img[y:y+cutout_size, x:x+cutout_size] = 0
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img

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
log("EXPERIMENT 3: EXTREME DATA AUGMENTATION")
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
log(f"\nTechnique: NONE (Raw images used)")
log(f"- Images loaded as-is without modification")
log(f"- Goal: Test if extreme augmentation alone can prevent brightness bias")
log(f"\nOriginal statistics (from previous analysis):")
log(f"- Synthetic mean: ~32.1")
log(f"- Original mean:  ~21.9")
log(f"- Mean difference: ~10.2\n")

# Data augmentation
log("="*80)
log("DATA AUGMENTATION (EXTREME)")
log("="*80)
log(f"\nTraining augmentation (AGGRESSIVE):")
log(f"- Random brightness adjustment: ±40%")
log(f"- Random contrast adjustment: ±40%")
log(f"- Gaussian noise injection: σ=0.02")
log(f"- Random gamma correction: γ ∈ [0.7, 1.3]")
log(f"- Random cutout/erasing: 10% of image area")
log(f"- Rotation: ±15 degrees")
log(f"- Horizontal flip: 50% probability")
log(f"- Vertical flip: 50% probability")
log(f"- Width shift: ±15%")
log(f"- Height shift: ±15%")
log(f"- Zoom: ±10%")
log(f"\nGoal: Force model to ignore brightness/contrast and focus on structure")
log(f"\nValidation/Test augmentation: None (original images only)\n")

# Create data splits
log("="*80)
log("DATA SPLITS")
log("="*80)
log(f"\nSplit ratio: 80% train / 10% validation / 10% test\n")

# Shuffle and split
random.shuffle(synthetic_files)
random.shuffle(original_files)

split_train = int(0.8 * num_samples)
split_val = int(0.9 * num_samples)

syn_train = synthetic_files[:split_train]
syn_val = synthetic_files[split_train:split_val]
syn_test = synthetic_files[split_val:]

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

with open(os.path.join(splits_dir, 'exp3_train_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_train))
with open(os.path.join(splits_dir, 'exp3_train_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_train))
with open(os.path.join(splits_dir, 'exp3_val_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_val))
with open(os.path.join(splits_dir, 'exp3_val_orig.txt'), 'w') as f:
    f.write('\n'.join(orig_val))
with open(os.path.join(splits_dir, 'exp3_test_syn.txt'), 'w') as f:
    f.write('\n'.join(syn_test))
with open(os.path.join(splits_dir, 'exp3_test_orig.txt'), 'w') as f:
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

# Prepare data generators with extreme augmentation
def data_generator(syn_files, orig_files, batch_size=16, augment=False):
    """Generate batches of data with extreme augmentation."""
    all_files = [(f, 0) for f in syn_files] + [(f, 1) for f in orig_files]
    random.shuffle(all_files)
    
    while True:
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            
            batch_images = []
            batch_labels = []
            
            for file_path, label in batch_files:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                
                if augment:
                    # Apply extreme augmentation (works on 2D)
                    img_2d = img.copy()
                    img_2d = extreme_augment(np.expand_dims(img_2d, axis=-1))
                    img = img_2d.squeeze()  # Back to 2D for CV2 operations
                    
                    # Additional geometric augmentations (CV2 works better with 2D)
                    # Rotation
                    if np.random.rand() > 0.5:
                        angle = np.random.uniform(-15, 15)
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                        img = cv2.warpAffine(img, M, (w, h), borderValue=0)
                    
                    # Flip
                    if np.random.rand() > 0.5:
                        img = np.fliplr(img)
                    if np.random.rand() > 0.5:
                        img = np.flipud(img)
                    
                    # Shift
                    if np.random.rand() > 0.5:
                        h, w = img.shape
                        shift_x = int(np.random.uniform(-0.15, 0.15) * w)
                        shift_y = int(np.random.uniform(-0.15, 0.15) * h)
                        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                        img = cv2.warpAffine(img, M, (w, h), borderValue=0)
                    
                    # Zoom
                    if np.random.rand() > 0.5:
                        zoom_factor = np.random.uniform(0.9, 1.1)
                        h, w = img.shape
                        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                        img_resized = cv2.resize(img, (new_w, new_h))
                        
                        if zoom_factor > 1:
                            # Crop center
                            start_h = (new_h - h) // 2
                            start_w = (new_w - w) // 2
                            img = img_resized[start_h:start_h+h, start_w:start_w+w]
                        else:
                            # Pad
                            pad_h = (h - new_h) // 2
                            pad_w = (w - new_w) // 2
                            img = np.pad(img_resized, ((pad_h, h-new_h-pad_h), 
                                                       (pad_w, w-new_w-pad_w)), 
                                        mode='constant', constant_values=0)
                            img = img[:h, :w]  # Ensure correct size
                
                # Add channel dimension at the end
                img = np.expand_dims(img, axis=-1)
                
                # Ensure correct shape
                if img.shape != (512, 512, 1):
                    img = cv2.resize(img.squeeze(), (512, 512))
                    img = np.expand_dims(img, axis=-1)
                
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
    ModelCheckpoint(os.path.join(SCRIPT_DIR, 'exp3_best_model.h5'), 
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

# Load test data (NO augmentation)
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
    log("✅ EXCELLENT - Augmentation successfully removed bias!")
    log("\nExtreme augmentation forced the model to ignore brightness/contrast.")
    log("The model cannot reliably distinguish based on remaining features.")
elif accuracy < 0.70:
    log("✅ GOOD - Augmentation reduced bias effectively")
    log("\nExtreme augmentation helped, but some patterns remain distinguishable.")
elif accuracy < 0.80:
    log("⚠️ MODERATE - Augmentation had limited effect")
    log("\nEven with extreme augmentation, the model finds distinguishing patterns.")
else:
    log("❌ Augmentation ineffective - strong structural differences")
    log("\nThe differences are too fundamental for augmentation to mask.")

log(f"\nComparison:")
log(f"- Random guessing: 50%")
log(f"- Your result: {accuracy*100:.2f}%")
log(f"- Previous baseline: 100%")
log("")

log("Key insight:")
if accuracy < 0.70:
    log("The brightness bias was the main distinguishing factor!")
    log("When forced to ignore it, the model struggles - indicating good GAN quality.")
else:
    log("Beyond brightness, there are structural/textural differences.")
    log("Compare with Experiment 1 and 2 to understand what they are.")

log("")

# Files saved
log("="*80)
log("FILES SAVED")
log("="*80)
log(f"\nResults:")
log(f"- experiment3_results.txt (this file)")
log(f"- exp3_best_model.h5")
log(f"\nData:")
log(f"- data_splits/ (train/val/test file lists)")
log("")

# Footer
log("="*80)
log("EXPERIMENT COMPLETED")
log("="*80)
log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total duration: {training_time//3600:.0f} hours {(training_time%3600)//60:.0f} minutes {training_time%60:.0f} seconds")
log("\nRecommendation: Compare all 3 experiments to understand what makes images distinguishable.")
log("="*80)

output.close()
print(f"\n✅ Experiment 3 complete! Results saved to: {OUTPUT_FILE}")

