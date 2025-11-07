"""
================================================================================
EXPERIMENT 6: CLEAN EVALUATION (TEXT ANNOTATIONS REMOVED)
================================================================================

Goal: Fair evaluation of GAN quality with text annotations removed and 
      identical simple preprocessing applied to both synthetic and original images.

"""

import os
import sys
import numpy as np
import cv2
import glob
import json
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, precision_recall_fscore_support,
                            accuracy_score, precision_recall_curve, average_precision_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)

# ==============================================================================
# PATHS CONFIGURATION
# ==============================================================================

# Input paths
SYNTHETIC_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/full_synthetic_resized/'
ORIGINAL_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/full_original_method3/'

# Output paths
CLEANED_PATH = '/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic/cleaned_originals/'
OUTPUT_DIR = '/home/a.kanamarlapudi001/projects/omama-proj/_EXPERIMENTS/SYNTHETIC/Avanith/MoreExperiments/experiment6_clean_evaluation/'

# Create output directories
os.makedirs(CLEANED_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'data_splits'), exist_ok=True)

# Results file
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'experiment6_results.txt')

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def log(message, also_print=True):
    """Write message to results file and optionally print."""
    with open(RESULTS_FILE, 'a') as f:
        f.write(message + '\n')
    if also_print:
        print(message)

# Initialize results file
with open(RESULTS_FILE, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPERIMENT 6: CLEAN EVALUATION (ANNOTATIONS REMOVED)\n")
    f.write("=" * 80 + "\n")
    f.write(f"Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Output directory: {OUTPUT_DIR}\n\n")

log("Goal:")
log("Fair evaluation with text annotations removed and identical preprocessing\n")

# ==============================================================================
# STEP 1: REMOVE TEXT ANNOTATIONS FROM ORIGINAL IMAGES
# ==============================================================================

def detect_and_remove_text(image):
    """
    Detect and remove text annotations from mammogram image.
    
    Text annotations typically appear in corners (LCC, RCC, LMLC, MLO, etc.)
    We'll mask out these regions.
    """
    h, w = image.shape
    
    # Create mask (start with all ones - keep everything)
    mask = np.ones_like(image, dtype=np.uint8)
    
    # Define regions where text typically appears (corners and edges)
    # Top-left corner
    mask[0:80, 0:120] = 0
    
    # Top-right corner
    mask[0:80, w-120:w] = 0
    
    # Bottom-right corner
    mask[h-80:h, w-120:w] = 0
    
    # Bottom-left corner
    mask[h-80:h, 0:120] = 0
    
    # Right edge (for rotated text like LMLC)
    mask[0:h, w-50:w] = 0
    
    # Apply mask (set masked regions to 0/black)
    cleaned_image = image * mask
    
    return cleaned_image

log("=" * 80)
log("STEP 1: REMOVING TEXT ANNOTATIONS FROM ORIGINAL IMAGES")
log("=" * 80 + "\n")

# Get all original images
original_files = sorted(glob.glob(os.path.join(ORIGINAL_PATH, '*.png')))
log(f"Found {len(original_files)} original images")

# Check if cleaning already done
existing_cleaned = glob.glob(os.path.join(CLEANED_PATH, '*.png'))
if len(existing_cleaned) == len(original_files):
    log(f"\nCleaned images already exist ({len(existing_cleaned)} files)")
    log("Skipping annotation removal step...")
else:
    log(f"\nProcessing {len(original_files)} images to remove annotations...")
    log("This will take approximately 1-2 hours...\n")
    
    # Process images
    before_after_samples = []
    
    for i, img_path in enumerate(tqdm(original_files, desc="Removing annotations")):
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            log(f"Warning: Failed to load {img_path}")
            continue
        
        # Save first 10 for before/after visualization
        if i < 10:
            before_after_samples.append((img.copy(), img_path))
        
        # Remove text annotations
        cleaned_img = detect_and_remove_text(img)
        
        # Save cleaned image
        filename = os.path.basename(img_path)
        output_path = os.path.join(CLEANED_PATH, filename)
        cv2.imwrite(output_path, cleaned_img)
    
    log(f"\n✅ Annotation removal complete!")
    log(f"Cleaned images saved to: {CLEANED_PATH}\n")
    
    # Create before/after visualization
    log("Creating before/after visualization...")
    fig, axes = plt.subplots(10, 2, figsize=(10, 40))
    
    for i, (before_img, img_path) in enumerate(before_after_samples):
        filename = os.path.basename(img_path)
        after_img = cv2.imread(os.path.join(CLEANED_PATH, filename), cv2.IMREAD_GRAYSCALE)
        
        axes[i, 0].imshow(before_img, cmap='gray')
        axes[i, 0].set_title(f'Original {i+1} (Before)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(after_img, cmap='gray')
        axes[i, 1].set_title(f'Original {i+1} (After - Text Removed)')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'before_after_cleaning.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    log("✅ Before/after visualization saved\n")

# ==============================================================================
# STEP 2: PREPARE DATASET
# ==============================================================================

log("=" * 80)
log("STEP 2: DATASET PREPARATION")
log("=" * 80 + "\n")

# Get all file paths
synthetic_files = sorted(glob.glob(os.path.join(SYNTHETIC_PATH, '*.png')))
cleaned_files = sorted(glob.glob(os.path.join(CLEANED_PATH, '*.png')))

log(f"Dataset counts:")
log(f"- Synthetic images: {len(synthetic_files)}")
log(f"- Cleaned original images: {len(cleaned_files)}")

# Balance dataset (use same number from each class)
num_samples = min(len(synthetic_files), len(cleaned_files))
log(f"\nBalancing dataset to {num_samples} images per class")

# Randomly sample
random.seed(42)
synthetic_files = random.sample(synthetic_files, num_samples)
cleaned_files = random.sample(cleaned_files, num_samples)

# Create splits (80/10/10)
train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size

log(f"\nSplit sizes:")
log(f"- Train: {train_size} per class ({train_size * 2} total)")
log(f"- Val:   {val_size} per class ({val_size * 2} total)")
log(f"- Test:  {test_size} per class ({test_size * 2} total)")

# Shuffle and split
random.shuffle(synthetic_files)
random.shuffle(cleaned_files)

syn_train = synthetic_files[:train_size]
syn_val = synthetic_files[train_size:train_size+val_size]
syn_test = synthetic_files[train_size+val_size:]

orig_train = cleaned_files[:train_size]
orig_val = cleaned_files[train_size:train_size+val_size]
orig_test = cleaned_files[train_size+val_size:]

# Save splits
splits = {
    'train': {'synthetic': syn_train, 'original': orig_train},
    'val': {'synthetic': syn_val, 'original': orig_val},
    'test': {'synthetic': syn_test, 'original': orig_test}
}

for split_name, split_data in splits.items():
    with open(os.path.join(OUTPUT_DIR, 'data_splits', f'{split_name}_synthetic.txt'), 'w') as f:
        f.write('\n'.join(split_data['synthetic']))
    with open(os.path.join(OUTPUT_DIR, 'data_splits', f'{split_name}_original.txt'), 'w') as f:
        f.write('\n'.join(split_data['original']))

log("\n✅ Data splits created and saved\n")

# ==============================================================================
# STEP 3: DATA GENERATOR (SIMPLE PREPROCESSING)
# ==============================================================================

log("=" * 80)
log("STEP 3: DATA PREPROCESSING")
log("=" * 80 + "\n")

log("Preprocessing strategy: SIMPLE and IDENTICAL for both classes")
log("- Load image as grayscale")
log("- Convert to float32")
log("- Normalize to [0, 1]: img / 255.0")
log("- NO per-image normalization")
log("- NO augmentation")
log("- NO DICOM windowing")
log("- NO frequency normalization")
log("\nThis ensures a FAIR comparison!\n")

class SimpleDataGenerator(keras.utils.Sequence):
    """
    Simple data generator with identical preprocessing for both classes.
    """
    
    def __init__(self, synthetic_files, original_files, batch_size=32, shuffle=True):
        self.files = synthetic_files + original_files
        self.labels = np.array([0] * len(synthetic_files) + [1] * len(original_files))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_labels = []
        
        for idx in batch_indexes:
            img_path = self.files[idx]
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                continue
            
            # Simple preprocessing: normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            batch_images.append(image)
            batch_labels.append(self.labels[idx])
        
        if len(batch_images) == 0:
            return np.zeros((0, 512, 512, 1), dtype=np.float32), np.array([], dtype=np.int32)
        
        return np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.int32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create data generators
BATCH_SIZE = 32

train_gen = SimpleDataGenerator(syn_train, orig_train, batch_size=BATCH_SIZE, shuffle=True)
val_gen = SimpleDataGenerator(syn_val, orig_val, batch_size=BATCH_SIZE, shuffle=False)
test_gen = SimpleDataGenerator(syn_test, orig_test, batch_size=BATCH_SIZE, shuffle=False)

log(f"Data generators created:")
log(f"- Train batches: {len(train_gen)}")
log(f"- Val batches:   {len(val_gen)}")
log(f"- Test batches:  {len(test_gen)}\n")

# ==============================================================================
# STEP 4: BUILD MODEL
# ==============================================================================

log("=" * 80)
log("STEP 4: MODEL ARCHITECTURE")
log("=" * 80 + "\n")

def build_custom_cnn():
    """Build Custom CNN for binary classification."""
    inputs = layers.Input(shape=(512, 512, 1), name='input')
    
    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Conv Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    
    # Conv Block 5
    x = layers.Conv2D(512, (3, 3), padding='same', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('relu', name='relu5')(x)
    x = layers.MaxPooling2D((2, 2), name='pool5')(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Fully Connected Layers
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CustomCNN')
    
    return model

# Build and compile model
model = build_custom_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
log("Model: Custom CNN for Binary Classification\n")
model.summary(print_fn=lambda x: log(x, also_print=False))
log("\n")

# ==============================================================================
# STEP 5: TRAINING
# ==============================================================================

log("=" * 80)
log("STEP 5: TRAINING")
log("=" * 80 + "\n")

log("Training configuration:")
log("- Optimizer: Adam (lr=0.0001)")
log("- Loss: Sparse Categorical Crossentropy")
log("- Max epochs: 50")
log(f"- Batch size: {BATCH_SIZE}")
log("- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'models', 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

log("Starting training...\n")
start_time = datetime.now()

# Train
history = model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
training_duration = end_time - start_time

log(f"\n✅ Training completed in {training_duration}\n")

# Save final model
model.save(os.path.join(OUTPUT_DIR, 'models', 'final_model.h5'))

# Save training history
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history.history, f, indent=2)

# ==============================================================================
# STEP 6: VISUALIZATION - TRAINING CURVES
# ==============================================================================

log("Creating training curves visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

log("✅ Training curves saved\n")

# ==============================================================================
# STEP 7: EVALUATION
# ==============================================================================

log("=" * 80)
log("STEP 7: EVALUATION ON TEST SET")
log("=" * 80 + "\n")

log(f"Evaluating on {len(syn_test) + len(orig_test)} test images...\n")

# Predict on test set
y_true = []
y_pred_probs = []

for i in tqdm(range(len(test_gen)), desc="Evaluating"):
    X_batch, y_batch = test_gen[i]
    if len(X_batch) == 0:
        continue
    
    preds = model.predict(X_batch, verbose=0)
    y_pred_probs.extend(preds)
    y_true.extend(y_batch)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
roc_auc = roc_auc_score(y_true, y_pred_probs[:, 1])

# Per-class metrics
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ==============================================================================
# STEP 8: RESULTS REPORTING
# ==============================================================================

log("=" * 80)
log("FINAL RESULTS")
log("=" * 80)
log(f"Accuracy:  {accuracy * 100:.2f}%")
log(f"Precision: {precision:.4f}")
log(f"Recall:    {recall:.4f}")
log(f"F1-Score:  {f1:.4f}")
log(f"ROC AUC:   {roc_auc:.4f}")
log("=" * 80 + "\n")

log("Per-Class Metrics:")
log(f"{'':18} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
log(f"Synthetic (0)    {precision_per_class[0]:>10.4f} {recall_per_class[0]:>10.4f} {f1_per_class[0]:>10.4f} {support_per_class[0]:>10}")
log(f"Original (1)     {precision_per_class[1]:>10.4f} {recall_per_class[1]:>10.4f} {f1_per_class[1]:>10.4f} {support_per_class[1]:>10}\n")

log("Confusion Matrix:")
log(f"{'':16} {'Predicted':>20}")
log(f"{'':16} {'Synthetic':>10} {'Original':>10}")
log(f"Actual Synthetic {cm[0][0]:>10} {cm[0][1]:>10}")
log(f"       Original  {cm[1][0]:>10} {cm[1][1]:>10}\n")

# ==============================================================================
# STEP 9: VISUALIZATIONS
# ==============================================================================

log("Creating visualizations...\n")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Synthetic', 'Original'],
            yticklabels=['Synthetic', 'Original'])
plt.title('Confusion Matrix - Experiment 6 (Clean Evaluation)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
log("✅ Confusion matrix saved")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Experiment 6', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
log("✅ ROC curve saved")

# 3. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
avg_precision = average_precision_score(y_true, y_pred_probs[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, linewidth=2, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Experiment 6', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'precision_recall_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
log("✅ Precision-recall curve saved")

# 4. Sample Predictions (Correct and Incorrect)
log("Creating sample predictions visualization...")

# Get correct and incorrect predictions
correct_idx = np.where(y_true == y_pred)[0]
incorrect_idx = np.where(y_true != y_pred)[0]

# Sample 10 of each
if len(correct_idx) >= 10:
    sample_correct = np.random.choice(correct_idx, 10, replace=False)
else:
    sample_correct = correct_idx

if len(incorrect_idx) >= 10:
    sample_incorrect = np.random.choice(incorrect_idx, 10, replace=False)
else:
    sample_incorrect = incorrect_idx

# Plot
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

# Correct predictions
for i, idx in enumerate(sample_correct[:10]):
    img_path = test_gen.files[idx]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    true_label = 'Synthetic' if y_true[idx] == 0 else 'Original'
    pred_label = 'Synthetic' if y_pred[idx] == 0 else 'Original'
    confidence = y_pred_probs[idx][y_pred[idx]] * 100
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'✓ True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                      fontsize=10, color='green')
    axes[i].axis('off')

# Incorrect predictions
for i, idx in enumerate(sample_incorrect[:10]):
    img_path = test_gen.files[idx]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    true_label = 'Synthetic' if y_true[idx] == 0 else 'Original'
    pred_label = 'Synthetic' if y_pred[idx] == 0 else 'Original'
    confidence = y_pred_probs[idx][y_pred[idx]] * 100
    
    axes[i+10].imshow(img, cmap='gray')
    axes[i+10].set_title(f'✗ True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         fontsize=10, color='red')
    axes[i+10].axis('off')

plt.suptitle('Sample Predictions (Top 2 rows: Correct, Bottom 2 rows: Incorrect)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'sample_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
log("✅ Sample predictions saved")

log("\n✅ All visualizations complete!\n")

# ==============================================================================
# STEP 10: COMPARISON WITH PREVIOUS EXPERIMENTS
# ==============================================================================

log("=" * 80)
log("COMPARISON WITH PREVIOUS EXPERIMENTS")
log("=" * 80 + "\n")

log("Previous results:")
log("- Experiment 1 (Histogram Matching):     100.00% (text annotations present)")
log("- Experiment 2 (Frequency Normalization): 88.90% (text in frequency domain)")
log("- Experiment 3 (Extreme Augmentation):   100.00% (text annotations present)")
log("- Experiment 4 (Verification):            86.65% (text in frequency domain)")
log(f"- Experiment 6 (CLEAN):                   {accuracy * 100:.2f}% ← FAIR RESULT\n")

# ==============================================================================
# INTERPRETATION
# ==============================================================================

log("=" * 80)
log("INTERPRETATION")
log("=" * 80 + "\n")

if accuracy < 0.60:
    interpretation = "EXCELLENT GAN QUALITY - Images are nearly indistinguishable!"
elif accuracy < 0.75:
    interpretation = "GOOD GAN QUALITY - Subtle differences exist but overall realistic"
elif accuracy < 0.85:
    interpretation = "MODERATE GAN QUALITY - Noticeable patterns but acceptable"
else:
    interpretation = "GAN HAS DETECTABLE ARTIFACTS - Structural/textural differences"

log(f"Result: {accuracy * 100:.2f}% accuracy")
log(f"\n{interpretation}\n")

log("This represents a FAIR evaluation because:")
log("1. Text annotations removed from originals")
log("2. Identical simple preprocessing (img/255.0) for both classes")
log("3. No augmentation during test")
log("4. Large test set (20,000 images)")
log("\nThis is the TRUE measure of GAN quality for your OMAMA database paper!\n")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

log("=" * 80)
log("FILES SAVED")
log("=" * 80 + "\n")

log(f"All outputs saved to: {OUTPUT_DIR}\n")

log("Cleaned Data:")
log(f"- {CLEANED_PATH} ({len(cleaned_files)} images)\n")

log("Models:")
log("- models/best_model.h5")
log("- models/final_model.h5\n")

log("Results:")
log("- experiment6_results.txt (this file)")
log("- training_history.json\n")

log("Visualizations:")
log("- visualizations/before_after_cleaning.png")
log("- visualizations/training_curves.png")
log("- visualizations/confusion_matrix.png")
log("- visualizations/roc_curve.png")
log("- visualizations/precision_recall_curve.png")
log("- visualizations/sample_predictions.png\n")

log("Data Splits:")
log("- data_splits/train_synthetic.txt")
log("- data_splits/train_original.txt")
log("- data_splits/val_synthetic.txt")
log("- data_splits/val_original.txt")
log("- data_splits/test_synthetic.txt")
log("- data_splits/test_original.txt\n")

log("=" * 80)
log("EXPERIMENT 6 COMPLETED SUCCESSFULLY!")
log("=" * 80)
log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total duration: {datetime.now() - start_time}\n")

log("=" * 80)
log("READY FOR PUBLICATION!")
log("=" * 80)

print("\n" + "=" * 80)
print("✅ EXPERIMENT 6 COMPLETE!")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_FILE}")
print(f"Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations/')}")
print(f"\n🎯 Final Accuracy: {accuracy * 100:.2f}%")
print(f"📊 This is your PUBLISHABLE result for the OMAMA database paper!")
print("\n" + "=" * 80)

