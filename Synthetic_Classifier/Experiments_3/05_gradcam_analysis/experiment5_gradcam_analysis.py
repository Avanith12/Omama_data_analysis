"""
EXPERIMENT 5: GRAD-CAM ANALYSIS
================================
Purpose: Use Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize 
what the classifier is actually looking at when distinguishing synthetic from original images.

This will definitively answer:
- Is the model using text annotations as a shortcut?
- Is the model focusing on tissue patterns/textures?
- What structural differences does the model detect?

Author: Avanith
Date: October 27, 2025
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime
import random

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPERIMENT 5: GRAD-CAM ANALYSIS")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path("/home/a.kanamarlapudi001/projects/omama-proj/_EXPERIMENTS/SYNTHETIC/Avanith/MoreExperiments")
EXP4_DIR = BASE_DIR / "experiment4_verification"
OUTPUT_DIR = BASE_DIR / "experiment5_gradcam"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model path (from Experiment 4)
MODEL_PATH = EXP4_DIR / "exp4_best_model.h5"

# Data paths (frequency-normalized from Experiment 4)
SYNTHETIC_DIR = EXP4_DIR / "freq_normalized_synthetic"
ORIGINAL_DIR = EXP4_DIR / "freq_normalized_original"

# Analysis parameters
NUM_SAMPLES_PER_CATEGORY = 10  # Number of images to analyze per category
IMG_SIZE = (512, 512)

# Class labels
CLASS_LABELS = {0: 'Synthetic', 1: 'Original'}

print(f"Output directory: {OUTPUT_DIR}")
print(f"Model path: {MODEL_PATH}")
print()

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading trained model from Experiment 4...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

def get_last_conv_layer_name(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image
    
    Args:
        img_array: Input image (preprocessed, shape: (1, H, W, C))
        model: Trained model
        last_conv_layer_name: Name of the last conv layer
        pred_index: Class index to visualize (None = predicted class)
    
    Returns:
        heatmap: Grad-CAM heatmap (normalized to [0, 1])
    """
    # Create a model that maps the input to the activations of the last conv layer
    # and the output predictions
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the predicted class with respect to 
    # the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Extract gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the conv output by "how important this channel is"
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # The channel-wise mean of the resulting feature map is our heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    
    return heatmap

def create_gradcam_overlay(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Create an overlay of the Grad-CAM heatmap on the original image
    
    Args:
        img: Original image (H, W) or (H, W, 1) grayscale
        heatmap: Grad-CAM heatmap (h, w) normalized to [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
    
    Returns:
        overlay: RGB image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert grayscale image to RGB for overlay
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()
    
    # Create overlay
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay, heatmap_colored

# ============================================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================================

last_conv_layer_name = get_last_conv_layer_name(model)
if last_conv_layer_name is None:
    print("❌ Error: Could not find a convolutional layer in the model!")
    exit(1)

print(f"Using last convolutional layer: '{last_conv_layer_name}'")
print()

# ============================================================================
# LOAD AND PREPARE TEST IMAGES
# ============================================================================

print("Loading test images...")

# Get all image paths
synthetic_paths = sorted(list(SYNTHETIC_DIR.glob("*.png")))
original_paths = sorted(list(ORIGINAL_DIR.glob("*.png")))

print(f"Found {len(synthetic_paths)} synthetic images")
print(f"Found {len(original_paths)} original images")
print()

# Randomly sample images
random.seed(42)
synthetic_sample = random.sample(synthetic_paths, min(NUM_SAMPLES_PER_CATEGORY * 2, len(synthetic_paths)))
original_sample = random.sample(original_paths, min(NUM_SAMPLES_PER_CATEGORY * 2, len(original_paths)))

def load_and_preprocess_image(img_path):
    """Load and preprocess image for model input"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img_normalized = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

# ============================================================================
# PREDICT AND CLASSIFY IMAGES
# ============================================================================

print("Running predictions to categorize images...")

# Categories for analysis
categories = {
    'synthetic_correct': [],      # Synthetic correctly classified as Synthetic
    'synthetic_wrong': [],         # Synthetic wrongly classified as Original
    'original_correct': [],        # Original correctly classified as Original
    'original_wrong': []           # Original wrongly classified as Synthetic
}

# Process synthetic images
for img_path in synthetic_sample:
    img, img_array = load_and_preprocess_image(img_path)
    pred = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    if pred_class == 0:  # Correctly classified as Synthetic
        categories['synthetic_correct'].append({
            'path': img_path,
            'img': img,
            'img_array': img_array,
            'pred': pred[0],
            'confidence': confidence
        })
    else:  # Wrongly classified as Original
        categories['synthetic_wrong'].append({
            'path': img_path,
            'img': img,
            'img_array': img_array,
            'pred': pred[0],
            'confidence': confidence
        })

# Process original images
for img_path in original_sample:
    img, img_array = load_and_preprocess_image(img_path)
    pred = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    if pred_class == 1:  # Correctly classified as Original
        categories['original_correct'].append({
            'path': img_path,
            'img': img,
            'img_array': img_array,
            'pred': pred[0],
            'confidence': confidence
        })
    else:  # Wrongly classified as Synthetic
        categories['original_wrong'].append({
            'path': img_path,
            'img': img,
            'img_array': img_array,
            'pred': pred[0],
            'confidence': confidence
        })

print("Classification summary:")
print(f"  Synthetic correctly classified: {len(categories['synthetic_correct'])}")
print(f"  Synthetic wrongly classified:   {len(categories['synthetic_wrong'])}")
print(f"  Original correctly classified:  {len(categories['original_correct'])}")
print(f"  Original wrongly classified:    {len(categories['original_wrong'])}")
print()

# ============================================================================
# GENERATE GRAD-CAM VISUALIZATIONS
# ============================================================================

print("Generating Grad-CAM visualizations...")
print("This may take several minutes...")
print()

results = {}

for category_name, samples in categories.items():
    if len(samples) == 0:
        print(f"⚠️  Skipping {category_name} (no samples)")
        continue
    
    print(f"Processing {category_name}: {len(samples)} samples")
    
    # Take top NUM_SAMPLES_PER_CATEGORY samples
    samples_to_analyze = samples[:NUM_SAMPLES_PER_CATEGORY]
    
    category_results = []
    
    for idx, sample in enumerate(samples_to_analyze):
        img = sample['img']
        img_array = sample['img_array']
        pred = sample['pred']
        pred_class = np.argmax(pred)
        confidence = sample['confidence']
        
        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_class)
        
        # Create overlay
        overlay, heatmap_colored = create_gradcam_overlay(img, heatmap, alpha=0.5)
        
        category_results.append({
            'img': img,
            'heatmap': heatmap,
            'overlay': overlay,
            'heatmap_colored': heatmap_colored,
            'pred_class': pred_class,
            'confidence': confidence,
            'pred': pred
        })
    
    results[category_name] = category_results

print("✅ Grad-CAM generation complete!")
print()

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

print("Creating visualization plots...")

# 1. Create grid for each category
for category_name, category_results in results.items():
    if len(category_results) == 0:
        continue
    
    n_samples = len(category_results)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(category_results):
        # Original image
        axes[idx, 0].imshow(result['img'], cmap='gray')
        axes[idx, 0].set_title(f"Original\n{CLASS_LABELS[result['pred_class']]} ({result['confidence']:.1%})")
        axes[idx, 0].axis('off')
        
        # Heatmap only
        axes[idx, 1].imshow(result['heatmap'], cmap='jet')
        axes[idx, 1].set_title("Grad-CAM Heatmap")
        axes[idx, 1].axis('off')
        
        # Heatmap colored
        axes[idx, 2].imshow(result['heatmap_colored'])
        axes[idx, 2].set_title("Heatmap (Colored)")
        axes[idx, 2].axis('off')
        
        # Overlay
        axes[idx, 3].imshow(result['overlay'])
        axes[idx, 3].set_title("Overlay")
        axes[idx, 3].axis('off')
    
    plt.suptitle(f"Grad-CAM Analysis: {category_name.replace('_', ' ').title()}", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"gradcam_{category_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: gradcam_{category_name}.png")

# 2. Create summary comparison plot
print("\nCreating summary comparison plot...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

row_titles = [
    "Synthetic (Correct)",
    "Synthetic (Misclassified)", 
    "Original (Correct)",
    "Original (Misclassified)"
]

category_keys = [
    'synthetic_correct',
    'synthetic_wrong',
    'original_correct',
    'original_wrong'
]

for row_idx, (category_key, row_title) in enumerate(zip(category_keys, row_titles)):
    if category_key not in results or len(results[category_key]) == 0:
        continue
    
    # Take first 5 samples for summary
    samples_for_row = results[category_key][:5]
    
    for col_idx, result in enumerate(samples_for_row):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(result['overlay'])
        ax.set_title(f"{CLASS_LABELS[result['pred_class']]}\n{result['confidence']:.1%}", 
                     fontsize=10)
        ax.axis('off')
    
    # Add row title
    fig.text(0.02, 0.875 - row_idx * 0.22, row_title, 
             fontsize=12, fontweight='bold', rotation=90, 
             va='center', ha='center')

plt.suptitle("Grad-CAM Summary: What is the Classifier Looking At?", 
             fontsize=18, fontweight='bold')

output_path = OUTPUT_DIR / "gradcam_summary.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: gradcam_summary.png")

# ============================================================================
# ANALYZE HEATMAP STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("HEATMAP ANALYSIS")
print("=" * 80)
print()

analysis_results = {}

for category_name, category_results in results.items():
    if len(category_results) == 0:
        continue
    
    heatmaps = [r['heatmap'] for r in category_results]
    
    # Compute statistics
    mean_heatmap = np.mean(heatmaps, axis=0)
    max_activation = np.max([np.max(h) for h in heatmaps])
    mean_activation = np.mean([np.mean(h) for h in heatmaps])
    
    # Find regions of high activation (top 10%)
    high_activation_threshold = 0.7
    high_activation_pixels = [np.sum(h > high_activation_threshold) / h.size for h in heatmaps]
    mean_high_activation = np.mean(high_activation_pixels) * 100  # percentage
    
    analysis_results[category_name] = {
        'mean_heatmap': mean_heatmap,
        'max_activation': max_activation,
        'mean_activation': mean_activation,
        'high_activation_percentage': mean_high_activation
    }
    
    print(f"{category_name.replace('_', ' ').title()}:")
    print(f"  Max activation:        {max_activation:.3f}")
    print(f"  Mean activation:       {mean_activation:.3f}")
    print(f"  High activation area:  {mean_high_activation:.2f}%")
    print()

# ============================================================================
# CREATE MEAN HEATMAP VISUALIZATION
# ============================================================================

print("Creating mean heatmap comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for idx, (category_name, analysis) in enumerate(analysis_results.items()):
    if idx >= 4:
        break
    
    mean_heatmap = analysis['mean_heatmap']
    
    im = axes[idx].imshow(mean_heatmap, cmap='jet')
    axes[idx].set_title(f"{category_name.replace('_', ' ').title()}\nMean Activation: {analysis['mean_activation']:.3f}", 
                        fontsize=12, fontweight='bold')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.suptitle("Mean Grad-CAM Heatmaps Across Categories", fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = OUTPUT_DIR / "mean_heatmaps.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: mean_heatmaps.png")

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================

print("\nSaving detailed results...")

results_text = []
results_text.append("=" * 80)
results_text.append("EXPERIMENT 5: GRAD-CAM ANALYSIS RESULTS")
results_text.append("=" * 80)
results_text.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
results_text.append(f"Output directory: {OUTPUT_DIR}")
results_text.append("")
results_text.append("=" * 80)
results_text.append("PURPOSE")
results_text.append("=" * 80)
results_text.append("")
results_text.append("Grad-CAM (Gradient-weighted Class Activation Mapping) reveals what regions")
results_text.append("of the image the classifier focuses on when making predictions.")
results_text.append("")
results_text.append("Key questions answered:")
results_text.append("1. Does the classifier exploit text annotations in original images?")
results_text.append("2. What structural/textural features does the classifier use?")
results_text.append("3. Are misclassified images focusing on unusual regions?")
results_text.append("")
results_text.append("=" * 80)
results_text.append("SAMPLE COUNTS")
results_text.append("=" * 80)
results_text.append("")
for category_name, samples in categories.items():
    results_text.append(f"{category_name.replace('_', ' ').title():.<40} {len(samples):>5}")
results_text.append("")
results_text.append("=" * 80)
results_text.append("HEATMAP STATISTICS")
results_text.append("=" * 80)
results_text.append("")
for category_name, analysis in analysis_results.items():
    results_text.append(f"{category_name.replace('_', ' ').title()}:")
    results_text.append(f"  Maximum activation:    {analysis['max_activation']:>8.4f}")
    results_text.append(f"  Mean activation:       {analysis['mean_activation']:>8.4f}")
    results_text.append(f"  High activation area:  {analysis['high_activation_percentage']:>7.2f}%")
    results_text.append("")

results_text.append("=" * 80)
results_text.append("INTERPRETATION GUIDE")
results_text.append("=" * 80)
results_text.append("")
results_text.append("How to read Grad-CAM visualizations:")
results_text.append("")
results_text.append("1. HEATMAP COLORS:")
results_text.append("   - Red/Yellow: High importance (model focuses here)")
results_text.append("   - Blue/Purple: Low importance (model ignores)")
results_text.append("")
results_text.append("2. TEXT ANNOTATION BIAS:")
results_text.append("   If model exploits text annotations, you'll see:")
results_text.append("   - Bright red/yellow on text regions (corners with 'LCC', 'RCC', etc.)")
results_text.append("   - Blue/dark on actual breast tissue")
results_text.append("")
results_text.append("3. GENUINE MEDICAL LEARNING:")
results_text.append("   If model learns medical content, you'll see:")
results_text.append("   - Red/yellow distributed across breast tissue")
results_text.append("   - Focus on texture patterns, edges, structures")
results_text.append("   - Blue/dark on text annotations (ignoring them)")
results_text.append("")
results_text.append("4. COMPARISON ACROSS CATEGORIES:")
results_text.append("   - Synthetic (Correct): Where model sees 'synthetic-ness'")
results_text.append("   - Original (Correct): Where model sees 'original-ness'")
results_text.append("   - Misclassified: What confuses the model")
results_text.append("")
results_text.append("=" * 80)
results_text.append("FILES GENERATED")
results_text.append("=" * 80)
results_text.append("")
results_text.append("Visualizations:")
results_text.append("  - gradcam_synthetic_correct.png")
results_text.append("  - gradcam_synthetic_wrong.png")
results_text.append("  - gradcam_original_correct.png")
results_text.append("  - gradcam_original_wrong.png")
results_text.append("  - gradcam_summary.png (overview of all categories)")
results_text.append("  - mean_heatmaps.png (average activation patterns)")
results_text.append("")
results_text.append("Results:")
results_text.append("  - experiment5_results.txt (this file)")
results_text.append("")
results_text.append("=" * 80)
results_text.append("NEXT STEPS")
results_text.append("=" * 80)
results_text.append("")
results_text.append("After reviewing the Grad-CAM visualizations:")
results_text.append("")
results_text.append("1. Check if red/yellow regions focus on:")
results_text.append("   → Text annotations (corners) = TEXT BIAS confirmed")
results_text.append("   → Tissue patterns (center) = Genuine learning")
results_text.append("   → Image borders/edges = Artifact bias")
results_text.append("")
results_text.append("2. Compare 'correct' vs 'misclassified' images:")
results_text.append("   → Do misclassified originals lack text? = TEXT BIAS")
results_text.append("   → Do they have unusual textures? = Medical content")
results_text.append("")
results_text.append("3. Based on findings:")
results_text.append("   → If TEXT BIAS: Need proper text removal")
results_text.append("   → If GENUINE: GAN has detectable structural differences")
results_text.append("")
results_text.append("=" * 80)
results_text.append("EXPERIMENT 5 COMPLETE")
results_text.append("=" * 80)
results_text.append("")

# Write results to file
results_file = OUTPUT_DIR / "experiment5_results.txt"
with open(results_file, 'w') as f:
    f.write('\n'.join(results_text))

print(f"✅ Saved: experiment5_results.txt")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("EXPERIMENT 5 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()
print(f"All results saved to: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  📊 gradcam_synthetic_correct.png")
print("  📊 gradcam_synthetic_wrong.png")
print("  📊 gradcam_original_correct.png")
print("  📊 gradcam_original_wrong.png")
print("  📊 gradcam_summary.png")
print("  📊 mean_heatmaps.png")
print("  📄 experiment5_results.txt")
print()
print("=" * 80)
print("🔍 WHAT TO LOOK FOR IN THE VISUALIZATIONS:")
print("=" * 80)
print()
print("1. TEXT BIAS INDICATORS:")
print("   ❌ Red/yellow focus on image corners (where 'LCC', 'RCC' text is)")
print("   ❌ Blue/dark on breast tissue (ignoring medical content)")
print("   ❌ High activation only on text-containing images")
print()
print("2. GENUINE LEARNING INDICATORS:")
print("   ✅ Red/yellow distributed across breast tissue")
print("   ✅ Focus on texture, density patterns, structures")
print("   ✅ Blue/dark on text annotations (ignoring them)")
print()
print("3. MISCLASSIFICATION PATTERNS:")
print("   🔍 Compare heatmaps of misclassified vs correctly classified")
print("   🔍 Do misclassified originals lack text annotations?")
print("   🔍 Do misclassified synthetics have unusual patterns?")
print()
print("=" * 80)
print("Review the images and let's discuss what the model is actually seeing!")
print("=" * 80)

