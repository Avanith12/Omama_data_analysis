import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# Load model
model = keras.models.load_model('exp2_best_model.h5')

# Load test file lists
with open('data_splits/exp2_test_syn.txt', 'r') as f:
    syn_test = [line.strip() for line in f.readlines()]
with open('data_splits/exp2_test_orig.txt', 'r') as f:
    orig_test = [line.strip() for line in f.readlines()]

print(f"Loading {len(syn_test)} synthetic and {len(orig_test)} original test images...")

# Load test images
test_images = []
test_labels = []

for f in syn_test:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        test_images.append(img)
        test_labels.append(0)

for f in orig_test:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        test_images.append(img)
        test_labels.append(1)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(f"Loaded {len(test_images)} test images")
print("Evaluating model...")

# Predict
predictions = model.predict(test_images, batch_size=16, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_classes)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_classes, average='weighted')
cm = confusion_matrix(test_labels, predicted_classes)

# Per-class metrics
precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
    test_labels, predicted_classes, average=None
)

# ROC AUC
test_labels_binary = keras.utils.to_categorical(test_labels, 2)
auc = roc_auc_score(test_labels_binary, predictions)

print("\n" + "="*70)
print("EXPERIMENT 2: FREQUENCY NORMALIZATION - RESULTS")
print("="*70)
print(f"\nAccuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("\nPer-Class Metrics:")
print(f"Synthetic (0) - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
print(f"Original  (1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Synthetic  Original")
print(f"Actual Synthetic   {cm[0,0]:,}       {cm[0,1]:,}")
print(f"       Original    {cm[1,0]:,}       {cm[1,1]:,}")
print("="*70)

# Save results
with open('experiment2_final_results.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("EXPERIMENT 2: FREQUENCY NORMALIZATION - FINAL RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"\nTest Set Size: {len(test_images)} images\n")
    f.write(f"Accuracy:  {accuracy*100:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write(f"ROC AUC:   {auc:.4f}\n")
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"Synthetic correctly classified: {cm[0,0]:,}\n")
    f.write(f"Synthetic misclassified: {cm[0,1]:,}\n")
    f.write(f"Original correctly classified: {cm[1,1]:,}\n")
    f.write(f"Original misclassified: {cm[1,0]:,}\n")

print("\nResults saved to experiment2_final_results.txt")
