import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import cv2
from unetmodel import build_unet

# --- Configuration ---
BASE_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/cancer_classifier_1024"
TEST_IMG_DIR = f"{BASE_DIR}/split/test/images"
TEST_MASK_DIR = f"{BASE_DIR}/split/test/masks"
WEIGHT_PATH = f"{BASE_DIR}/weights/best_model.h5"
RESULTS_TXT = f"{BASE_DIR}/test_results_original.txt"
IMG_SIZE = 1024
BATCH_SIZE = 1

# --- Load U-Net ---
model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
model.load_weights(WEIGHT_PATH)
print("✅ Model loaded from best_model.h5")

# --- Define Data Generator ---
class CancerSegmentationDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=BATCH_SIZE):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.file_ids = sorted([f.replace(".npz", "") for f in os.listdir(image_dir) if f.endswith(".npz")])

    def __len__(self):
        return int(np.ceil(len(self.file_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.file_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for file_id in batch_ids:
            img_path = os.path.join(self.image_dir, file_id + ".npz")
            mask_path = os.path.join(self.mask_dir, file_id + ".png")

            img_npz = np.load(img_path)
            img = img_npz[img_npz.files[0]].astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.expand_dims(img, axis=-1)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

# --- Evaluation ---
test_gen = CancerSegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR)
all_ious = []
all_accuracies = []
all_f1s = []

with open(RESULTS_TXT, "w") as f:
    for i in tqdm(range(len(test_gen)), desc="🔍 Predicting on test set"):
        x_batch, y_batch = test_gen[i]
        preds = model.predict(x_batch)

        for j in range(len(x_batch)):
            img_id = test_gen.file_ids[i * BATCH_SIZE + j]
            pred = (preds[j, :, :, 0] > 0.5).astype(np.uint8)
            gt = (y_batch[j, :, :, 0] > 0.5).astype(np.uint8)

            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            iou = intersection / union if union != 0 else 1.0

            tp = np.logical_and(pred == 1, gt == 1).sum()
            tn = np.logical_and(pred == 0, gt == 0).sum()
            fp = np.logical_and(pred == 1, gt == 0).sum()
            fn = np.logical_and(pred == 0, gt == 1).sum()

            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) != 0 else 1.0

            all_ious.append(iou)
            all_accuracies.append(accuracy)
            all_f1s.append(f1_score)

            f.write(f"{img_id} | IoU: {iou:.4f} | Acc: {accuracy:.4f} | F1: {f1_score:.4f}\n")

    mean_iou = np.mean(all_ious)
    mean_acc = np.mean(all_accuracies)
    mean_f1 = np.mean(all_f1s)

    f.write("\n--- Averages across test set ---\n")
    f.write(f"Mean IoU: {mean_iou:.4f}\n")
    f.write(f"Mean Accuracy: {mean_acc:.4f}\n")
    f.write(f"Mean F1 Score: {mean_f1:.4f}\n")

print(f"\n✅ Predictions complete. Results saved to: {RESULTS_TXT}")
