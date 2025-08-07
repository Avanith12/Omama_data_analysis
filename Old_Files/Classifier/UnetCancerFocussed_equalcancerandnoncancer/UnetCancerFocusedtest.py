import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from tqdm import tqdm

# ---------------------------------------------
# Configuration
# ---------------------------------------------
BASE_DIR = "/hpcstor6/scratch01/a/a.kanamarlapudi001/cancer_classifier_1024x1024"
IMG_DIR = f"{BASE_DIR}/split/test/images"
MASK_DIR = f"{BASE_DIR}/split/test/masks"
WEIGHT_PATH = f"{BASE_DIR}/weights/cancer_focused_model.h5"
RESULTS_PATH = f"{BASE_DIR}/test_results_cancer_focused.txt"
IMG_SIZE = 1024
BATCH_SIZE = 1

# ---------------------------------------------
# Data Generator
# ---------------------------------------------
class CancerSegmentationDataset(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=BATCH_SIZE):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.file_ids = sorted([f.replace(".npz", "") for f in os.listdir(image_dir) if f.endswith(".npz")])
        # Optional: keep only those with existing masks
        self.file_ids = [fid for fid in self.file_ids if os.path.exists(os.path.join(mask_dir, fid + ".png"))]

    def __len__(self):
        return int(np.ceil(len(self.file_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.file_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for file_id in batch_ids:
            img_path = os.path.join(self.image_dir, file_id + ".npz")
            mask_path = os.path.join(self.mask_dir, file_id + ".png")

            img = np.load(img_path)
            img = img[img.files[0]].astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.expand_dims(img, axis=-1)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

# ---------------------------------------------
# Evaluation
# ---------------------------------------------
def evaluate_model_on_the_fly(model, generator):
    y_true, y_pred = [], []

    for i in tqdm(range(len(generator)), desc="🔍 Evaluating"):
        try:
            X_batch, y_batch = generator[i]
            preds = model.predict(X_batch, verbose=0) > 0.5
            y_true.extend(y_batch.flatten())
            y_pred.extend(preds.flatten())
        except Exception as e:
            print(f"⚠️ Skipping batch {i}: {str(e)}")

    if len(y_true) == 0:
        raise ValueError(" No predictions were collected. Check your data generator.")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    return acc, f1, iou

# ---------------------------------------------
# Main Logic
# ---------------------------------------------
if __name__ == "__main__":
    print(" Loading model...")
    model = load_model(WEIGHT_PATH, compile=False)

    print(" Preparing test data...")
    test_generator = CancerSegmentationDataset(IMG_DIR, MASK_DIR)
    print(f" Total test batches: {len(test_generator)}")

    print("🔍 Starting evaluation...")
    acc, f1, iou = evaluate_model_on_the_fly(model, test_generator)

    print("\n Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

    with open(RESULTS_PATH, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"IoU: {iou:.4f}\n")

    print(f" Results saved to: {RESULTS_PATH}")
