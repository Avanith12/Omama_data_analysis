import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from glob import glob
from tqdm import tqdm

# === Paths ===
MODEL_PATH = "/hpcstor6/scratch01/a/a.kanamarlapudi001/downloads/classifier/model_best.h5"
IMAGE_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/cancer_classifier_1024/2d_resized_1024/images"
META_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/cancer_classifier_1024/2d_resized_1024/metadata"
REPORT_PATH = "evaluation_report.txt"

# === Data generator
class TestNPZGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, labels, batch_size=32):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X = []
        for id_ in batch_ids:
            path = os.path.join(IMAGE_DIR, id_ + ".npz")
            data = np.load(path)["data"].astype(np.float32)
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) - 0.5
            X.append(data[..., np.newaxis])
        return np.array(X), np.array(batch_labels)

# === Load metadata
def get_labelled_ids():
    ids, labels = [], []
    for meta_file in tqdm(glob(os.path.join(META_DIR, "*.json")), desc="📄 Scanning metadata"):
        with open(meta_file) as f:
            meta = json.load(f)
            label = meta.get("label")
            if label == "IndexCancer":
                labels.append(1)
            elif label == "NonCancer":
                labels.append(0)
            else:
                continue
            ids.append(os.path.splitext(os.path.basename(meta_file))[0])
    return np.array(ids), np.array(labels)

# === Main
if __name__ == "__main__":
    ids, labels = get_labelled_ids()
    test_gen = TestNPZGenerator(ids, labels, batch_size=32)

    print("🔍 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("🔮 Predicting...")
    y_probs = model.predict(test_gen, verbose=1)
    y_pred = (y_probs > 0.5).astype(int).flatten()
    y_true = labels[:len(y_pred)]

    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n📝 Classification Report:")
    report = classification_report(y_true, y_pred)
    print(report)

    # Save to file
    with open(REPORT_PATH, "w") as f:
        f.write("📊 Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n📝 Classification Report:\n")
        f.write(report)

    print(f"\n✅ Results saved to: {REPORT_PATH}")
