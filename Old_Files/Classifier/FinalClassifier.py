import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

# === Description of What This Code Does ===
# This script builds a cancer classifier that:
# 1. Uses a Keras Sequence generator to load .npz and .json files in batches.
# 2. Trains on all images labeled "IndexCancer" (1) or "NonCancer" (0).
# 3. Performs a 60/20/20 split (train/val/test).
# 4. Trains using an A100-compatible setup (mixed precision + large batch).
# 5. Saves best model to given path and prints training progress with tqdm.

# === PATH SETUP ===
IMAGE_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/cancer_classifier_1024/2d_resized_1024/images"
META_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/cancer_classifier_1024/2d_resized_1024/metadata"
SAVE_DIR = "/hpcstor6/scratch01/a/a.kanamarlapudi001/downloads/classifier"
MODEL_PATH = os.path.join(SAVE_DIR, "model_best.h5")
os.makedirs(SAVE_DIR, exist_ok=True)

# === USE MIXED PRECISION FOR A100 ===
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# === DATA SPLITTING ===
def get_labelled_ids():
    ids = []
    labels = []
    for json_path in tqdm(glob(os.path.join(META_DIR, "*.json")), desc="🔍 Scanning metadata"):
        with open(json_path, "r") as f:
            meta = json.load(f)
            label = meta.get("label")
            if label == "IndexCancer":
                labels.append(1)
            elif label == "NonCancer":
                labels.append(0)
            else:
                continue
            ids.append(os.path.splitext(os.path.basename(json_path))[0])
    return ids, labels

# === DATA GENERATOR ===
class NPZGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, labels, batch_size=32, shuffle=True):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.ids))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.ids[i] for i in batch_idx]
        batch_labels = [self.labels[i] for i in batch_idx]

        X = []
        for id_ in batch_ids:
            npz_path = os.path.join(IMAGE_DIR, id_ + ".npz")
            data = np.load(npz_path)["data"].astype(np.float32)
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) - 0.5
            X.append(data[..., np.newaxis])
        return np.array(X), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# === MODEL ===
def build_model(input_shape=(1024, 1024, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === TRAINING ===
def train():
    ids, labels = get_labelled_ids()
    ids = np.array(ids)
    labels = np.array(labels)
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(ids, labels, test_size=0.4, stratify=labels, random_state=42)
    val_ids, test_ids, val_labels, test_labels = train_test_split(temp_ids, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    print(f"\n🧾 Dataset splits:")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_gen = NPZGenerator(train_ids, train_labels, batch_size=32)
    val_gen = NPZGenerator(val_ids, val_labels, batch_size=32, shuffle=False)
    test_gen = NPZGenerator(test_ids, test_labels, batch_size=32, shuffle=False)

    model = build_model()

    cb = [
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    print("\n🚀 Starting training...\n")
    model.fit(train_gen, validation_data=val_gen, epochs=40, callbacks=cb, verbose=1)
    print("\n✅ Training complete.")

    print("\n🔍 Evaluating on test set...")
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"📁 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
