import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import Sequence
from sklearn.metrics import jaccard_score, accuracy_score

# ---------------------------------------------
# Configuration
# ---------------------------------------------
BASE_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/cancer_classifier_1024"
TRAIN_IMG_DIR = f"{BASE_DIR}/split/train/images"
TRAIN_MASK_DIR = f"{BASE_DIR}/split/train/masks"
VAL_IMG_DIR   = f"{BASE_DIR}/split/val/images"
VAL_MASK_DIR  = f"{BASE_DIR}/split/val/masks"
TEST_IMG_DIR  = f"{BASE_DIR}/split/test/images"
TEST_MASK_DIR = f"{BASE_DIR}/split/test/masks"
WEIGHT_PATH   = f"{BASE_DIR}/weights/best_model.h5"
LOG_PATH      = f"{BASE_DIR}/training_output.txt"

IMG_SIZE = 1024
BATCH_SIZE = 1
EPOCHS = 30

# ---------------------------------------------
# Data Generator
# ---------------------------------------------
class CancerSegmentationDataset(Sequence):
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
# U-Net Model
# ---------------------------------------------
def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 1024)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# ---------------------------------------------
# Evaluation Metrics
# ---------------------------------------------
def evaluate_model(model, generator):
    y_true, y_pred = [], []
    for X_batch, y_batch in generator:
        preds = model.predict(X_batch) > 0.5
        y_true.extend(y_batch.flatten())
        y_pred.extend(preds.flatten())

    acc = accuracy_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred, average='binary')
    return acc, iou

# ---------------------------------------------
# Main Training Logic
# ---------------------------------------------
def main():
    print("🔄 Initializing data generators...")
    train_gen = CancerSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_gen = CancerSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR)
    test_gen = CancerSegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR)

    print("📐 Building U-Net model...")
    model = build_unet()
    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)

    checkpoint = callbacks.ModelCheckpoint(WEIGHT_PATH, save_best_only=True, monitor='val_loss')
    earlystop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reducelr = callbacks.ReduceLROnPlateau(patience=3, factor=0.5)

    print("🚀 Starting training...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
              callbacks=[checkpoint, earlystop, reducelr], verbose=1)

    print("✅ Training complete. Loading best model...")
    model.load_weights(WEIGHT_PATH)

    print("🧪 Evaluating on test set...")
    acc, iou = evaluate_model(model, test_gen)

    print("\n📊 Final Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"IoU: {iou:.4f}")

    with open(LOG_PATH, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"IoU: {iou:.4f}\n")

if __name__ == "__main__":
    main()
