# === Imports ===
import os
import cv2
import numpy as np
from custom_unet import CustomUNet  # use the correct file (custom_unet OR custom_unet_copy)
from tqdm.notebook import tqdm

# === Set dataset and output paths ===
train_image_folder = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/splits/train/images"
train_mask_folder  = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/splits/train/masks"
val_image_folder   = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/splits/val/images"
val_mask_folder    = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/splits/val/masks"
test_image_folder  = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/splits/test/images"
test_mask_folder   = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/splits/test/masks"
model_weights_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/model_weights_30.weights.h5"
prediction_folder  = "/hpcstor6/scratch01/a/a.kanamarlapudi001/datasets/predictions"

# === Create prediction folder if it doesn't exist ===
os.makedirs(prediction_folder, exist_ok=True)

# === Initialize the UNet model ===
unet = CustomUNet(
    img_height=512,
    img_width=512,
    batch_size=32,
    train_image_dir=train_image_folder,
    train_mask_dir=train_mask_folder,
    val_image_dir=val_image_folder,
    val_mask_dir=val_mask_folder,
    test_image_dir=test_image_folder,
    test_mask_dir=test_mask_folder,
    save_model_path=model_weights_path,
)

# === Get one batch to inspect input and mask shapes ===
train_gen = unet.custom_data_generator(
    unet.train_image_folder,
    unet.train_mask_folder,
    batch_size=1,
    img_height=unet.img_height,
    img_width=unet.img_width,
)

sample_images, sample_masks = next(train_gen)
print("Input image shape:", sample_images.shape)
print("Mask shape:", sample_masks.shape)

# === (Optional) Train the model ===
history = unet.compile_and_train(epochs=30)

# === Save model weights ===
model_weights_path = unet.model.save_weights(unet.save_model_path)

print(f"Training complete. Model saved to {model_weights_path}")
