import numpy as np
import cv2
import os
import json
from tqdm import tqdm

original_npz_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/2d_resized_512/images"
original_metadata_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/2d_resized_512/metadata"
synthetic_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/train"

output_original = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/data_1000samples/original_no_norm"
output_synthetic = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/data_1000samples/synthetic_fd"

os.makedirs(output_original, exist_ok=True)
os.makedirs(output_synthetic, exist_ok=True)

def extract_value(param, default):
    if isinstance(param, list):
        return float(param[0]) if len(param) > 0 else default
    return float(param) if param is not None else default

def apply_window_level(img, window_center, window_width, rescale_intercept=0, rescale_slope=1):
    img = img.astype(np.float32)
    img = img * rescale_slope + rescale_intercept
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    img = np.clip(img, window_min, window_max)
    img = (img - window_min) / (window_max - window_min)
    img = (img * 255).astype(np.uint8)
    return img

def detect_breast_region(img):
    _, breast_mask = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(breast_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        breast_mask = (labels == largest_label).astype(np.uint8) * 255
    return breast_mask

def detect_breast_side(img):
    breast_mask = detect_breast_region(img)
    h, w = img.shape
    mid_x = w // 2
    left_half = breast_mask[:, :mid_x]
    right_half = breast_mask[:, mid_x:]
    left_tissue = np.sum(left_half > 0)
    right_tissue = np.sum(right_half > 0)
    return 'left' if left_tissue > right_tissue else 'right'

def find_left_start(img):
    for col in range(img.shape[1]):
        if img[:, col].mean() > 5:
            return col
    return 0

def find_top_bottom(img):
    top = None
    for row in range(img.shape[0]):
        if img[row, :].mean() > 5:
            top = row
            break
    bottom = None
    for row in range(img.shape[0]-1, -1, -1):
        if img[row, :].mean() > 5:
            bottom = row
            break
    return top, bottom

def process_synthetic_image(img):
    top, bottom = find_top_bottom(img)
    left_start = find_left_start(img)
    content_rows = img[top:bottom+1, :]
    cropped_w = content_rows[:, left_start:]
    
    if cropped_w.shape[1] < 512:
        pad_right = 512 - cropped_w.shape[1]
        cropped_w = np.pad(cropped_w, ((0, 0), (0, pad_right)), mode='constant', constant_values=0)
    
    if cropped_w.shape[0] > 512:
        result = cv2.resize(cropped_w, (512, 512), interpolation=cv2.INTER_AREA)
    elif cropped_w.shape[0] < 512:
        total_pad = 512 - cropped_w.shape[0]
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        result = np.pad(cropped_w, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    else:
        result = cropped_w
    
    return result

def get_npz_files(folder_path):
    files = os.listdir(folder_path)
    npz_files = []
    for f in files:
        if f.endswith('.npz'):
            npz_files.append(os.path.join(folder_path, f))
    return sorted(npz_files)

def get_png_files(folder_path):
    files = os.listdir(folder_path)
    png_files = []
    for f in files:
        if f.endswith('.png'):
            png_files.append(os.path.join(folder_path, f))
    return sorted(png_files)

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def select_files(cancer_files, noncancer_files):
    cancer_files = cancer_files[:500]
    noncancer_files = noncancer_files[:500]
    return cancer_files + noncancer_files

all_npz_files = get_npz_files(original_npz_path)

cancer_files = []
noncancer_files = []

for npz_file in all_npz_files:
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    metadata_file = os.path.join(original_metadata_path, base_name + ".json")
    
    if os.path.exists(metadata_file):
        metadata = load_metadata(metadata_file)
        label = metadata.get('label', '').strip()
        if label == 'IndexCancer':
            cancer_files.append(npz_file)
        elif label == 'NonCancer':
            noncancer_files.append(npz_file)

print("Found:", len(cancer_files), "cancer files,", len(noncancer_files), "non-cancer files")

original_npz_files = select_files(cancer_files, noncancer_files)
synthetic_files = get_png_files(synthetic_path)[:1000]

print("Selected:", len(cancer_files[:500]), "cancer,", len(noncancer_files[:500]), "non-cancer")
print("Total original images:", len(original_npz_files))
print("Total synthetic images:", len(synthetic_files))

print("Step 1: Processing original images...")
original_processed = []

for npz_file in tqdm(original_npz_files):
    data = np.load(npz_file, allow_pickle=True)
    img = data['data']
    
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    metadata_file = os.path.join(original_metadata_path, base_name + ".json")
    metadata = load_metadata(metadata_file)
    
    window_center = extract_value(metadata.get('WindowCenter', 400), 400)
    window_width = extract_value(metadata.get('WindowWidth', 1200), 1200)
    rescale_intercept = extract_value(metadata.get('RescaleIntercept', 0), 0)
    rescale_slope = extract_value(metadata.get('RescaleSlope', 1), 1)
    
    img = apply_window_level(img, window_center, window_width, rescale_intercept, rescale_slope)
    
    side = detect_breast_side(img)
    if side == 'right':
        img = cv2.flip(img, 1)
    
    original_processed.append((img, base_name))

print("Step 2: Processing synthetic images...")
synthetic_processed = []

for syn_file in tqdm(synthetic_files):
    img = cv2.imread(syn_file, cv2.IMREAD_GRAYSCALE)
    img = process_synthetic_image(img)
    synthetic_processed.append((img, os.path.basename(syn_file)))

print("Step 3: Saving original images...")
for img, base_name in tqdm(original_processed):
    output_file = os.path.join(output_original, base_name + ".png")
    cv2.imwrite(output_file, img)

print("Saving synthetic images...")
for img, filename in tqdm(synthetic_processed):
    output_file = os.path.join(output_synthetic, filename)
    cv2.imwrite(output_file, img)

print("Done!")
print("Original images saved to:", output_original)
print("Synthetic images saved to:", output_synthetic)
