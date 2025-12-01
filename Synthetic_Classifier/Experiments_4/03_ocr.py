import keras_ocr
import cv2
import numpy as np
import os
from tqdm import tqdm

input_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/data_1000samples/original_cropped"
output_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/data_1000samples/original_cropped_ocr"

os.makedirs(output_path, exist_ok=True)

pipeline = keras_ocr.pipeline.Pipeline()

def detect_breast_region(img):
    _, mask = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    return mask

def is_in_corner(x_min, y_min, x_max, y_max, h, w):
    corner_size = 150
    
    top_left = y_max < corner_size and x_max < corner_size
    top_right = y_max < corner_size and x_min > (w - corner_size)
    bottom_left = y_min > (h - corner_size) and x_max < corner_size
    bottom_right = y_min > (h - corner_size) and x_min > (w - corner_size)
    
    top_edge = y_max < corner_size and corner_size <= x_min and x_max <= (w - corner_size)
    right_edge = x_min > (w - corner_size) and corner_size <= y_min and y_max <= (h - corner_size)
    
    return top_left or top_right or bottom_left or bottom_right or top_edge or right_edge

def check_overlap(x_min, y_min, x_max, y_max, breast_mask):
    x_min = max(0, int(x_min))
    x_max = min(breast_mask.shape[1], int(x_max))
    y_min = max(0, int(y_min))
    y_max = min(breast_mask.shape[0], int(y_max))
    
    box_area = (x_max - x_min) * (y_max - y_min)
    if box_area == 0:
        return False
    
    region = breast_mask[y_min:y_max, x_min:x_max]
    overlap = np.sum(region > 0)
    overlap_ratio = overlap / box_area
    
    return overlap_ratio > 0.1

def remove_text(img, pipeline):
    h, w = img.shape
    cleaned = img.copy()
    breast_mask = detect_breast_region(img)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    predictions = pipeline.recognize([img_rgb])
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for word, box in predictions[0]:
        x_coords = [box[0][0], box[1][0], box[2][0], box[3][0]]
        y_coords = [box[0][1], box[1][1], box[2][1], box[3][1]]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        if box_w > 150 or box_h > 150:
            continue
        
        in_corner = is_in_corner(x_min, y_min, x_max, y_max, h, w)
        
        if in_corner:
            cv2.rectangle(mask, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), 255, -1)
        else:
            if not check_overlap(x_min, y_min, x_max, y_max, breast_mask):
                cv2.rectangle(mask, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), 255, -1)
    
    if np.sum(mask) > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cleaned[mask > 0] = 0
    
    return cleaned

files = sorted(os.listdir(input_path))
png_files = [f for f in files if f.endswith('.png')]

print("Found", len(png_files), "images")

for filename in tqdm(png_files):
    try:
        img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
        cleaned = remove_text(img, pipeline)
        cv2.imwrite(os.path.join(output_path, filename), cleaned)
    except:
        continue

print("Done!")

