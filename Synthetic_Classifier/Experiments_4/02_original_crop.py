import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def crop_image(img):
    h, w = img.shape
    
    top = None
    for row in range(h):
        if img[row, :].mean() > 5:
            top = row
            break
    
    bottom = None
    for row in range(h-1, -1, -1):
        if img[row, :].mean() > 5:
            bottom = row
            break
    
    left = 0
    for col in range(w):
        if img[:, col].mean() > 5:
            left = col
            break
    
    right = w
    for col in range(w-1, -1, -1):
        if img[:, col].mean() > 5:
            right = col + 1
            break
    
    cropped = img[top:bottom+1, left:right]
    h_crop, w_crop = cropped.shape
    
    margin_right = 25
    cropped = np.pad(cropped, ((0, 0), (0, margin_right)), mode='constant', constant_values=0)
    h_crop, w_crop = cropped.shape
    
    scale = 512.0 / h_crop
    new_w = int(w_crop * scale)
    
    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4
    
    if new_w > 512:
        result = cv2.resize(cropped, (512, 512), interpolation=interp)
    else:
        resized = cv2.resize(cropped, (new_w, 512), interpolation=interp)
        pad_right = 512 - new_w
        result = np.pad(resized, ((0, 0), (0, pad_right)), mode='constant', constant_values=0)
    
    return result

input_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/data_1000samples/original_no_norm"
output_path = "/hpcstor6/scratch01/a/a.kanamarlapudi001/synthetic_classifier/data_1000samples/original_cropped"

os.makedirs(output_path, exist_ok=True)

files = sorted(os.listdir(input_path))
png_files = [f for f in files if f.endswith('.png')]

print("Found", len(png_files), "images")

before_list = []
after_list = []

for filename in png_files:
    img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
    
    if len(before_list) < 10:
        before_list.append(img.copy())
    
    cropped = crop_image(img)
    
    if len(after_list) < 10:
        after_list.append(cropped.copy())
    
    cv2.imwrite(os.path.join(output_path, filename), cropped)

print("Done!")

fig, axes = plt.subplots(10, 2, figsize=(10, 50))
for i in range(10):
    axes[i, 0].imshow(before_list[i], cmap='gray')
    axes[i, 0].set_title('Before ' + str(i+1))
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(after_list[i], cmap='gray')
    axes[i, 1].set_title('After ' + str(i+1))
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/a.kanamarlapudi001/projects/omama-proj/_EXPERIMENTS/SYNTHETIC/Avanith/classifier_try_november/crop_comparison.png')
print("Comparison saved")
