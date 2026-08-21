#!/usr/bin/env python3
"""
Image-level 2d → 256x256 uint16 (pad method, same as Omama 512/1024 / study-level).

Keep aspect, pad darker side with mode fill, remap coords.
Reads flat images/ + metadata/; writes the same under -o.
"""

import argparse
import json
import os
from multiprocessing import Manager, Pool
from pathlib import Path

import numpy as np
from scipy import stats
from skimage.transform import resize
from tqdm import tqdm

DEFAULT_IN = "/raid/mpsych/OMAMA/DATA/data/2d/images"
DEFAULT_JSON = "/raid/mpsych/OMAMA/DATA/data/2d/metadata"
DEFAULT_OUT = (
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/2d_resized_256_uint16"
)


def _mode_value(arr):
    m = stats.mode(arr, axis=None, keepdims=True)
    val = getattr(m, "mode", m[0])
    return float(np.asarray(val).ravel()[0])


def resize_and_pad_image(image, target_size=(256, 256)):
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = resize(image, (new_h, new_w), order=1, preserve_range=True)

    upper_half = resized_image[: new_h // 2, :]
    lower_half = resized_image[new_h // 2 :, :]
    left_half = resized_image[:, : new_w // 2]
    right_half = resized_image[:, new_w // 2 :]

    upper_half_mean = np.mean(upper_half)
    lower_half_mean = np.mean(lower_half)
    left_half_mean = np.mean(left_half)
    right_half_mean = np.mean(right_half)

    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h

    if left_half_mean < right_half_mean:
        pad_x = (delta_w, 0)
        pad_color_x = _mode_value(left_half)
    else:
        pad_x = (0, delta_w)
        pad_color_x = _mode_value(right_half)

    if upper_half_mean < lower_half_mean:
        pad_y = (delta_h, 0)
        pad_color_y = _mode_value(upper_half)
    else:
        pad_y = (0, delta_h)
        pad_color_y = _mode_value(lower_half)

    padded_image = np.pad(
        resized_image,
        [(pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])],
        mode="constant",
        constant_values=((pad_color_y, pad_color_y), (pad_color_x, pad_color_x)),
    )
    return padded_image.astype(np.uint16), (pad_y, pad_x), scale


def worker(npz_file, input_dir, json_dir, out_images, out_meta, target_size):
    pixel_array = np.load(os.path.join(input_dir, npz_file))["data"]
    h, w = pixel_array.shape
    resized_image, (pad_y, pad_x), scale = resize_and_pad_image(
        pixel_array, target_size
    )

    stem = os.path.splitext(npz_file)[0]
    json_path = os.path.join(json_dir, f"{stem}.json")
    with open(json_path, "r") as f:
        metadata = json.load(f)

    coords = metadata.get("coords")
    if coords and len(coords) == 4:
        metadata["coords"] = [
            int(coords[0] * scale) + pad_x[0],
            int(coords[1] * scale) + pad_y[0],
            int(coords[2] * scale) + pad_x[0],
            int(coords[3] * scale) + pad_y[0],
        ]

    np.savez_compressed(os.path.join(out_images, npz_file), data=resized_image)
    with open(os.path.join(out_meta, f"{stem}.json"), "w") as f:
        json.dump(metadata, f)


def worker_with_progress(args_tuple):
    npz_file, input_dir, json_dir, out_images, out_meta, target_size, lock, counter = (
        args_tuple
    )
    worker(npz_file, input_dir, json_dir, out_images, out_meta, target_size)
    with lock:
        counter.value += 1
        tqdm.write(f"Processed: {counter.value}", end="\r")


def main():
    parser = argparse.ArgumentParser(
        description="Image-level 2d → 256 uint16 pad resize."
    )
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_IN)
    parser.add_argument("-j", "--json", type=str, default=DEFAULT_JSON)
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("-w", "--width", type=int, default=256)
    parser.add_argument("-ht", "--height", type=int, default=256)
    parser.add_argument("-p", "--processes", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max images (0=all). use e.g. 20 for smoke test",
    )
    args = parser.parse_args()

    input_dir = args.input
    json_dir = args.json
    out_root = Path(args.output)
    out_images = str(out_root / "images")
    out_meta = str(out_root / "metadata")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_meta, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npz"))
    if args.limit > 0:
        npz_files = npz_files[: args.limit]

    target_size = (args.height, args.width)
    print(f"images: {len(npz_files)}  out: {out_root}  size: {target_size}")

    manager = Manager()
    lock = manager.Lock()
    counter = manager.Value("i", 0)

    jobs = [
        (
            npz_file,
            input_dir,
            json_dir,
            out_images,
            out_meta,
            target_size,
            lock,
            counter,
        )
        for npz_file in npz_files
    ]
    with Pool(processes=args.processes) as pool:
        pool.map(worker_with_progress, jobs)

    print("\nDone.")


if __name__ == "__main__":
    main()
