#!/usr/bin/env python3
"""
HQ study-level resize + pad 

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

DEFAULT_IN = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_Data/original_study_level"
)
DEFAULT_OUT = Path(
    "/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_Data/512_study_level"
)


def _mode_value(arr):
    """SciPy-compatible mode (old [0][0] vs new .mode scalar/array)."""
    m = stats.mode(arr, axis=None, keepdims=True)
    val = getattr(m, "mode", m[0])
    return float(np.asarray(val).ravel()[0])


def resize_and_pad_image(image, target_size=(512, 512)):
    """Omama: scale keep aspect, pad darker side with mode fill."""
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


def worker(in_npz, in_json, out_npz, out_json, target_size):
    pixel_array = np.load(in_npz)["data"]
    resized_image, (pad_y, pad_x), scale = resize_and_pad_image(pixel_array, target_size)

    with open(in_json, "r") as f:
        metadata = json.load(f)

    # Remap only if DeepSight coords exist (skip for now if missing)
    coords = metadata.get("coords")
    if coords and len(coords) == 4:
        coords = [
            int(coords[0] * scale) + pad_x[0],
            int(coords[1] * scale) + pad_y[0],
            int(coords[2] * scale) + pad_x[0],
            int(coords[3] * scale) + pad_y[0],
        ]
        metadata["coords"] = coords

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    np.savez_compressed(out_npz, data=resized_image)
    with open(out_json, "w") as f:
        json.dump(metadata, f)


def worker_with_progress(args_tuple):
    in_npz, in_json, out_npz, out_json, target_size, lock, counter = args_tuple
    worker(in_npz, in_json, out_npz, out_json, target_size)
    with lock:
        counter.value += 1
        tqdm.write(f"Processed: {counter.value}", end="\r")


def collect_jobs(input_root, output_root):
    """Pair each study npz with its json; mirror paths under output_root."""
    images_root = Path(input_root) / "images"
    meta_root = Path(input_root) / "metadata"
    out_images = Path(output_root) / "images"
    out_meta = Path(output_root) / "metadata"

    jobs = []
    for study_dir in sorted(images_root.iterdir()):
        if not study_dir.is_dir():
            continue
        study_id = study_dir.name
        for npz_path in sorted(study_dir.glob("*.npz")):
            uid = npz_path.stem
            json_path = meta_root / study_id / f"{uid}.json"
            if not json_path.is_file():
                print(f"SKIP missing json: {json_path}")
                continue
            jobs.append(
                (
                    str(npz_path),
                    str(json_path),
                    str(out_images / study_id / f"{uid}.npz"),
                    str(out_meta / study_id / f"{uid}.json"),
                )
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="HQ study-level resize+pad (Omama method)."
    )
    parser.add_argument("-i", "--input", type=str, default=str(DEFAULT_IN))
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("-w", "--width", type=int, default=512)
    parser.add_argument("-ht", "--height", type=int, default=512)
    parser.add_argument("-p", "--processes", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process first N images (smoke test).",
    )
    args = parser.parse_args()

    target_size = (args.height, args.width)
    jobs = collect_jobs(args.input, args.output)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"size:   {args.width}x{args.height}")
    print(f"jobs:   {len(jobs)}  processes={args.processes}")

    os.makedirs(args.output, exist_ok=True)
    manager = Manager()
    lock = manager.Lock()
    counter = manager.Value("i", 0)

    work = [
        (in_npz, in_json, out_npz, out_json, target_size, lock, counter)
        for in_npz, in_json, out_npz, out_json in jobs
    ]

    with Pool(processes=args.processes) as pool:
        pool.map(worker_with_progress, work)

    print(f"\nDone. Wrote under {args.output}")


if __name__ == "__main__":
    main()
