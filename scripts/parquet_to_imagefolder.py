"""Convert HuggingFace ImageNet parquet files to torchvision ImageFolder layout.

Expected input layout (as downloaded from HuggingFace):
    <parquet_root>/
        data/
            train-*.parquet
            validation-*.parquet  (or test-*.parquet for imagenet-100)

Output layout:
    <output_root>/
        train/<class_name>/<filename>.JPEG
        val/<class_name>/<filename>.JPEG

Usage:
    python scripts/parquet_to_imagefolder.py \
        --parquet_root /path/to/hf_download \
        --output_root  /path/to/imagenet_imagefolder
"""

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def convert_split(parquet_files: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for pq_file in sorted(parquet_files):
        print(f"  Reading {pq_file.name} ...")
        table = pq.read_table(pq_file)
        df = table.to_pydict()

        n_rows = len(df["image"])
        for i in tqdm(range(n_rows), desc=pq_file.name, leave=False):
            img_entry = df["image"][i]

            # image bytes
            if isinstance(img_entry, dict):
                img_bytes = img_entry.get("bytes") or img_entry.get("path")
                orig_path = img_entry.get("path", "")
            else:
                img_bytes = img_entry
                orig_path = ""

            # derive class name from the original path when available
            # HF imagenet paths look like "n01440764/n01440764_10026.JPEG"
            if orig_path and "/" in orig_path:
                class_name = orig_path.split("/")[0]
                filename = Path(orig_path).name
            else:
                # fall back to label index as class dir
                class_name = str(df["label"][i])
                filename = f"{total:08d}.JPEG"

            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            out_path = class_dir / filename

            if not out_path.exists():
                try:
                    # images are already JPEG-encoded bytes — write directly
                    out_path.write_bytes(img_bytes)
                except Exception as e:
                    print(f"    Warning: skipping {filename}: {e}")

            total += 1

    print(f"  Done — {total} images written to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_root", required=True,
                        help="Root dir of the HuggingFace download")
    parser.add_argument("--output_root", required=True,
                        help="Where to write the ImageFolder structure")
    args = parser.parse_args()

    parquet_root = Path(args.parquet_root)
    output_root  = Path(args.output_root)

    data_dir = parquet_root / "data"
    if not data_dir.exists():
        data_dir = parquet_root  # some repos put parquets directly at root

    train_files = sorted(data_dir.glob("train-*.parquet"))
    val_files   = sorted(data_dir.glob("validation-*.parquet"))
    if not val_files:
        val_files = sorted(data_dir.glob("val-*.parquet"))

    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet found under {data_dir}")
    if not val_files:
        raise FileNotFoundError(f"No validation-*.parquet found under {data_dir}")

    print(f"Found {len(train_files)} train parquet(s), {len(val_files)} val parquet(s).")

    print("\n=== Converting train split ===")
    convert_split(train_files, output_root / "train")

    print("\n=== Converting val split ===")
    convert_split(val_files, output_root / "val")

    print(f"\nAll done. ImageFolder dataset at: {output_root}")


if __name__ == "__main__":
    main()
