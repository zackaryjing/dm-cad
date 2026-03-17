#!/usr/bin/env python3
"""
Dataset Partition Script for Omni-CAD (High Performance Version)

Optimizations:
1. os.scandir instead of glob/Path.iterdir
2. Early exit when 8 images found
3. Single pass with os.walk for image scanning
4. Reduced IPC overhead with batch processing
5. No image_counts tracking (only validity check)

Usage:
    python data/partition_dataset.py --data_dir datasets/dataset_v0 --train_ratio 0.8 --seed 42
"""

import os
import json
import argparse
import random
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


def collect_all_descriptions(desc_dir: str) -> Set[str]:
    """Collect all sample IDs from cad_desc JSON files."""
    all_samples = set()

    for fname in os.listdir(desc_dir):
        if not fname.endswith('.json'):
            continue
        filepath = os.path.join(desc_dir, fname)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            if "id" in entry:
                all_samples.add(entry["id"].strip())

    print(f"Total sample IDs from cad_desc: {len(all_samples)}")
    return all_samples


def process_group_batch(group_dirs: List[str], img_base: str) -> Set[str]:
    """
    Process a batch of group directories.
    Returns set of valid sample IDs (samples with >= 8 images).
    """
    valid_samples = set()

    for group_id in group_dirs:
        group_path = os.path.join(img_base, group_id)
        try:
            with os.scandir(group_path) as it:
                for sample_entry in it:
                    if not sample_entry.is_dir(follow_symlinks=False):
                        continue

                    sample_id = f"{group_id}/{sample_entry.name}"
                    sample_path = sample_entry.path

                    # Count .png files, early exit at 8
                    count = 0
                    with os.scandir(sample_path) as sample_it:
                        for img_entry in sample_it:
                            if img_entry.name.endswith('.png'):
                                count += 1
                                if count >= 8:
                                    valid_samples.add(sample_id)
                                    break
        except (OSError, IOError):
            continue

    return valid_samples


def collect_all_images_parallel(img_dir: str, max_workers: int = None) -> Set[str]:
    """
    Collect all sample IDs with >= 8 images using parallel processing.
    Uses batch processing to reduce IPC overhead.
    Progress bar implemented with simple character-based display (no external deps).
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 32)  # Cap at 8 for I/O bound work

    # Get all group directories
    group_dirs = []
    for entry in os.scandir(img_dir):
        if entry.is_dir(follow_symlinks=False):
            group_dirs.append(entry.name)

    print(f"Found {len(group_dirs)} group directories in cad_img")
    print(f"Processing with {max_workers} workers (batch mode)...")

    # Batch groups for each worker to reduce IPC overhead
    batch_size = max(1, len(group_dirs) // max_workers)
    batches = [group_dirs[i:i + batch_size] for i in range(0, len(group_dirs), batch_size)]

    all_samples = set()
    total = len(batches)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_group_batch, batch, img_dir) for batch in batches]
        for future in futures:
            all_samples.update(future.result())
            completed += 1
            # Simple progress bar - no sync overhead, just local counting
            bar_len = 40
            filled = int(bar_len * completed / total)
            pct = 100.0 * completed / total
            print(f"\r[{('=' * filled).ljust(bar_len)}] {completed}/{total} ({pct:.1f}%)", end='', flush=True)

    print(f"\nSamples with >= 8 images: {len(all_samples)}")
    return all_samples


def collect_all_vectors(vec_dir: str) -> Set[str]:
    """Collect all sample IDs from cad_vec directory using os.scandir."""
    all_samples = set()

    for group_entry in os.scandir(vec_dir):
        if not group_entry.is_dir(follow_symlinks=False):
            continue

        group_id = group_entry.name
        group_path = group_entry.path

        for file_entry in os.scandir(group_path):
            if file_entry.name.endswith('.h5'):
                sample_id = f"{group_id}/{file_entry.name[:-3]}"  # Remove .h5
                all_samples.add(sample_id)

    print(f"Total sample IDs from cad_vec: {len(all_samples)}")
    return all_samples


def find_valid_samples(desc_dir: str, img_dir: str, vec_dir: str, max_workers: int = None) -> Tuple[Set[str], Dict]:
    """Find samples that exist in all three modalities."""
    print("\n" + "="*60)
    print("Step 1: Collecting samples from each modality...")
    print("="*60)

    desc_samples = collect_all_descriptions(desc_dir)
    img_samples = collect_all_images_parallel(img_dir, max_workers=max_workers)
    vec_samples = collect_all_vectors(vec_dir)

    print("\n" + "="*60)
    print("Step 2: Computing intersection...")
    print("="*60)

    valid_samples = desc_samples & img_samples & vec_samples

    stats = {
        'desc_count': len(desc_samples),
        'img_count': len(img_samples),
        'vec_count': len(vec_samples),
        'valid_count': len(valid_samples),
    }

    missing_desc = (img_samples & vec_samples) - desc_samples
    missing_img = (desc_samples & vec_samples) - img_samples
    missing_vec = (desc_samples & img_samples) - vec_samples

    print(f"\nValid samples (all 3 modalities): {len(valid_samples)}")
    print(f"Missing only description: {len(missing_desc)}")
    print(f"Missing only images: {len(missing_img)}")
    print(f"Missing only vec: {len(missing_vec)}")

    stats['missing_desc'] = len(missing_desc)
    stats['missing_img'] = len(missing_img)
    stats['missing_vec'] = len(missing_vec)

    return valid_samples, stats


def partition_samples(valid_samples: Set[str], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Partition valid samples into train/test sets with stratified sampling."""
    print("\n" + "="*60)
    print("Step 3: Partitioning into train/test sets...")
    print("="*60)

    random.seed(seed)

    # Group samples by group ID
    groups: Dict[str, List[str]] = {}
    for sample_id in valid_samples:
        group_id = sample_id.split("/")[0]
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(sample_id)

    train_samples = []
    test_samples = []

    print(f"\nStratified sampling across {len(groups)} groups...")

    for group_id in sorted(groups.keys()):
        group_samples = groups[group_id]
        n_train = max(1, int(len(group_samples) * train_ratio))
        random.shuffle(group_samples)
        train_samples.extend(group_samples[:n_train])
        test_samples.extend(group_samples[n_train:])

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Actual ratio: {len(train_samples)/len(valid_samples):.2%} / {len(test_samples)/len(valid_samples):.2%}")

    return train_samples, test_samples


def save_results(output_dir: str, valid_samples: Set[str], train_samples: List[str], test_samples: List[str], stats: Dict):
    """Save partition results to files."""
    print("\n" + "="*60)
    print("Step 4: Saving results...")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Save valid IDs
    valid_file = os.path.join(output_dir, "valid_ids.txt")
    with open(valid_file, 'w') as f:
        for sample_id in sorted(valid_samples):
            f.write(f"{sample_id}\n")
    print(f"Saved valid IDs to: {valid_file} ({len(valid_samples)} entries)")

    # Save train IDs
    train_file = os.path.join(output_dir, "train_ids.txt")
    with open(train_file, 'w') as f:
        for sample_id in sorted(train_samples):
            f.write(f"{sample_id}\n")
    print(f"Saved train IDs to: {train_file} ({len(train_samples)} entries)")

    # Save test IDs
    test_file = os.path.join(output_dir, "test_ids.txt")
    with open(test_file, 'w') as f:
        for sample_id in sorted(test_samples):
            f.write(f"{sample_id}\n")
    print(f"Saved test IDs to: {test_file} ({len(test_samples)} entries)")

    # Save statistics
    stats_file = os.path.join(output_dir, "partition_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_file}")


def print_summary(stats: Dict, train_samples: List[str], test_samples: List[str]):
    """Print a summary of the partition."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total descriptions found:    {stats['desc_count']:,}")
    print(f"Samples with >= 8 images:    {stats['img_count']:,}")
    print(f"Samples with CAD vectors:    {stats['vec_count']:,}")
    print(f"-------------------------------------------")
    print(f"Valid samples (intersection): {stats['valid_count']:,}")
    print(f"  - Train set:               {len(train_samples):,} ({len(train_samples)/stats['valid_count']*100:.1f}%)")
    print(f"  - Test set:                {len(test_samples):,} ({len(test_samples)/stats['valid_count']*100:.1f}%)")
    print(f"-------------------------------------------")
    print(f"Excluded samples:")
    print(f"  - Missing images:          {stats['missing_img']:,}")
    print(f"  - Missing vectors:         {stats['missing_vec']:,}")
    print(f"  - Missing descriptions:    {stats['missing_desc']:,}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Partition Omni-CAD dataset into train/test sets")
    parser.add_argument("--data_dir", type=str, default="datasets/dataset_v0",
                        help="Path to dataset directory (default: datasets/dataset_v0)")
    parser.add_argument("--output_dir", type=str, default="datasets/dataset_v0",
                        help="Output directory for partition files (default: datasets/dataset_v0)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: min(cpu_count, 8))")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(base_dir, args.data_dir)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, args.output_dir)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Workers: {args.workers or 'auto'}")

    desc_dir = os.path.join(data_dir, "cad_desc")
    img_dir = os.path.join(data_dir, "cad_img")
    vec_dir = os.path.join(data_dir, "cad_vec")

    for dir_path, name in [(desc_dir, "cad_desc"), (img_dir, "cad_img"), (vec_dir, "cad_vec")]:
        if not os.path.exists(dir_path):
            print(f"Error: {name} directory not found: {dir_path}")
            return

    valid_samples, stats = find_valid_samples(desc_dir, img_dir, vec_dir, max_workers=args.workers)

    if len(valid_samples) == 0:
        print("Error: No valid samples found!")
        return

    train_samples, test_samples = partition_samples(valid_samples, train_ratio=args.train_ratio, seed=args.seed)
    save_results(output_dir, valid_samples, train_samples, test_samples, stats)
    print_summary(stats, train_samples, test_samples)


if __name__ == "__main__":
    main()
