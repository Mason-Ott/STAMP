import argparse
from pathlib import Path

import h5py
import numpy as np


def count_patches(h5_path: Path, dataset_key: str = "feats") -> int:
    """Return the number of patches stored in an HDF5 dataset."""

    with h5py.File(h5_path, "r") as f:
        if dataset_key not in f:
            available = []

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    available.append(name)

            f.visititems(collect)

            raise KeyError(
                f"Dataset '{dataset_key}' not found. "
                f"Available datasets: {available}"
            )

        dataset = f[dataset_key]

        if dataset.ndim == 0:
            raise ValueError(f"Dataset '{dataset_key}' is scalar")

        return dataset.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description="Calculate patch-count statistics for HDF5 files."
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .h5 files",
    )

    parser.add_argument(
        "--dataset-key",
        default="feats",
        help="HDF5 dataset containing patch features (default: feats)",
    )

    args = parser.parse_args()

    h5_files = sorted(args.folder.glob("*.h5"))

    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    print(f"Found {len(h5_files)} HDF5 files\n")

    results = []
    errors = []

    for h5_path in h5_files:
        try:
            patches = count_patches(h5_path, args.dataset_key)
            results.append((h5_path.name, patches))

        except Exception as e:
            errors.append((h5_path.name, str(e)))

    if not results:
        print("No valid files could be processed.")
        for name, error in errors:
            print(f"  {name}: {error}")
        return

    counts = np.array([count for _, count in results])

    print("=" * 55)
    print("PATCH COUNT STATISTICS")
    print("=" * 55)

    print(f"Files processed: {len(results)}")
    print(f"Files failed:    {len(errors)}")
    print(f"Total patches:   {counts.sum():,}")
    print(f"Mean:            {counts.mean():,.2f}")
    print(f"Median:          {np.median(counts):,.2f}")
    print(f"Std deviation:   {counts.std():,.2f}")
    print(f"Minimum:         {counts.min():,}")
    print(f"Maximum:         {counts.max():,}")

    print("\nPercentiles:")
    for percentile in [5, 10, 25, 50, 75, 90, 95, 99]:
        value = np.percentile(counts, percentile)
        print(f"  {percentile:>2}th: {value:,.2f}")

    print("\nSmallest slides:")
    for name, count in sorted(results, key=lambda x: x[1])[:10]:
        print(f"  {name:<30} {count:>10,}")

    print("\nLargest slides:")
    for name, count in sorted(
        results, key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"  {name:<30} {count:>10,}")

    if errors:
        print("\nFailed files:")
        for name, error in errors:
            print(f"  {name}: {error}")


if __name__ == "__main__":
    main()

# Example
# python patch_count_stats.py /csse/research/cancerimages/preprocess_test/conch/conch-2f775ba8