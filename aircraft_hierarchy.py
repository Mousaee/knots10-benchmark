"""
FGVC-Aircraft Hierarchy Distance Matrix
Constructs a 100x100 variant distance matrix based on manufacturer -> family -> variant hierarchy.
"""

import os
from pathlib import Path
import numpy as np


def build_aircraft_distance_matrix(data_dir):
    """
    Build a 100x100 distance matrix for aircraft variants.

    Distance computation:
    - same_variant: 0.0
    - same_family: 0.33
    - same_manufacturer: 0.67
    - different_manufacturer: 1.0

    Args:
        data_dir (str): Path to fgvc-aircraft-2013b dataset root

    Returns:
        tuple: (distance_matrix, variants_list)
            - distance_matrix: np.ndarray of shape (100, 100)
            - variants_list: list of 100 variant names
    """
    data_dir = Path(data_dir)

    # Read variants list
    variants_file = data_dir / "data" / "variants.txt"
    with open(variants_file, 'r') as f:
        variants = [line.strip() for line in f if line.strip()]

    num_variants = len(variants)
    print(f"Loaded {num_variants} variants")

    # Build mappings: variant_name -> family and variant_name -> manufacturer
    variant_to_family = {}
    variant_to_manufacturer = {}

    # Process training images to establish mappings
    # First, build image_id -> variant, family, manufacturer mappings
    image_variant_file = data_dir / "data" / "images_variant_train.txt"
    image_family_file = data_dir / "data" / "images_family_train.txt"
    image_mfr_file = data_dir / "data" / "images_manufacturer_train.txt"

    # Build image_id -> variant mapping
    image_to_variant = {}
    if image_variant_file.exists():
        with open(image_variant_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    variant = ' '.join(parts[1:])
                    image_to_variant[image_id] = variant

    # Build image_id -> family mapping
    image_to_family = {}
    if image_family_file.exists():
        with open(image_family_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    family = ' '.join(parts[1:])
                    image_to_family[image_id] = family

    # Build image_id -> manufacturer mapping
    image_to_manufacturer = {}
    if image_mfr_file.exists():
        with open(image_mfr_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    manufacturer = ' '.join(parts[1:])
                    image_to_manufacturer[image_id] = manufacturer

    # Build variant -> family and variant -> manufacturer mappings
    for image_id, variant in image_to_variant.items():
        if variant not in variant_to_family and image_id in image_to_family:
            variant_to_family[variant] = image_to_family[image_id]
        if variant not in variant_to_manufacturer and image_id in image_to_manufacturer:
            variant_to_manufacturer[variant] = image_to_manufacturer[image_id]

    # Ensure all variants have mappings (fallback: extract family from variant name)
    for variant in variants:
        if variant not in variant_to_family:
            # Try to extract family from variant (heuristic: last token before numbers)
            parts = variant.split()
            if len(parts) > 0:
                # Assume first part(s) before hyphen/numbers form the family
                variant_to_family[variant] = parts[0]
            else:
                variant_to_family[variant] = variant

        if variant not in variant_to_manufacturer:
            # Default: assign placeholder (should be overridden by data)
            variant_to_manufacturer[variant] = "Unknown"

    print(f"Mapped {len(variant_to_family)} variants to families")
    print(f"Mapped {len(variant_to_manufacturer)} variants to manufacturers")

    # Compute distance matrix
    distance_matrix = np.zeros((num_variants, num_variants), dtype=np.float32)

    for i, variant_i in enumerate(variants):
        for j, variant_j in enumerate(variants):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                family_i = variant_to_family.get(variant_i, "Unknown")
                family_j = variant_to_family.get(variant_j, "Unknown")
                mfr_i = variant_to_manufacturer.get(variant_i, "Unknown")
                mfr_j = variant_to_manufacturer.get(variant_j, "Unknown")

                if family_i == family_j:
                    distance_matrix[i, j] = 0.33
                elif mfr_i == mfr_j:
                    distance_matrix[i, j] = 0.67
                else:
                    distance_matrix[i, j] = 1.0

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance matrix stats: min={distance_matrix.min():.2f}, "
          f"max={distance_matrix.max():.2f}, "
          f"mean={distance_matrix.mean():.2f}")

    return distance_matrix, variants


if __name__ == "__main__":
    # Test with default path
    default_path = os.path.expanduser("~/data/fgvc-aircraft-2013b")
    if os.path.exists(default_path):
        dist_matrix, variant_list = build_aircraft_distance_matrix(default_path)
        print(f"\nBuilt distance matrix for {len(variant_list)} aircraft variants")
        print(f"Matrix shape: {dist_matrix.shape}")
        print(f"Sample distances:\n{dist_matrix[:5, :5]}")
    else:
        print(f"Dataset not found at {default_path}")
        print("Please set AIRCRAFT_DIR environment variable or ensure dataset is at ~/data/fgvc-aircraft-2013b")
