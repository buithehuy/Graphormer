"""
Standalone script to process rice disease images to graph .pt files.
No Graphormer/fairseq dependencies needed - only PyTorch Geometric.

Usage:
    python process_images.py --image_dir /path/to/images --output_dir /path/to/output
    python process_images.py --image_dir /path/to/images --output_dir /path/to/output --augment
    python process_images.py --image_dir /path/to/images --output_dir /path/to/output --augment --target_count 1500
"""

import os
import os.path as osp
import torch
import json
import zipfile
import argparse
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# Import converter (no Graphormer dependency)
import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
from rice_image_to_graph import ImageToGraphConverter


# Class mapping
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

AUGMENTATIONS = [
    'rotate_90',
    'rotate_180',
    'rotate_270',
    'flip_horizontal',
    'flip_vertical',
    'zoom_in',
    'zoom_out',
]


def augment_image(image: Image.Image, aug_type: str) -> Image.Image:
    """
    Apply a single augmentation to a PIL image.

    Args:
        image: Source PIL Image (should already be 224×224)
        aug_type: One of AUGMENTATIONS list

    Returns:
        Augmented PIL Image (same size as input)
    """
    w, h = image.size

    if aug_type == 'rotate_90':
        return image.rotate(90, expand=False)
    elif aug_type == 'rotate_180':
        return image.rotate(180, expand=False)
    elif aug_type == 'rotate_270':
        return image.rotate(270, expand=False)
    elif aug_type == 'flip_horizontal':
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif aug_type == 'flip_vertical':
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif aug_type == 'zoom_in':
        # Crop centre 80% then resize back
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        crop_box = (margin_x, margin_y, w - margin_x, h - margin_y)
        return image.crop(crop_box).resize((w, h), Image.Resampling.LANCZOS)
    elif aug_type == 'zoom_out':
        # Shrink to 80% then pad with black to original size
        new_w, new_h = int(w * 0.80), int(h * 0.80)
        small = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (w, h), (0, 0, 0))
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas.paste(small, (offset_x, offset_y))
        return canvas
    else:
        return image


def augment_class_images(
    image_paths: list,
    target_count: int,
    seed: int = 42,
) -> list:
    """
    Generate augmented (PIL Image, original_path) pairs until reaching target_count.

    Args:
        image_paths: List of file paths for one class
        target_count: Desired total sample count after augmentation
        seed: Random seed for reproducibility

    Returns:
        List of (PIL Image, source_path, aug_type_or_None) tuples.
        Original images come first (aug_type=None), then augmented ones.
    """
    rng = random.Random(seed)
    results = []  # (pil_image, path, aug_type)

    # First: all original images
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            results.append((img, path, None))
        except Exception as e:
            print(f"    Warning: Failed to load {path}: {e}")

    needed = target_count - len(results)
    if needed <= 0:
        return results

    # Cycle through source images + augmentations to generate needed extras
    aug_types = AUGMENTATIONS.copy()
    source_pool = [(path,) for path in image_paths]
    rng.shuffle(source_pool)

    # Build a cycling iterator of (path, aug_type) pairs
    combo_pool = [(p, a) for p in image_paths for a in aug_types]
    rng.shuffle(combo_pool)

    generated = 0
    pool_idx = 0
    while generated < needed:
        path, aug_type = combo_pool[pool_idx % len(combo_pool)]
        pool_idx += 1
        try:
            img = Image.open(path).convert('RGB')
            aug_img = augment_image(img, aug_type)
            img.close()
            results.append((aug_img, path, aug_type))
            generated += 1
        except Exception as e:
            print(f"    Warning: Augmentation failed for {path}: {e}")

    return results


# ---------------------------------------------------------------------------
# Main processing helpers
# ---------------------------------------------------------------------------

def find_images(image_dir):
    """Find all images organized by class."""
    data_dict = {class_name: [] for class_name in CLASS_NAMES}
    
    # Try LabelledRice/Labelled structure
    labelled_dir = Path(image_dir) / "LabelledRice" / "Labelled"
    if not labelled_dir.exists():
        labelled_dir = Path(image_dir) / "Labelled"
    
    if labelled_dir.exists():
        for class_name in CLASS_NAMES:
            class_dir = labelled_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + \
                        list(class_dir.glob("*.jpeg")) + \
                        list(class_dir.glob("*.png"))
                data_dict[class_name] = [str(p) for p in images]
    else:
        # Try RiceDiseaseDataset structure
        for split_name in ['train', 'validation']:
            split_dir = Path(image_dir) / "RiceDiseaseDataset" / split_name
            if split_dir.exists():
                for class_name in CLASS_NAMES:
                    class_dir = split_dir / class_name
                    if class_dir.exists():
                        images = list(class_dir.glob("*.jpg")) + \
                                list(class_dir.glob("*.jpeg")) + \
                                list(class_dir.glob("*.png"))
                        data_dict[class_name].extend([str(p) for p in images])
    
    return data_dict


def process_images_to_graphs(
    image_dir,
    output_dir,
    n_segments=75,
    seed=42,
    augment=False,
    target_count=0,
):
    """
    Process all images and save as individual .pt files.

    Args:
        image_dir: Directory containing rice disease images
        output_dir: Directory to save processed graphs
        n_segments: Number of superpixels per image
        seed: Random seed for splits
        augment: If True, oversample minority classes via augmentation
        target_count: Target samples per class (0 = use max class count)

    Returns:
        Path to processed directory
    """
    # Create output directory
    processed_dir = osp.join(output_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 60)
    print("Processing rice disease images to graphs...")
    print("=" * 60)

    # Find all images
    data_dict = find_images(image_dir)

    # Determine target_count for augmentation
    class_sizes = {c: len(data_dict[c]) for c in CLASS_NAMES}
    max_class_size = max(class_sizes.values())
    if target_count <= 0:
        target_count = max_class_size

    if augment:
        print(f"\n[Augmentation ON] Target samples per class: {target_count}")
        print(f"  Original class sizes:")
        for c, n in class_sizes.items():
            extra = max(0, target_count - n)
            print(f"    {c:12s}: {n:5d}  →  {min(n, target_count) + extra:5d}  (+{extra}  augmented)")

    # Initialize converter
    converter = ImageToGraphConverter(n_segments=n_segments)

    # Process each image (original + augmented)
    all_labels = []
    all_image_paths = []
    all_is_augmented = []
    graph_idx = 0
    total_augmented = 0

    for class_name in CLASS_NAMES:
        class_idx = CLASS_TO_IDX[class_name]
        image_paths = data_dict[class_name]

        if augment and len(image_paths) < target_count:
            samples = augment_class_images(image_paths, target_count, seed=seed)
            n_orig = len(image_paths)
            n_aug = len(samples) - n_orig
            total_augmented += n_aug
            print(f"\nProcessing {class_name}: {n_orig} original + {n_aug} augmented...")
        else:
            # No augmentation needed — build (pil, path, None) tuples on the fly
            samples = None
            print(f"\nProcessing {class_name} ({len(image_paths)} images)...")

        if samples is not None:
            # Augmentation path: images already loaded into PIL objects
            for pil_img, src_path, aug_type in tqdm(samples, desc=f"  {class_name}"):
                try:
                    graph = converter.convert(pil_img, label=class_idx)
                    save_path = osp.join(processed_dir, f'data_{graph_idx}.pt')
                    torch.save(graph, save_path)

                    all_labels.append(class_idx)
                    all_image_paths.append(str(src_path))
                    all_is_augmented.append(aug_type is not None)
                    graph_idx += 1

                    del graph
                    pil_img.close()
                except Exception as e:
                    print(f"    Warning: Failed to process (aug={aug_type}) {src_path}: {e}")
                    continue
        else:
            # No augmentation: load images lazily to save RAM
            for img_path in tqdm(image_paths, desc=f"  {class_name}"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    graph = converter.convert(image, label=class_idx)

                    save_path = osp.join(processed_dir, f'data_{graph_idx}.pt')
                    torch.save(graph, save_path)

                    all_labels.append(class_idx)
                    all_image_paths.append(str(img_path))
                    all_is_augmented.append(False)
                    graph_idx += 1

                    del image, graph
                except Exception as e:
                    print(f"    Warning: Failed to process {img_path}: {e}")
                    continue

    print(f"\n✓ Processed {graph_idx} graphs total")
    if augment:
        print(f"  (of which {total_augmented} are augmented)")

    # Print per-class summary
    from collections import Counter
    label_counter = Counter(all_labels)
    print("\nFinal class distribution:")
    for c in CLASS_NAMES:
        idx = CLASS_TO_IDX[c]
        print(f"  {c:12s}: {label_counter[idx]}")

    # Create splits
    print("\nCreating train/val/test splits...")
    indices = np.arange(graph_idx)
    labels_np = np.array(all_labels)

    # 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels_np, random_state=seed
    )

    # Split temp 50/50 → 15% val, 15% test
    temp_labels = labels_np[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    # Save splits
    split_path = osp.join(processed_dir, 'split_indices.pt')
    torch.save({
        'train_idx': torch.from_numpy(train_idx),
        'val_idx': torch.from_numpy(val_idx),
        'test_idx': torch.from_numpy(test_idx)
    }, split_path)
    print(f"  Saved: split_indices.pt")

    # Save metadata
    metadata = {
        'num_graphs': graph_idx,
        'class_names': CLASS_NAMES,
        'n_segments': n_segments,
        'labels': all_labels,
        'image_paths': all_image_paths,
        'is_augmented': all_is_augmented,
        'num_train': len(train_idx),
        'num_val': len(val_idx),
        'num_test': len(test_idx),
        # augmentation info
        'augmentation_enabled': augment,
        'target_count': target_count if augment else None,
        'num_augmented': total_augmented,
    }

    metadata_path = osp.join(processed_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

    return processed_dir


def create_zip_archive(processed_dir, output_zip_path):
    """Create zip archive of processed graphs."""
    print(f"\nCreating zip archive...")
    print("-" * 60)
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        pt_files = list(Path(processed_dir).glob("*.pt"))
        json_files = list(Path(processed_dir).glob("*.json"))
        files_to_zip = pt_files + json_files
        
        for file_path in tqdm(files_to_zip, desc="Compressing"):
            arcname = file_path.name
            zipf.write(file_path, arcname)
    
    zip_size_mb = Path(output_zip_path).stat().st_size / (1024 * 1024)
    
    print("-" * 60)
    print(f"✓ Zip archive created!")
    print(f"  Location: {output_zip_path}")
    print(f"  Size: {zip_size_mb:.2f} MB")
    print(f"  Files: {len(files_to_zip)}")
    
    return output_zip_path


def main():
    parser = argparse.ArgumentParser(
        description='Process rice disease images to graphs with optional augmentation'
    )
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing rice disease images')
    parser.add_argument('--output_dir', type=str, default='rice_diseases_graphs',
                        help='Output directory for processed graphs')
    parser.add_argument('--n_segments', type=int, default=75,
                        help='Number of superpixels per image')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits and augmentation')
    parser.add_argument('--create_zip', action='store_true',
                        help='Create zip archive after processing')
    # ── Augmentation flags ──────────────────────────────────────────────────
    parser.add_argument(
        '--augment', action='store_true',
        help=(
            'Enable oversampling augmentation for minority classes. '
            'Applies: rotate 90/180/270, flip h/v, zoom in/out.'
        )
    )
    parser.add_argument(
        '--target_count', type=int, default=0,
        help=(
            'Target number of samples per class after augmentation. '
            'Default 0 = use the size of the largest class.'
        )
    )

    args = parser.parse_args()

    # Process images
    processed_dir = process_images_to_graphs(
        args.image_dir,
        args.output_dir,
        args.n_segments,
        args.seed,
        augment=args.augment,
        target_count=args.target_count,
    )

    # Create zip if requested
    if args.create_zip:
        zip_path = osp.join(
            osp.dirname(args.output_dir),
            osp.basename(args.output_dir) + '.zip'
        )
        create_zip_archive(processed_dir, zip_path)


if __name__ == '__main__':
    main()
