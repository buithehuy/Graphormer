"""
Standalone script to process rice disease images to graph .pt files.
No Graphormer/fairseq dependencies needed - only PyTorch Geometric.

Usage:
    python process_images.py --image_dir /path/to/images --output_dir /path/to/output
"""

import os
import os.path as osp
import torch
import json
import zipfile
import argparse
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


def process_images_to_graphs(image_dir, output_dir, n_segments=75, seed=42):
    """
    Process all images and save as individual .pt files.
    
    Args:
        image_dir: Directory containing rice disease images
        output_dir: Directory to save processed graphs
        n_segments: Number of superpixels per image
        seed: Random seed for splits
    
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
    
    # Initialize converter
    converter = ImageToGraphConverter(n_segments=n_segments)
    
    # Process each image
    all_labels = []
    all_image_paths = []
    graph_idx = 0
    
    for class_name in CLASS_NAMES:
        class_idx = CLASS_TO_IDX[class_name]
        image_paths = data_dict[class_name]
        
        print(f"\nProcessing {class_name} ({len(image_paths)} images)...")
        
        for img_path in tqdm(image_paths, desc=f"  {class_name}"):
            try:
                # Load and convert
                image = Image.open(img_path).convert('RGB')
                graph = converter.convert(image, label=class_idx)
                
                # Save immediately
                save_path = osp.join(processed_dir, f'data_{graph_idx}.pt')
                torch.save(graph, save_path)
                
                # Track metadata
                all_labels.append(class_idx)
                all_image_paths.append(img_path)
                graph_idx += 1
                
                # Free memory
                del image
                del graph
            
            except Exception as e:
                print(f"    Warning: Failed to process {img_path}: {e}")
                continue
    
    print(f"\n✓ Processed {graph_idx} graphs")
    
    # Create splits
    print("\nCreating train/val/test splits...")
    indices = np.arange(graph_idx)
    labels_np = np.array(all_labels)
    
    # 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels_np, random_state=seed
    )
    
    # Split temp 50/50
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
        'num_train': len(train_idx),
        'num_val': len(val_idx),
        'num_test': len(test_idx)
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
    parser = argparse.ArgumentParser(description='Process rice disease images to graphs')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing rice disease images')
    parser.add_argument('--output_dir', type=str, default='rice_diseases_graphs',
                        help='Output directory for processed graphs')
    parser.add_argument('--n_segments', type=int, default=75,
                        help='Number of superpixels per image')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    parser.add_argument('--create_zip', action='store_true',
                        help='Create zip archive after processing')
    
    args = parser.parse_args()
    
    # Process images
    processed_dir = process_images_to_graphs(
        args.image_dir,
        args.output_dir,
        args.n_segments,
        args.seed
    )
    
    # Create zip if requested
    if args.create_zip:
        zip_path = osp.join(osp.dirname(args.output_dir), 
                           osp.basename(args.output_dir) + '.zip')
        create_zip_archive(processed_dir, zip_path)


if __name__ == '__main__':
    main()
