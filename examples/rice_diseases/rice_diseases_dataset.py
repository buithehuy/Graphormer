"""
Rice Diseases Graph Dataset - PyTorch Geometric Format

Custom PyG dataset for rice disease classification using graph representations.
Saves each graph as individual .pt file for fairseq compatibility.
"""

import os
import os.path as osp
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Dataset, Data
import zipfile
import json

from .rice_image_to_graph import ImageToGraphConverter


# Disease class mapping
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class RiceDiseasesDataset(Dataset):
    """
    PyTorch Geometric Dataset for rice disease images converted to graphs.
    
    Each graph is saved as a separate .pt file in the processed directory.
    Compatible with Graphormer fairseq framework.
    
    Args:
        root: Root directory where dataset is stored
        image_dir: Directory containing the rice disease images
        split: 'train', 'val', or 'test'
        n_segments: Number of superpixels for graph conversion
        transform: Optional transform to apply to graphs
        pre_transform: Optional pre-transform
        pre_filter: Optional pre-filter
        force_process: If True, reprocess even if processed files exist
    """
    
    def __init__(
        self,
        root,
        image_dir=None,
        split='train',
        n_segments=75,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_process=False,
        seed=42
    ):
        self.image_dir = image_dir
        self.split = split
        self.n_segments = n_segments
        self.seed = seed
        self._force_process = force_process
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load split indices
        split_path = osp.join(self.processed_dir, 'split_indices.pt')
        if osp.exists(split_path):
            splits = torch.load(split_path)
            self.data_indices = splits[f'{split}_idx']
        else:
            # If no splits exist yet, use all data
            self.data_indices = torch.arange(len(self.processed_file_names))
    
    @property
    def raw_file_names(self):
        """
        List of raw file names. We don't have raw files since images
        are processed from external directory.
        """
        return []
    
    @property
    def processed_file_names(self):
        """
        List of processed file names. Returns list of data_*.pt files.
        """
        # Check if metadata exists
        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        if osp.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            num_graphs = metadata['num_graphs']
            return [f'data_{i}.pt' for i in range(num_graphs)]
        else:
            # During first processing, return empty list
            return []
    
    def download(self):
        """
        Download method (not used - data is copied from Google Drive manually).
        """
        if self.image_dir is None:
            print("Note: Images should be copied from Google Drive manually.")
            print("Use colab_setup.copy_and_extract_dataset() first.")
        pass
    
    def process(self):
        """
        Process all images and save each graph as individual .pt file.
        
        This method processes images one at a time to avoid RAM overflow.
        Each graph is saved immediately and deleted from memory.
        """
        if not self._force_process and len(self.processed_file_names) > 0:
            print("Processed files already exist. Set force_process=True to reprocess.")
            return
        
        if self.image_dir is None:
            raise ValueError(
                "image_dir must be provided for processing. "
                "Please set it when creating the dataset."
            )
        
        print("=" * 60)
        print("Processing rice disease images to graphs...")
        print("=" * 60)
        
        # Find all images
        data_dict = self._find_images()
        
        # Initialize converter
        converter = ImageToGraphConverter(n_segments=self.n_segments)
        
        # Process each image and save immediately
        all_labels = []
        all_image_paths = []
        graph_idx = 0
        
        for class_name in CLASS_NAMES:
            class_idx = CLASS_TO_IDX[class_name]
            image_paths = data_dict[class_name]
            
            print(f"\nProcessing {class_name} ({len(image_paths)} images)...")
            
            for img_path in tqdm(image_paths, desc=f"  {class_name}"):
                try:
                    # Load and convert image
                    image = Image.open(img_path).convert('RGB')
                    graph = converter.convert(image, label=class_idx)
                    
                    # Save immediately to disk
                    save_path = osp.join(self.processed_dir, f'data_{graph_idx}.pt')
                    torch.save(graph, save_path)
                    
                    # Track metadata
                    all_labels.append(class_idx)
                    all_image_paths.append(img_path)
                    
                    graph_idx += 1
                    
                    # Explicitly delete to free RAM
                    del image
                    del graph
                
                except Exception as e:
                    print(f"    Warning: Failed to process {img_path}: {e}")
                    continue
        
        print(f"\n✓ Processed {graph_idx} graphs")
        
        # Create train/val/test splits
        print("\nCreating train/val/test splits...")
        train_idx, val_idx, test_idx = self._create_splits(graph_idx, all_labels)
        
        # Save splits
        split_path = osp.join(self.processed_dir, 'split_indices.pt')
        torch.save({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }, split_path)
        print(f"  Saved splits to: {split_path}")
        
        # Save metadata
        metadata = {
            'num_graphs': graph_idx,
            'class_names': CLASS_NAMES,
            'n_segments': self.n_segments,
            'labels': all_labels,
            'image_paths': all_image_paths,
            'num_train': len(train_idx),
            'num_val': len(val_idx),
            'num_test': len(test_idx)
        }
        
        metadata_path = osp.join(self.processed_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata to: {metadata_path}")
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
    
    def _find_images(self):
        """Find all images in the image directory."""
        data_dict = {class_name: [] for class_name in CLASS_NAMES}
        
        # Try LabelledRice/Labelled structure
        labelled_dir = Path(self.image_dir) / "LabelledRice" / "Labelled"
        if not labelled_dir.exists():
            labelled_dir = Path(self.image_dir) / "Labelled"
        
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
                split_dir = Path(self.image_dir) / "RiceDiseaseDataset" / split_name
                if split_dir.exists():
                    for class_name in CLASS_NAMES:
                        class_dir = split_dir / class_name
                        if class_dir.exists():
                            images = list(class_dir.glob("*.jpg")) + \
                                    list(class_dir.glob("*.jpeg")) + \
                                    list(class_dir.glob("*.png"))
                            data_dict[class_name].extend([str(p) for p in images])
        
        return data_dict
    
    def _create_splits(self, num_graphs, labels):
        """Create stratified train/val/test splits."""
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(num_graphs)
        labels_np = np.array(labels)
        
        # 70% train, 30% temp
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, stratify=labels_np, random_state=self.seed
        )
        
        # Split temp into 50% val, 50% test
        temp_labels = labels_np[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=temp_labels, random_state=self.seed
        )
        
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val:   {len(val_idx)} samples")
        print(f"  Test:  {len(test_idx)} samples")
        
        return (
            torch.from_numpy(train_idx),
            torch.from_numpy(val_idx),
            torch.from_numpy(test_idx)
        )
    
    def len(self):
        """Return the number of graphs in this split."""
        return len(self.data_indices)
    
    def get(self, idx):
        """
        Load and return a single graph.
        
        Args:
            idx: Index within the current split
        
        Returns:
            PyG Data object
        """
        # Map split index to global index
        global_idx = self.data_indices[idx].item()
        
        # Load graph from disk
        data_path = osp.join(self.processed_dir, f'data_{global_idx}.pt')
        data = torch.load(data_path)
        
        return data
    
    @property
    def num_classes(self):
        """Number of disease classes."""
        return len(CLASS_NAMES)


def create_dataset_zip(processed_dir, output_zip_path):
    """
    Create a zip archive of all processed .pt files.
    
    Args:
        processed_dir: Directory containing processed .pt files
        output_zip_path: Path for output zip file
    
    Returns:
        output_zip_path: Path to created zip file
    """
    print(f"\nCreating zip archive: {output_zip_path}")
    print("-" * 60)
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        pt_files = list(Path(processed_dir).glob("*.pt"))
        json_files = list(Path(processed_dir).glob("*.json"))
        
        files_to_zip = pt_files + json_files
        
        for file_path in tqdm(files_to_zip, desc="Compressing"):
            arcname = file_path.name
            zipf.write(file_path, arcname)
    
    # Get zip file size
    zip_size_mb = Path(output_zip_path).stat().st_size / (1024 * 1024)
    
    print("-" * 60)
    print(f"✓ Zip archive created successfully!")
    print(f"  Location: {output_zip_path}")
    print(f"  Size: {zip_size_mb:.2f} MB")
    print(f"  Files: {len(files_to_zip)}")
    
    return output_zip_path


# For backward compatibility with fairseq registration
from graphormer.data import register_dataset
from graphormer.data.pyg_datasets import GraphormerPYGDataset


@register_dataset("rice_diseases")
def create_rice_diseases_dataset(root="/content/rice_diseases_graphs", seed=42):
    """
    Create rice diseases dataset for Graphormer/fairseq.
    
    This function is called by fairseq when using --dataset-name rice_diseases.
    
    Args:
        root: Root directory containing processed graphs
        seed: Random seed (not used, splits are pre-determined)
    
    Returns:
        GraphormerPYGDataset compatible with fairseq
    """
    train_set = RiceDiseasesDataset(root=root, split='train')
    valid_set = RiceDiseasesDataset(root=root, split='val')
    test_set = RiceDiseasesDataset(root=root, split='test')
    
    return GraphormerPYGDataset(
        None, seed, None, None, None,
        train_set, valid_set, test_set
    )


if __name__ == "__main__":
    print("Rice Diseases Dataset - PyTorch Geometric Format")
    print("\nUsage:")
    print("  # Process images and create dataset")
    print("  dataset = RiceDiseasesDataset(")
    print("      root='/content/rice_diseases_graphs',")
    print("      image_dir='/content/rice_diseases_data',")
    print("      split='train',")
    print("      n_segments=75,")
    print("      force_process=True  # Set to False after first processing")
    print("  )")
    print("\n  # Create zip archive")
    print("  create_dataset_zip(")
    print("      '/content/rice_diseases_graphs/processed',")
    print("      '/content/rice_diseases_graphs.zip'")
    print("  )")
