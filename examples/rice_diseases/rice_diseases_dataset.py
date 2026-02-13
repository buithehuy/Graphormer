"""
Rice Diseases Graph Dataset

Custom dataset for rice disease classification using graph representations.
Compatible with Graphormer fairseq framework.
"""

import os
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from graphormer.data import register_dataset
from .rice_image_to_graph import ImageToGraphConverter


# Disease class mapping
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class RiceDiseasesGraphDataset:
    """
    Dataset for rice disease images converted to graphs.
    
    Args:
        data_dir: Root directory containing the dataset
        cache_dir: Directory to cache preprocessed graphs
        n_segments: Number of superpixels for graph conversion
        force_reprocess: If True, reprocess even if cache exists
        use_labelled: If True, use LabelledRice/Labelled structure
                     If False, use RiceDiseaseDataset/train+validation
        seed: Random seed for train/val/test split
    """
    
    def __init__(
        self,
        data_dir="/content/rice_diseases_data",
        cache_dir="/content/rice_diseases_graphs",
        n_segments=75,
        force_reprocess=False,
        use_labelled=True,
        seed=42
    ):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.n_segments = n_segments
        self.force_reprocess = force_reprocess
        self.use_labelled = use_labelled
        self.seed = seed
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize converter
        self.converter = ImageToGraphConverter(n_segments=n_segments)
        
        # Load or process dataset
        self.graphs, self.labels, self.image_paths = self._load_or_process_dataset()
        
        # Create train/val/test splits
        self.train_idx, self.valid_idx, self.test_idx = self._create_splits()
    
    def _find_dataset_structure(self):
        """
        Automatically find the dataset structure.
        
        Returns:
            dict: {class_name: list of image paths}
        """
        data_dict = {class_name: [] for class_name in CLASS_NAMES}
        
        if self.use_labelled:
            # Look for LabelledRice/Labelled structure
            labelled_dir = Path(self.data_dir) / "LabelledRice" / "Labelled"
            
            if not labelled_dir.exists():
                # Try without LabelledRice prefix
                labelled_dir = Path(self.data_dir) / "Labelled"
            
            if labelled_dir.exists():
                for class_name in CLASS_NAMES:
                    class_dir = labelled_dir / class_name
                    if class_dir.exists():
                        images = list(class_dir.glob("*.jpg")) + \
                                list(class_dir.glob("*.jpeg")) + \
                                list(class_dir.glob("*.png"))
                        data_dict[class_name] = [str(p) for p in images]
        else:
            # Use RiceDiseaseDataset/train and validation structure
            train_dir = Path(self.data_dir) / "RiceDiseaseDataset" / "train"
            val_dir = Path(self.data_dir) / "RiceDiseaseDataset" / "validation"
            
            for split_dir in [train_dir, val_dir]:
                if split_dir.exists():
                    for class_name in CLASS_NAMES:
                        class_dir = split_dir / class_name
                        if class_dir.exists():
                            images = list(class_dir.glob("*.jpg")) + \
                                    list(class_dir.glob("*.jpeg")) + \
                                    list(class_dir.glob("*.png"))
                            data_dict[class_name].extend([str(p) for p in images])
        
        return data_dict
    
    def _load_or_process_dataset(self):
        """
        Load preprocessed graphs from cache or process images.
        
        Returns:
            graphs: list of PyG Data objects
            labels: list of integer labels
            image_paths: list of image file paths
        """
        cache_file = Path(self.cache_dir) / f"processed_graphs_n{self.n_segments}.pkl"
        
        # Try to load from cache
        if cache_file.exists() and not self.force_reprocess:
            print(f"Loading preprocessed graphs from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✓ Loaded {len(data['graphs'])} graphs from cache")
            return data['graphs'], data['labels'], data['image_paths']
        
        # Process images
        print(f"Processing images to graphs (n_segments={self.n_segments})...")
        
        data_dict = self._find_dataset_structure()
        
        graphs = []
        labels = []
        image_paths = []
        
        for class_name in CLASS_NAMES:
            class_idx = CLASS_TO_IDX[class_name]
            class_images = data_dict[class_name]
            
            print(f"\nProcessing {class_name} ({len(class_images)} images)...")
            
            for img_path in tqdm(class_images, desc=f"  {class_name}"):
                try:
                    # Load image
                    image = Image.open(img_path).convert('RGB')
                    
                    # Convert to graph
                    graph = self.converter.convert(image, label=class_idx)
                    
                    graphs.append(graph)
                    labels.append(class_idx)
                    image_paths.append(img_path)
                
                except Exception as e:
                    print(f"    Warning: Failed to process {img_path}: {e}")
                    continue
        
        print(f"\n✓ Processed {len(graphs)} graphs total")
        
        # Save to cache
        print(f"Saving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'graphs': graphs,
                'labels': labels,
                'image_paths': image_paths,
                'class_names': CLASS_NAMES,
                'n_segments': self.n_segments
            }, f)
        
        print("✓ Cache saved")
        
        return graphs, labels, image_paths
    
    def _create_splits(self):
        """
        Create train/val/test splits.
        
        If use_labelled=False, uses existing train/val structure:
        - All train images -> train
        - 50% validation images -> validation
        - 50% validation images -> test
        
        If use_labelled=True, creates new split:
        - 70% -> train
        - 15% -> validation
        - 15% -> test
        
        Returns:
            train_idx, valid_idx, test_idx: numpy arrays of indices
        """
        n_samples = len(self.graphs)
        indices = np.arange(n_samples)
        
        if not self.use_labelled:
            # Split based on existing train/val structure
            # This requires tracking which came from train vs val
            # For now, do stratified split
            pass
        
        # Stratified split by class
        labels_np = np.array(self.labels)
        
        # First split: 70% train, 30% temp
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.3,
            stratify=labels_np,
            random_state=self.seed
        )
        
        # Second split: split temp into 50% val, 50% test
        temp_labels = labels_np[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            stratify=temp_labels,
            random_state=self.seed
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Valid: {len(val_idx)} samples")
        print(f"  Test:  {len(test_idx)} samples")
        
        return train_idx, val_idx, test_idx
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    @property
    def num_classes(self):
        return len(CLASS_NAMES)
    
    @property
    def train_data(self):
        return [self.graphs[i] for i in self.train_idx]
    
    @property
    def valid_data(self):
        return [self.graphs[i] for i in self.valid_idx]
    
    @property
    def test_data(self):
        return [self.graphs[i] for i in self.test_idx]


@register_dataset("rice_diseases")
def create_rice_diseases_dataset(
    data_dir="/content/rice_diseases_data",
    cache_dir="/content/rice_diseases_graphs",
    n_segments=75,
    seed=42
):
    """
    Create rice diseases graph dataset compatible with Graphormer.
    
    This function can be called by fairseq with --dataset-name rice_diseases
    
    Returns:
        dict with keys:
            - dataset: RiceDiseasesGraphDataset object
            - train_idx: training indices
            - valid_idx: validation indices
            - test_idx: test indices
            - source: "pyg" (PyTorch Geometric)
    """
    dataset = RiceDiseasesGraphDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        n_segments=n_segments,
        seed=seed
    )
    
    return {
        "dataset": dataset,
        "train_idx": dataset.train_idx,
        "valid_idx": dataset.valid_idx,
        "test_idx": dataset.test_idx,
        "source": "pyg",
        "num_classes": dataset.num_classes
    }


if __name__ == "__main__":
    # Example usage
    print("Creating Rice Diseases Graph Dataset...")
    
    dataset = RiceDiseasesGraphDataset(
        data_dir="/content/rice_diseases_data",
        cache_dir="/content/rice_diseases_graphs"
    )
    
    print(f"\nDataset created successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"  Class names: {CLASS_NAMES}")
    
    # Show sample graph
    sample_graph = dataset[0]
    print(f"\nSample graph structure:")
    print(f"  Nodes: {sample_graph.x.shape[0]}")
    print(f"  Edges: {sample_graph.edge_index.shape[1]}")
    print(f"  Node features: {sample_graph.x.shape}")
    print(f"  Edge features: {sample_graph.edge_attr.shape}")
    print(f"  Label: {sample_graph.y.item()} ({CLASS_NAMES[sample_graph.y.item()]})")
