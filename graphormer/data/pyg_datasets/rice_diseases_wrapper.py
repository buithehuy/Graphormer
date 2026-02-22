# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Wrapper for RiceDiseasesDataset to make it importable from pyg_datasets.
This avoids complex import path issues.
"""

import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
from pathlib import Path


CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']


class RiceDiseasesDatasetWrapper(Dataset):
    """
    Wrapper for rice diseases graph dataset.
    Loads individual .pt graph files from processed directory.
    """
    
    def __init__(self, root="/content/Graphormer/examples/rice_diseases/rice_diseases_graphs", split='train'):
        """
        Args:
            root: Root directory containing processed/ folder
            split: 'train', 'val', or 'test'
        """
        self.root = root
        self.split = split
        super().__init__(root)
        
        # Load split indices
        split_path = osp.join(self.processed_dir, 'split_indices.pt')
        if not osp.exists(split_path):
            raise FileNotFoundError(f"Split indices not found at {split_path}. Please process the dataset first.")
        
        splits = torch.load(split_path)
        if split == 'train':
            self.data_indices = splits['train_idx']
        elif split == 'val':
            self.data_indices = splits['val_idx']
        elif split == 'test':
            self.data_indices = splits['test_idx']
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        # Not used but required by Dataset interface
        return ['data_0.pt']
    
    def len(self):
        return len(self.data_indices)
    
    def get(self, idx):
        """Load graph from disk on-demand."""
        global_idx = self.data_indices[idx].item()
        data_path = osp.join(self.processed_dir, f'data_{global_idx}.pt')
        return torch.load(data_path)


# ─────────────────────────────────────────────────────────────────────────────
# Float-safe Graphormer dataset wrapper for RGB node features
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache
from graphormer.data.pyg_datasets.pyg_dataset import GraphormerPYGDataset
from graphormer.data.wrapper import preprocess_item_float


class RiceDiseasesGraphormerDataset(GraphormerPYGDataset):
    """
    Subclass of GraphormerPYGDataset that uses preprocess_item_float.

    The default GraphormerPYGDataset.__getitem__ calls preprocess_item which
    runs convert_to_single_emb(x), adding integer offsets (+1, +513, +1025)
    to the RGB float features — completely corrupting them.
    This subclass bypasses that by using preprocess_item_float instead.
    """

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item_float(item)  # float-safe: no convert_to_single_emb on x
        else:
            raise TypeError("index to a RiceDiseasesGraphormerDataset can only be an integer.")

