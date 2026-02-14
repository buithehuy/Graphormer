# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist

# Import custom datasets
RiceDiseasesDataset = None
try:
    # Strategy 1: Direct import if in path
    from examples.rice_diseases.rice_diseases_dataset import RiceDiseasesDataset
except ImportError:
    try:
        # Strategy 2: Add absolute path based on graphormer location
        import sys
        import os
        # Get Graphormer root (3 levels up from this file)
        graphormer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        examples_path = os.path.join(graphormer_root, 'examples')
        
        if os.path.exists(examples_path) and examples_path not in sys.path:
            sys.path.insert(0, examples_path)
        
        from rice_diseases.rice_diseases_dataset import RiceDiseasesDataset
    except ImportError:
        # Strategy 3: Try without examples prefix
        try:
            import sys
            import os
            rice_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'rice_diseases'))
            if os.path.exists(rice_path) and rice_path not in sys.path:
                sys.path.insert(0, rice_path)
            
            from rice_diseases_dataset import RiceDiseasesDataset
        except ImportError:
            pass  # RiceDiseasesDataset remains None



class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        elif name == "rice_diseases":
            # Custom rice diseases dataset
            if RiceDiseasesDataset is None:
                raise ImportError("RiceDiseasesDataset not found. Make sure examples/rice_diseases is in path.")
            
            # Get root from params or use default
            rice_root = "/content/Graphormer/examples/rice_diseases/rice_diseases_graphs"
            for param in params:
                if "=" in param:
                    pname, value = param.split("=")
                    if pname == "root":
                        rice_root = value
            
            # Create train/val/test splits like ZINC
            train_set = RiceDiseasesDataset(root=rice_root, split='train')
            valid_set = RiceDiseasesDataset(root=rice_root, split='val')
            test_set = RiceDiseasesDataset(root=rice_root, split='test')
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )
