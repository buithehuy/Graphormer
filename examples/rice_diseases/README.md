# Rice Diseases Image-to-Graph Conversion

This module converts rice disease images to graph structures using superpixel segmentation (SLIC algorithm), compatible with Microsoft's Graphormer fairseq framework.

## Overview

**Input**: Rice disease images (4 classes: BrownSpot, Healthy, Hispa, LeafBlast)  
**Output**: Individual .pt files + zip archive (PyTorch Geometric format, fairseq-compatible)

**Graph Structure**:
- **Nodes**: Superpixels with RGB color features
- **Edges**: Adjacency between neighboring superpixels  
- **Edge Features**: Color differences
- **Labels**: Disease class (0-3)

## Dataset Format (PyTorch Geometric)

```
rice_diseases_graphs/
├── processed/
│   ├── data_0.pt          # Individual graph files
│   ├── data_1.pt
│   ├── ...
│   ├── data_N.pt
│   ├── split_indices.pt   # Train/val/test indices
│   └── metadata.json      # Dataset metadata
└── rice_diseases_graphs.zip  # Complete archive
```

## Quick Start (Google Colab)

### 1. Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/microsoft/Graphormer.git
%cd Graphormer
!bash install_updated.sh
!pip install -r examples/rice_diseases/requirements.txt
```

### 2. Process Dataset

Run the notebook **`rice_diseases_convert_data.ipynb`** or use Python:

```python
from examples.rice_diseases.colab_setup import copy_and_extract_dataset
from examples.rice_diseases.rice_diseases_dataset import RiceDiseasesDataset, create_dataset_zip

# Copy and extract images
data_dir = copy_and_extract_dataset(
    drive_zip_path="MyDrive/Rice_Diseases_Dataset/rice-diseases-image-dataset.zip",
    extract_dir="/content/rice_diseases_data"
)

# Process images → save as .pt files
dataset = RiceDiseasesDataset(
    root="/content/rice_diseases_graphs",
    image_dir=data_dir,
    n_segments=75,
    force_process=True  # First run only
)

# Create zip archive
create_dataset_zip(
    processed_dir="/content/rice_diseases_graphs/processed",
    output_zip_path="/content/rice_diseases_graphs.zip"
)
```

**Note**: Processing is RAM-efficient - each graph is saved and deleted immediately.

### 3. Train with Graphormer

```bash
cd /content/Graphormer/examples/rice_diseases
bash rice_diseases.sh
```

Or use fairseq-train directly:

```bash
fairseq-train \
  --user-dir ../../graphormer \
  --dataset-name rice_diseases \
  --dataset-source pyg \
  --task graph_prediction \
  --criterion multiclass_cross_entropy \
  --num-classes 4 \
  --batch-size 32 \
  ...
```

## Module Structure

```
examples/rice_diseases/
├── rice_image_to_graph.py           # SLIC superpixel converter
├── rice_diseases_dataset.py         # PyG Dataset class
├── colab_setup.py                   # Drive utilities
├── visualize_graphs.py              # Visualization tools
├── rice_diseases_convert_data.ipynb # Main notebook
├── rice_diseases.sh                 # Training script
├── requirements.txt
└── README.md
```

## Key Features

### RAM-Efficient Processing

The `process()` method processes images **one-by-one**:

```python
# DON'T do this (loads all in memory):
# graphs = [convert(img) for img in images]

# DO this (saves and deletes immediately):
for i, img_path in enumerate(image_paths):
    graph = converter.convert(image)
    torch.save(graph, f'data_{i}.pt')
    del graph  # Free RAM
```

### PyTorch Geometric Compatible

```python
class RiceDiseasesDataset(Dataset):  # Inherits from PyG Dataset
    def process(self):
        # Save individual .pt files
        ...
    
    def get(self, idx):
        # Load single graph from disk
        data = torch.load(f'data_{idx}.pt')
        return data
```

### Fairseq Integration

Dataset is registered and loaded automatically:

```python
@register_dataset("rice_diseases")
def create_rice_diseases_dataset(root, seed):
    train_set = RiceDiseasesDataset(root=root, split='train')
    valid_set = RiceDiseasesDataset(root=root, split='val')
    test_set = RiceDiseasesDataset(root=root, split='test')
    return GraphormerPYGDataset(None, seed, None, None, None,
                                train_set, valid_set, test_set)
```

## Dataset Statistics

- **4 classes**: BrownSpot (0), Healthy (1), Hispa (2), LeafBlast (3)
- **Split**: 70% train / 15% val / 15% test (stratified)
- **Graphs**: ~50-100 nodes per graph (75 superpixels default)
- **Node features**: RGB color (3D)
- **Edge features**: Color difference (1D)

## Files Created

After processing, you'll have:

1. **Individual .pt files**: One per graph in `processed/`
2. **split_indices.pt**: Train/val/test split information
3. **metadata.json**: Dataset metadata (labels, paths, stats)
4. **rice_diseases_graphs.zip**: Complete archive (~50-200 MB)

## Usage Examples

### Load Dataset for Training

```python
from examples.rice_diseases.rice_diseases_dataset import RiceDiseasesDataset

# Load specific split
train_data = RiceDiseasesDataset(root="/content/rice_diseases_graphs", split='train')
val_data = RiceDiseasesDataset(root="/content/rice_diseases_graphs", split='val')

# Access samples
sample = train_data[0]
print(f"Nodes: {sample.x.shape[0]}")
print(f"Edges: {sample.edge_index.shape[1]}")
print(f"Label: {sample.y.item()}")
```

### Generate Visualizations

```python
from examples.rice_diseases.visualize_graphs import visualize_image_to_graph
from examples.rice_diseases.rice_image_to_graph import ImageToGraphConverter

converter = ImageToGraphConverter(n_segments=75)
fig = visualize_image_to_graph(
    image_path="path/to/image.jpg",
    converter=converter,
    save_path="visualization.png"
)
```

## Troubleshooting

### RAM Overflow

**Symptom**: Colab crashes during processing  
**Solution**: The implementation already handles this by processing one image at a time. If still crashes, reduce `n_segments` to create smaller graphs.

### Dataset Not Found

**Symptom**: `FileNotFoundError` when loading dataset  
**Solution**: Ensure you've run processing first with `force_process=True`.

### Fairseq Can't Find Dataset

**Symptom**: Error when running `fairseq-train`  
**Solution**: Make sure:
- Dataset root is `/content/rice_diseases_graphs`
- `processed/` directory contains `.pt` files
- Use `--dataset-name rice_diseases --dataset-source pyg`

## Training Script Parameters

The `rice_diseases.sh` script is pre-configured for this dataset:

- **Architecture**: `graphormer_base` (12 layers, 128 embedding dim)
- **Batch size**: 32
- **Learning rate**: 2e-4 with polynomial decay
- **Epochs**: 100 (with early stopping patience=20)
- **Loss**: multiclass_cross_entropy (4-class classification)

Adjust parameters in `rice_diseases.sh` as needed.

## Next Steps

1. ✅ Process images (run notebook)
2. ✅ Create zip archive
3. ✅ Train model (`bash rice_diseases.sh`)
4. Evaluate on test set
5. Fine-tune hyperparameters

## Citation

If using this module:

1. **Graphormer**: Cite the Graphormer paper
2. **Rice Diseases Dataset**: Credit Kaggle dataset (minhhuy2810)
3. **SLIC Algorithm**: Cite scikit-image

## License

MIT License (same as Graphormer repository)
