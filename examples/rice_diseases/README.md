# Rice Diseases Image-to-Graph Conversion

This module converts rice disease images to graph structures using superpixel segmentation (SLIC algorithm), making them compatible with Microsoft's Graphormer fairseq framework for graph neural network training.

## Overview

**Input**: Rice disease images (4 classes: BrownSpot, Healthy, Hispa, LeafBlast)  
**Output**: Graph structures where:
- **Nodes**: Superpixels with RGB color features
- **Edges**: Adjacency between neighboring superpixels  
- **Edge Features**: Color differences between adjacent superpixels
- **Graph Labels**: Disease class (0-3)

## Dataset Structure

The dataset should be organized as follows:

```
rice-diseases-image-dataset/
├── LabelledRice/
│   └── Labelled/
│       ├── BrownSpot/
│       ├── Healthy/
│       ├── Hispa/
│       └── LeafBlast/
└── RiceDiseaseDataset/
    ├── train/
    └── validation/
```

## Installation & Setup (Google Colab)

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone Graphormer Repository

```bash
!git clone https://github.com/microsoft/Graphormer.git
%cd Graphormer
```

### Step 3: Install Dependencies

```bash
# Install fairseq and Graphormer
!bash install_updated.sh

# Install rice_diseases specific dependencies
!pip install -r examples/rice_diseases/requirements.txt
```

### Step 4: Copy and Extract Dataset

```python
import sys
sys.path.append('/content/Graphormer')

from examples.rice_diseases.colab_setup import copy_and_extract_dataset

# Copy from Drive and extract to local storage
data_dir = copy_and_extract_dataset(
    drive_zip_path="MyDrive/Rice_Diseases_Dataset/rice-diseases-image-dataset.zip",
    extract_dir="/content/rice_diseases_data"
)
```

## Usage

### Convert Images to Graphs

```python
from examples.rice_diseases.rice_diseases_dataset import RiceDiseasesGraphDataset

# Create dataset (this will process and cache graphs)
dataset = RiceDiseasesGraphDataset(
    data_dir="/content/rice_diseases_data",
    cache_dir="/content/rice_diseases_graphs",
    n_segments=75,  # Number of superpixels
    force_reprocess=False,  # Set True to reprocess even if cache exists
    use_labelled=True,  # Use LabelledRice/Labelled structure
    seed=42
)

print(f"Dataset created: {len(dataset)} samples")
print(f"Train: {len(dataset.train_idx)}, Val: {len(dataset.valid_idx)}, Test: {len(dataset.test_idx)}")
```

**Note**: The first run will process all images and save to cache. Subsequent runs will load from cache (much faster).

### Generate Visualizations

```python
from examples.rice_diseases.visualize_graphs import (
    create_sample_visualizations,
    plot_dataset_statistics
)

# Create sample visualizations (2 per class)
create_sample_visualizations(
    dataset,
    output_dir="/content/Graphormer/examples/rice_diseases/visualizations",
    samples_per_class=2
)

# Plot dataset statistics
fig = plot_dataset_statistics(dataset, save_path="dataset_stats.png")
```

### Use with Graphormer/Fairseq Training

The dataset is registered as `"rice_diseases"` and can be used directly with fairseq training scripts:

```bash
fairseq-train \
  --user-dir ../../graphormer \
  --num-workers 2 \
  --ddp-backend=legacy_ddp \
  --dataset-name rice_diseases \
  --dataset-source pyg \
  --task graph_prediction \
  --criterion multiclass_cross_entropy \
  --arch graphormer_base \
  --num-classes 4 \
  --batch-size 32 \
  ...
```

## Module Structure

```
examples/rice_diseases/
├── rice_image_to_graph.py       # Core image-to-graph converter (SLIC)
├── rice_diseases_dataset.py     # Dataset wrapper with caching
├── colab_setup.py               # Colab utilities (Drive copy, extraction)
├── visualize_graphs.py          # Visualization utilities
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── rice_diseases_convert_data.ipynb  # Main notebook for data conversion
```

## Dataset Statistics

- **4 classes**: BrownSpot, Healthy, Hispa, LeafBlast
- **Split**: 70% train, 15% validation, 15% test (stratified by class)
- **Graph size**: ~50-100 nodes per graph (depending on `n_segments` parameter)

## Graph Structure Details

Each image is converted to a graph using the following process:

1. **Superpixel Segmentation**: SLIC algorithm generates ~75 superpixels per image
2. **Node Features**: 3D RGB color (mean color of each superpixel)
3. **Edge Creation**: Superpixels that share a boundary are connected
4. **Edge Features**: Euclidean distance between RGB colors of connected superpixels

## Caching System

Processed graphs are cached to disk to avoid reprocessing:

- **Cache location**: `/content/rice_diseases_graphs/`
- **Cache file**: `processed_graphs_n75.pkl` (n depends on `n_segments`)
- **Reprocessing**: Set `force_reprocess=True` to regenerate cache

## Troubleshooting

### RAM Overflow on Colab

- **Symptom**: Colab crashes during dataset extraction
- **Solution**: The `copy_and_extract_dataset()` function copies the zip from Drive to local `/tmp` first, then extracts. This avoids Drive RAM issues.

### Missing Dataset

- **Symptom**: `FileNotFoundError` when running setup
- **Solution**: Ensure the zip file is at `MyDrive/Rice_Diseases_Dataset/rice-diseases-image-dataset.zip` in your Google Drive

### Slow First Run

- **Symptom**: First dataset creation takes a long time
- **Solution**: This is expected - the module processes all images and converts to graphs. Subsequent runs will load from cache and be much faster.

### Import Errors

- **Symptom**: `ModuleNotFoundError` for various packages
- **Solution**: Ensure all dependencies are installed:
  ```bash
  !pip install -r examples/rice_diseases/requirements.txt
  ```

## Next Steps: Training with Graphormer

After converting the dataset, you can train a Graphormer model for rice disease classification. Example training script coming soon.

## Citation

If you use this dataset conversion module, please cite:

1. **Graphormer**: Original Graphormer paper
2. **Rice Diseases Dataset**: Kaggle dataset by minhhuy2810

## License

This module follows the same license as the Graphormer repository (MIT License).
