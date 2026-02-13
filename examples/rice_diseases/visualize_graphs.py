"""
Visualization Utilities for Rice Diseases Graph Dataset

Generate visualizations showing the image-to-graph conversion process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
from PIL import Image
import os
from pathlib import Path

from .rice_image_to_graph import ImageToGraphConverter


def visualize_image_to_graph(image_path, converter, save_path=None, show_graph_structure=True):
    """
    Create a visualization showing: original image -> superpixels -> graph.
    
    Args:
        image_path: Path to input image
        converter: ImageToGraphConverter instance
        save_path: Optional path to save the visualization
        show_graph_structure: If True, include graph structure visualization
    
    Returns:
        fig: matplotlib figure
    """
    # Load image
    image = Image.open (image_path).convert('RGB')
    image_np = np.array(image)
    
    # Convert to graph
    graph = converter.convert(image)
    segments, boundaries = converter.get_segmentation_mask(image)
    
    # Create figure
    if show_graph_structure:
        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 3, figure=fig)
    else:
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig)
    
    # Plot 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Superpixel segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(boundaries)
    ax2.set_title(f'Superpixel Segmentation\n({graph.x.shape[0]} segments)', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Plot 3: Graph structure (if requested)
    if show_graph_structure:
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        num_nodes = graph.x.shape[0]
        for i in range(num_nodes):
            G.add_node(i)
        
        # Add edges
        edge_index = graph.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:  # Add each edge only once (undirected)
                G.add_edge(src, dst)
        
        # Node colors from RGB features
        node_colors = graph.x.numpy()  # RGB values in [0, 1]
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            node_size=100,
            ax=ax3
        )
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            alpha=0.3,
            width=0.5,
            ax=ax3
        )
        
        ax3.set_title(f'Graph Structure\n({graph.x.shape[0]} nodes, {G.number_of_edges()} edges)',
                     fontsize=14, fontweight='bold')
        ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")
    
    return fig


def create_sample_visualizations(
    dataset,
    output_dir="visualizations",
    samples_per_class=2,
    n_segments=75
):
    """
    Create sample visualizations for each disease class.
    
    Args:
        dataset: RiceDiseasesGraphDataset instance
        output_dir: Directory to save visualizations
        samples_per_class: Number of samples to visualize per class
        n_segments: Number of superpixels for visualization
    
    Returns:
        list of saved file paths
    """
    from .rice_diseases_dataset import CLASS_NAMES
    
    print(f"Creating sample visualizations...")
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per class: {samples_per_class}")
    print("-" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    converter = ImageToGraphConverter(n_segments=n_segments)
    saved_files = []
    
    # Group indices by class
    class_indices = {class_name: [] for class_name in CLASS_NAMES}
    for idx, label in enumerate(dataset.labels):
        class_name = CLASS_NAMES[label]
        class_indices[class_name].append(idx)
    
    # Create visualizations
    for class_name in CLASS_NAMES:
        indices = class_indices[class_name][:samples_per_class]
        
        print(f"\n{class_name}:")
        
        for i, idx in enumerate(indices):
            image_path = dataset.image_paths[idx]
            save_path = os.path.join(
                output_dir,
                f"{class_name}_sample_{i+1}.png"
            )
            
            visualize_image_to_graph(
                image_path,
                converter,
                save_path=save_path,
                show_graph_structure=True
            )
            
            saved_files.append(save_path)
            plt.close()  # Close figure to free memory
    
    print("\n" + "=" * 60)
    print(f"✓ Created {len(saved_files)} visualizations")
    print(f"✓ Saved to: {output_dir}")
    print("=" * 60)
    
    return saved_files


def plot_dataset_statistics(dataset, save_path=None):
    """
    Plot dataset statistics (class distribution, splits, etc.).
    
    Args:
        dataset: RiceDiseasesGraphDataset instance
        save_path: Optional path to save the plot
    
    Returns:
        fig: matplotlib figure
    """
    from .rice_diseases_dataset import CLASS_NAMES
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Class distribution (overall)
    labels_np = np.array(dataset.labels)
    class_counts = [np.sum(labels_np == i) for i in range(len(CLASS_NAMES))]
    
    axes[0].bar(CLASS_NAMES, class_counts, color='skyblue', edgecolor='black')
    axes[0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_xlabel('Disease Class')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, count in enumerate(class_counts):
        axes[0].text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Split distribution
    split_names = ['Train', 'Validation', 'Test']
    split_counts = [
        len(dataset.train_idx),
        len(dataset.valid_idx),
        len(dataset.test_idx)
    ]
    
    axes[1].bar(split_names, split_counts, color=['green', 'orange', 'red'], 
                edgecolor='black', alpha=0.7)
    axes[1].set_title('Train/Val/Test Split', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Samples')
    
    for i, count in enumerate(split_counts):
        pct = 100 * count / len(dataset)
        axes[1].text(i, count + 10, f'{count}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Class distribution per split
    splits = {
        'Train': dataset.train_idx,
        'Val': dataset.valid_idx,
        'Test': dataset.test_idx
    }
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    for i, (split_name, indices) in enumerate(splits.items()):
        split_labels = labels_np[indices]
        split_class_counts = [np.sum(split_labels == j) for j in range(len(CLASS_NAMES))]
        
        offset = width * (i - 1)
        axes[2].bar(x + offset, split_class_counts, width, 
                   label=split_name, alpha=0.8, edgecolor='black')
    
    axes[2].set_title('Class Distribution per Split', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Number of Samples')
    axes[2].set_xlabel('Disease Class')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASS_NAMES, rotation=45)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved statistics plot: {save_path}")
    
    return fig


def display_sample_graphs(dataset, num_samples=4):
    """
    Display a grid of sample graph conversions.
    
    Args:
        dataset: RiceDiseasesGraphDataset instance
        num_samples: Number of random samples to display
    """
    from .rice_diseases_dataset import CLASS_NAMES
    
    # Random sample indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    converter = ImageToGraphConverter(n_segments=dataset.n_segments)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        image_path = dataset.image_paths[idx]
        label = dataset.labels[idx]
        class_name = CLASS_NAMES[label]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get segmentation
        segments, boundaries = converter.get_segmentation_mask(image)
        
        # Plot original
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Sample {i+1}: {class_name}', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Plot segmentation
        axes[i, 1].imshow(boundaries)
        axes[i, 1].set_title(f'Superpixels ({segments.max() + 1} segments)', fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities for Rice Diseases Graph Dataset")
    print("\nAvailable functions:")
    print("  - visualize_image_to_graph(): Single image visualization")
    print("  - create_sample_visualizations(): Generate samples for all classes")
    print("  - plot_dataset_statistics(): Dataset distribution plots")
    print("  - display_sample_graphs(): Interactive sample display")
