"""
Image to Graph Converter using Superpixel Segmentation (SLIC Algorithm)

This module converts images to graph structures where:
- Nodes are superpixels with RGB color features
- Edges connect adjacent superpixels
- Edge features are color differences between adjacent superpixels
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
from PIL import Image
import torch
from torch_geometric.data import Data
from scipy.spatial import distance
from collections import defaultdict


class ImageToGraphConverter:
    """
    Convert images to graph structures using superpixel segmentation.
    
    Args:
        n_segments (int): Approximate number of superpixels to generate
        compactness (float): Balance between color similarity and space proximity
        sigma (float): Gaussian smoothing parameter before segmentation
    """
    
    def __init__(self, n_segments=75, compactness=10, sigma=1):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
    
    def extract_superpixels(self, image):
        """
        Extract superpixels from an image using SLIC algorithm.
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
        
        Returns:
            segments: numpy array (H, W) with segment labels
            n_segments: actual number of segments created
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is float in [0, 1]
        image = img_as_float(image)
        
        # Apply SLIC segmentation
        segments = slic(
            image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0
        )
        
        return segments, segments.max() + 1
    
    def build_adjacency(self, segments):
        """
        Build adjacency matrix from superpixel segments.
        
        Args:
            segments: numpy array (H, W) with segment labels
        
        Returns:
            adjacency: dict mapping segment_id -> set of adjacent segment_ids
        """
        adjacency = defaultdict(set)
        
        h, w = segments.shape
        
        # Check horizontal neighbors
        for i in range(h):
            for j in range(w - 1):
                seg1, seg2 = segments[i, j], segments[i, j + 1]
                if seg1 != seg2:
                    adjacency[seg1].add(seg2)
                    adjacency[seg2].add(seg1)
        
        # Check vertical neighbors
        for i in range(h - 1):
            for j in range(w):
                seg1, seg2 = segments[i, j], segments[i + 1, j]
                if seg1 != seg2:
                    adjacency[seg1].add(seg2)
                    adjacency[seg2].add(seg1)
        
        return adjacency
    
    def compute_node_features(self, image, segments, n_segments):
        """
        Compute node features as mean RGB color of each superpixel.
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            segments: numpy array (H, W) with segment labels
            n_segments: number of segments
        
        Returns:
            node_features: numpy array (n_segments, 3) with RGB features
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure float representation
        image = img_as_float(image)
        
        node_features = np.zeros((n_segments, 3), dtype=np.float32)
        
        for seg_id in range(n_segments):
            mask = segments == seg_id
            if mask.sum() > 0:
                node_features[seg_id] = image[mask].mean(axis=0)
        
        return node_features
    
    def compute_edge_features(self, node_features, edge_index, n_bins=10):
        """
        Compute edge features as quantized color difference bins.
        
        Args:
            node_features: numpy array (n_segments, 3)
            edge_index: numpy array (2, num_edges)
            n_bins: number of bins to quantize edge features into
        
        Returns:
            edge_features: numpy array (num_edges,) with integer bin indices
        """
        num_edges = edge_index.shape[1]
        edge_distances = np.zeros(num_edges, dtype=np.float32)
        
        # Compute actual distances first
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            color_diff = distance.euclidean(
                node_features[src], 
                node_features[dst]
            )
            edge_distances[i] = color_diff
        
        # Quantize into bins (convert to integer edge types)
        # Max possible RGB distance is sqrt(3) ≈ 1.732
        max_dist = np.sqrt(3.0)
        # Bin edges: [0, max_dist/n_bins, 2*max_dist/n_bins, ..., max_dist]
        edge_features = np.digitize(edge_distances, 
                                     bins=np.linspace(0, max_dist, n_bins+1)) - 1
        edge_features = np.clip(edge_features, 0, n_bins-1).astype(np.int64)
        
        return edge_features
    
    def convert(self, image, label=None):
        """
        Convert an image to a graph structure.
        
        Args:
            image: PIL Image or numpy array
            label: Optional label for the graph
        
        Returns:
            data: PyTorch Geometric Data object with:
                - x: node features (n_nodes, 3)
                - edge_index: edge connectivity (2, num_edges)
                - edge_attr: edge features (num_edges, 1)
                - y: graph label (optional)
        """
        target_size = (224, 224)  # Giảm từ ví dụ 512x512
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        # Extract superpixels
        segments, n_segments = self.extract_superpixels(image)
        
        # Build adjacency
        adjacency = self.build_adjacency(segments)
        
        # Compute node features
        node_features = self.compute_node_features(image, segments, n_segments)
        
        # Build edge index
        edge_list = []
        for src in adjacency:
            for dst in adjacency[src]:
                edge_list.append([src, dst])
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        
        # Compute edge features (now returns integers)
        edge_features = self.compute_edge_features(node_features, edge_index)
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.long)  # Changed to long!
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        
        return data
    
    def get_segmentation_mask(self, image):
        """
        Get the superpixel segmentation mask for visualization.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            segments: numpy array (H, W) with segment labels
            boundaries: numpy array (H, W, 3) with boundary overlay
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        image_float = img_as_float(image_np)
        segments = slic(
            image_float,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0
        )
        
        boundaries = mark_boundaries(image_float, segments, color=(1, 0, 0))
        
        return segments, boundaries
