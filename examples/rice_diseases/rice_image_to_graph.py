"""
Image to Graph Converter using Superpixel Segmentation (SLIC Algorithm)

This module converts images to graph structures where:
- Nodes are superpixels with features: [R, G, B, cx, cy] (5-dim) or CNN (128-dim)
    - R, G, B : mean colour of the superpixel (float in [0, 1])
    - cx, cy  : normalised centroid of the superpixel (float in [0, 1])
- Edges connect adjacent superpixels
- Edge features: 1-dim (color distance, legacy) or 3-dim (color + spatial + cosine, --rich)

Optional enhancements:
    C1: --hierarchical   Add KMeans coarse nodes on top of fine superpixel nodes
    C2: --rich_edges     Use 3-dim edge features instead of 1-dim color distance
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
from PIL import Image
import torch
import torch.nn.functional as F
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
    
    def __init__(self, n_segments=75, compactness=10, sigma=1,
                 use_cnn_features=False, cnn_feature_dim=128, device='cpu'):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.use_cnn_features = use_cnn_features

        if use_cnn_features:
            from cnn_feature_extractor import CNNFeatureExtractor
            self.cnn_extractor = CNNFeatureExtractor(
                feature_dim=cnn_feature_dim, device=device
            )
    
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
        Compute node features combining mean RGB colour and normalised centroid.

        Args:
            image: PIL Image or numpy array (H, W, 3)
            segments: numpy array (H, W) with segment labels
            n_segments: number of segments

        Returns:
            node_features: numpy array (n_segments, 5) with features
                           [R, G, B, cx, cy] where (cx, cy) are normalised
                           centroid coordinates in [0, 1].
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure float representation
        image = img_as_float(image)
        h, w = segments.shape

        # --- colour features (3 dims) ---
        colour_features = np.zeros((n_segments, 3), dtype=np.float32)
        for seg_id in range(n_segments):
            mask = segments == seg_id
            if mask.sum() > 0:
                colour_features[seg_id] = image[mask].mean(axis=0)

        # --- positional features: normalised centroid (2 dims) ---
        pos_features = np.zeros((n_segments, 2), dtype=np.float32)
        props = regionprops(segments + 1)  # regionprops requires labels >= 1
        for prop in props:
            seg_id = prop.label - 1  # convert back to 0-based
            cy_px, cx_px = prop.centroid  # (row, col) → (y, x)
            pos_features[seg_id, 0] = cx_px / (w - 1)  # normalise to [0, 1]
            pos_features[seg_id, 1] = cy_px / (h - 1)

        # Concatenate: [R, G, B, cx, cy]
        node_features = np.concatenate([colour_features, pos_features], axis=1)
        return node_features
    
    def compute_edge_features(self, node_features, edge_index, n_bins=10):
        """
        Compute edge features as quantized color difference bins (legacy 1-dim).
        
        Args:
            node_features: numpy array (n_segments, 3) — RGB colours
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

    def compute_rich_edge_features(self, node_features, node_pos, edge_index, n_bins=10):
        """
        C2: Compute 3-dimensional edge features.

        Dimensions:
            [0] Color distance   — RGB Euclidean distance, max = sqrt(3)
            [1] Spatial distance — centroid Euclidean distance, max = sqrt(2) (normalised coords)
            [2] Feature similarity — cosine similarity mapped to [0, 1] and binned

        Args:
            node_features: numpy or torch array (N, D) — full node features (CNN or RGB)
            node_pos:       numpy or torch array (N, 2) — normalised centroid (cx, cy)
            edge_index:     numpy array (2, E)
            n_bins:         number of quantisation bins per dimension

        Returns:
            edge_attr: torch.LongTensor (E, 3) — binned edge features
        """
        if isinstance(node_features, torch.Tensor):
            node_features_np = node_features.numpy()
        else:
            node_features_np = node_features

        if isinstance(node_pos, torch.Tensor):
            node_pos_np = node_pos.numpy()
        else:
            node_pos_np = node_pos

        num_edges = edge_index.shape[1]
        srcs = edge_index[0]  # (E,)
        dsts = edge_index[1]  # (E,)

        # ── Dim 0: RGB colour distance ─────────────────────────────────────
        # Use first 3 dims of node_features (always RGB-like, in [0,1])
        rgb = node_features_np[:, :3]
        color_diffs = np.linalg.norm(rgb[srcs] - rgb[dsts], axis=1)  # (E,)
        max_color = np.sqrt(3.0)
        color_bins = np.digitize(
            color_diffs, bins=np.linspace(0, max_color, n_bins + 1)
        ) - 1
        color_bins = np.clip(color_bins, 0, n_bins - 1).astype(np.int64)

        # ── Dim 1: Spatial distance (normalised centroid) ──────────────────
        spatial_diffs = np.linalg.norm(node_pos_np[srcs] - node_pos_np[dsts], axis=1)  # (E,)
        max_spatial = np.sqrt(2.0)  # max dist in [0,1]×[0,1] space
        spatial_bins = np.digitize(
            spatial_diffs, bins=np.linspace(0, max_spatial, n_bins + 1)
        ) - 1
        spatial_bins = np.clip(spatial_bins, 0, n_bins - 1).astype(np.int64)

        # ── Dim 2: Cosine similarity → mapped to [0, 1] then binned ────────
        feat_tensor = torch.tensor(node_features_np, dtype=torch.float32)
        feat_norm = F.normalize(feat_tensor, p=2, dim=1)  # (N, D)
        cosine_sim = (feat_norm[srcs] * feat_norm[dsts]).sum(dim=1).numpy()  # (E,)
        # cosine sim ∈ [-1, 1] → shift to [0, 1]
        cosine_01 = (cosine_sim + 1.0) / 2.0
        cosine_bins = np.digitize(
            cosine_01, bins=np.linspace(0, 1.0, n_bins + 1)
        ) - 1
        cosine_bins = np.clip(cosine_bins, 0, n_bins - 1).astype(np.int64)

        # ── Stack into (E, 3) ───────────────────────────────────────────────
        edge_attr = np.stack([color_bins, spatial_bins, cosine_bins], axis=1)  # (E, 3)
        return torch.tensor(edge_attr, dtype=torch.long)

    def build_hierarchical_graph(
        self, fine_graph, n_clusters=12, n_bins=10, use_rich_edges=False
    ):
        """
        C1: Build a hierarchical (2-level) graph from a fine superpixel graph.

        Architecture:
            - 75 fine nodes  (original superpixels)
            - 12 coarse nodes (KMeans cluster centres of CNN/node features)
            - Edges:
                fine ↔ fine   : original adjacency (fine_graph.edge_index)
                fine ↔ coarse : each fine node connects bidirectionally to its cluster centre
                coarse ↔ coarse : fully connected (bidirectional)

        Args:
            fine_graph:    PyG Data with x (N_fine, D), edge_index, edge_attr, pos (N_fine, 2)
            n_clusters:    number of coarse nodes (default 12)
            n_bins:        bins for edge quantisation
            use_rich_edges: if True, use 3-dim edge_attr; else use single-dim colour binning

        Returns:
            PyG Data with all_x (N_fine+n_clusters, D), hierarchical edge_index,
            matching edge_attr, node_type (long tensor: 0=fine, 1=coarse), pos.
        """
        from sklearn.cluster import KMeans

        n_fine = fine_graph.x.size(0)  # typically 75
        features_np = fine_graph.x.detach().numpy()  # (N_fine, D)

        # ── KMeans clustering ───────────────────────────────────────────────
        n_clusters = min(n_clusters, n_fine)  # guard: can't have more clusters than nodes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_np)  # (N_fine,)

        # ── Coarse node features = mean of cluster members ──────────────────
        coarse_x_list = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            if mask.sum() > 0:
                coarse_x_list.append(fine_graph.x[mask].mean(0))
            else:
                coarse_x_list.append(fine_graph.x.mean(0))  # fallback: global mean
        coarse_x = torch.stack(coarse_x_list, dim=0)  # (n_clusters, D)

        # ── Coarse node positions = mean centroid of cluster members ─────────
        has_pos = hasattr(fine_graph, 'pos') and fine_graph.pos is not None
        if has_pos:
            coarse_pos_list = []
            for c in range(n_clusters):
                mask = cluster_labels == c
                if mask.sum() > 0:
                    coarse_pos_list.append(fine_graph.pos[mask].mean(0))
                else:
                    coarse_pos_list.append(fine_graph.pos.mean(0))
            coarse_pos = torch.stack(coarse_pos_list, dim=0)  # (n_clusters, 2)

        # ── Cross-level edges: fine ↔ coarse ────────────────────────────────
        cross_src_f2c = torch.arange(n_fine)                       # fine node indices
        cross_dst_f2c = torch.tensor(cluster_labels) + n_fine       # coarse indices (offset)
        # Bidirectional
        cross_edge_index = torch.stack([
            torch.cat([cross_src_f2c, cross_dst_f2c]),
            torch.cat([cross_dst_f2c, cross_src_f2c]),
        ], dim=0)  # (2, 2*N_fine)

        # ── Coarse-to-coarse: fully connected ────────────────────────────────
        coarse_idx = torch.arange(n_clusters) + n_fine
        if n_clusters > 1:
            coarse_pairs = torch.combinations(coarse_idx, r=2).T  # (2, C*(C-1)/2)
            coarse_edge_index = torch.cat(
                [coarse_pairs, coarse_pairs.flip(0)], dim=1
            )  # bidirectional
        else:
            coarse_edge_index = torch.zeros(2, 0, dtype=torch.long)

        # ── Merge all edges ──────────────────────────────────────────────────
        all_edge_index = torch.cat(
            [fine_graph.edge_index, cross_edge_index, coarse_edge_index], dim=1
        )  # (2, E_total)

        # ── Merge all nodes ──────────────────────────────────────────────────
        all_x = torch.cat([fine_graph.x, coarse_x], dim=0)  # (N_fine + n_clusters, D)
        all_pos = torch.cat([fine_graph.pos, coarse_pos], dim=0) if has_pos else None

        # ── Node type mask: 0=fine, 1=coarse ───────────────────────────────
        node_type = torch.cat([
            torch.zeros(n_fine, dtype=torch.long),
            torch.ones(n_clusters, dtype=torch.long),
        ])  # (N_fine + n_clusters,)

        # ── Build edge_attr for all edges ────────────────────────────────────
        if use_rich_edges and has_pos:
            # Re-compute rich features over larger node set
            rich_attr = self.compute_rich_edge_features(
                all_x, all_pos, all_edge_index.numpy(), n_bins=n_bins
            )  # (E_total, 3)
            all_edge_attr = rich_attr
        else:
            # Legacy: propagate existing fine edge_attr + create 0-bin attrs for new edges
            n_new_edges = cross_edge_index.size(1) + coarse_edge_index.size(1)
            existing_attr = fine_graph.edge_attr
            if existing_attr.dim() == 1:
                existing_attr = existing_attr.unsqueeze(1)  # (E_fine, 1)
            new_attr = torch.zeros(
                n_new_edges, existing_attr.size(-1), dtype=torch.long
            )  # (E_new, D)
            all_edge_attr = torch.cat([existing_attr, new_attr], dim=0)  # (E_total, D)

        # ── Construct new Data object ────────────────────────────────────────
        new_data = Data(
            x=all_x,
            edge_index=all_edge_index,
            edge_attr=all_edge_attr,
            y=fine_graph.y,
            node_type=node_type,
        )
        if all_pos is not None:
            new_data.pos = all_pos

        return new_data
    
    def convert(self, image, label=None, use_rich_edges=False, use_hierarchical=False,
                n_clusters=12, n_bins=10, use_full_connectivity=False):
        """
        Convert an image to a graph structure.

        Args:
            image:            PIL Image or numpy array
            label:            Optional label for the graph
            use_rich_edges:   C2 — if True, use 3-dim edge features
            use_hierarchical: C1 — if True, add KMeans coarse nodes
            n_clusters:       Number of coarse nodes for C1 (default 12)
            n_bins:           Quantisation bins for edge features (default 10)
            use_full_connectivity: If True, fully connect all superpixels.

        Returns:
            data: PyTorch Geometric Data object with:
                - x          : node features (N, D) — D=128 (CNN) or D=5 (legacy)
                - pos        : normalised centroid coordinates (N, 2) — [cx, cy]
                - edge_index : edge connectivity (2, E)
                - edge_attr  : (E,) 1-dim int OR (E, 3) long — depending on use_rich_edges
                - y          : graph label (optional)
                - node_type  : (N,) long, 0=fine / 1=coarse  [only if use_hierarchical]
        """
        target_size = (224, 224)
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Extract superpixels
        segments, n_segments = self.extract_superpixels(image)

        # Compute raw node features (always needed for edge features & pos)
        raw_node_features = self.compute_node_features(image, segments, n_segments)

        if self.use_cnn_features:
            # CNN features: [N, cnn_feature_dim]
            x = self.cnn_extractor.extract_features(image, segments, n_segments)
        else:
            # Legacy: [R, G, B, cx, cy] = 5-dim
            x = torch.tensor(raw_node_features, dtype=torch.float)

        # Build edge index
        if use_full_connectivity:
            # Fully connected graph: all pairs except self-loops
            src_nodes = np.repeat(np.arange(n_segments), n_segments - 1)
            dst_nodes = np.tile(np.arange(n_segments), n_segments)
            dst_nodes = dst_nodes[dst_nodes != np.repeat(np.arange(n_segments), n_segments)]
            edge_index_np = np.stack([src_nodes, dst_nodes], axis=0).astype(np.int64)
        else:
            # Default: use local superpixel adjacency
            adjacency = self.build_adjacency(segments)
            edge_list = []
            for src in adjacency:
                for dst in adjacency[src]:
                    edge_list.append([src, dst])
            edge_index_np = np.array(edge_list, dtype=np.int64).T  # (2, E)

        # ── Positional features ─────────────────────────────────────────────
        pos = torch.tensor(raw_node_features[:, 3:], dtype=torch.float)  # (N, 2)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        # ── Edge features (C2: rich 3-dim, or legacy 1-dim) ─────────────────
        if use_rich_edges:
            edge_attr = self.compute_rich_edge_features(
                x, pos, edge_index_np, n_bins=n_bins
            )  # (E, 3) long
        else:
            edge_feats_np = self.compute_edge_features(
                raw_node_features[:, :3], edge_index_np, n_bins=n_bins
            )  # (E,) int
            edge_attr = torch.tensor(edge_feats_np, dtype=torch.long)

        # ── Create fine-level PyG Data object ────────────────────────────────
        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)

        # ── C1: Hierarchical graph ────────────────────────────────────────────
        if use_hierarchical:
            data = self.build_hierarchical_graph(
                data,
                n_clusters=n_clusters,
                n_bins=n_bins,
                use_rich_edges=use_rich_edges,
            )
            # Re-attach label in case build_hierarchical_graph lost it
            if label is not None and (not hasattr(data, 'y') or data.y is None):
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
