"""
CNN Feature Extractor for Superpixel Nodes.

Replaces raw 5-dim features [R, G, B, cx, cy] with 128-dim CNN features
extracted from a pretrained ResNet18 backbone via bilinear sampling.

Pipeline:
    Image 224×224 → ResNet18 backbone → feature map [512, 7, 7]
    → Bilinear sample at each superpixel centroid → [512]
    → Linear projection → [feature_dim] (default 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image


class CNNFeatureExtractor:
    """
    Extract CNN features for each superpixel region via bilinear sampling
    from a ResNet18 feature map.

    Args:
        feature_dim: Output dimension per node (default: 128)
        device: 'cpu' or 'cuda'
    """

    def __init__(self, feature_dim=128, device='cpu'):
        self.device = device
        self.feature_dim = feature_dim

        # 1. Load ResNet18 pretrained, remove classification head
        resnet = models.resnet18(pretrained=True)
        # Keep everything up to layer4 (before avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval()
        self.backbone.to(device)

        # 2. Projection layer: 512 → feature_dim
        self.projector = nn.Linear(512, feature_dim)
        self.projector.eval()
        self.projector.to(device)

        # 3. ImageNet normalization (required for pretrained ResNet)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] → [0,1], shape [3,224,224]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _get_feature_map(self, pil_image):
        """
        Forward the image through the backbone to get a feature map.

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            feature_map: Tensor [512, H_feat, W_feat] (typically [512, 7, 7])
        """
        input_tensor = self.preprocess(pil_image)  # [3, 224, 224]
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        with torch.no_grad():
            feat_map = self.backbone(input_batch)  # [1, 512, 7, 7]

        return feat_map.squeeze(0)  # [512, 7, 7]

    def extract_features(self, pil_image, segments, n_segments):
        """
        Extract CNN features for each superpixel via bilinear sampling.

        Algorithm:
            1. Forward image → feature map [512, 7, 7]
            2. For each superpixel → compute centroid (cx, cy) normalized to [0, 1]
            3. Map centroid → position on feature map
            4. Bilinear interpolate → 512-dim vector
            5. Project → feature_dim vector

        Args:
            pil_image: PIL Image (RGB, will be resized to 224×224)
            segments: numpy array [H, W] with segment labels
            n_segments: number of segments

        Returns:
            node_features: Tensor [n_segments, feature_dim]
        """
        from skimage.measure import regionprops

        # Step 1: Get feature map
        feat_map = self._get_feature_map(pil_image)  # [512, 7, 7]

        # Step 2: Compute centroids for each superpixel
        h_img, w_img = segments.shape
        props = regionprops(segments + 1)  # regionprops requires labels >= 1
        prop_dict = {prop.label - 1: prop for prop in props}

        grid_points = []
        for seg_id in range(n_segments):
            if seg_id in prop_dict:
                cy_px, cx_px = prop_dict[seg_id].centroid  # (row, col)
                cx_norm = cx_px / max(w_img - 1, 1)  # normalize [0, 1]
                cy_norm = cy_px / max(h_img - 1, 1)
            else:
                cx_norm, cy_norm = 0.5, 0.5  # fallback center

            # F.grid_sample expects coords in [-1, 1]
            gx = cx_norm * 2.0 - 1.0
            gy = cy_norm * 2.0 - 1.0
            grid_points.append([gx, gy])

        # Step 3 & 4: Bilinear sample from feature map
        # grid shape: [1, 1, N, 2]
        grid = torch.tensor(grid_points, dtype=torch.float32, device=self.device)
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

        feat_map_4d = feat_map.unsqueeze(0)  # [1, 512, 7, 7]

        # Bilinear interpolate at each centroid
        sampled = F.grid_sample(
            feat_map_4d,  # [1, C, H, W]
            grid,  # [1, 1, N, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        # sampled shape: [1, 512, 1, N] → [N, 512]
        sampled = sampled.squeeze(0).squeeze(1).T  # [N, 512]

        # Step 5: Project 512 → feature_dim
        with torch.no_grad():
            node_features = self.projector(sampled)  # [N, feature_dim]

        return node_features.cpu()  # [n_segments, feature_dim]
