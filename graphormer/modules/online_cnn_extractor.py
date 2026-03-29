import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Scale ngẫu ngược gradient
        return grad_output * ctx.scale, None


class OnlineCNNExtractor(nn.Module):
    """
    Backbone ResNet18 trích xuất CNN field.
    Dùng F.grid_sample để lấy node embedding tương ứng với centroid (pos).
    
    Args:
        embed_dim: 차원 output sau projection mask.
        freeze_bn: Giữ Batch Norm ở chế độ eval để features không bị jump.
    """
    def __init__(self, embed_dim=128, freeze_bn=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Drop FC & AvgPool (2 layers cuối)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projector = nn.Linear(512, embed_dim)
        
        self.freeze_bn = freeze_bn
        
        # Setup mode ban đầu cho Backbone
        if self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    # Freeze BN parameter updates
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    
    def train(self, mode=True):
        """Override train để giữ BN ở eval mode nếu freeze_bn=True"""
        super().train(mode)
        if self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def extract_feature_map(self, raw_image):
        """
        raw_image: [B, 3, 224, 224] tensor, range [0, 1] usually,
                   BUT resnet is pretrained with standard imagenet normalization.
                   If not grouped inside dataloader, consider applying transform here.
        """
        # (Assuming raw_image is already normalized with ImageNet mean/std internally or via dataloader)
        return self.backbone(raw_image)  # [B, 512, 7, 7]

    def forward(self, raw_image, pos, mask=None, cnn_lr_scale=0.1):
        """
        Args:
            raw_image: Tensor [B, 3, 224, 224]
            pos: Tensor [B, max_nodes, 2] chứa (cx, cy) chuẩn hóa [0, 1].
                 Padding values thường có thể = 0 (và node sẽ bị mask trong attention).
            mask: Tensor [B, 224, 224] chứa ID của superpixel (0 -> N-1). Dùng cho Exact Mask Pooling.
            cnn_lr_scale: Multiplier cho learning rate scale
            
        Returns:
            features: Tensor [B, max_nodes, embed_dim]
        """
        # 1. Forward backbone
        feat_map = self.extract_feature_map(raw_image) # [B, 512, H_feat, W_feat]
        
        # Áp dụng Gradient Multiplier để cnn_lr_scale (giảm LR cho parameters của backbone)
        feat_map = GradMultiply.apply(feat_map, cnn_lr_scale)
        
        B, C, H_feat, W_feat = feat_map.shape
        max_nodes = pos.size(1)

        # 2. Extract features
        if mask is not None:
            # === Exact Mask Pooling ===
            # Tính fallback: dự phòng trường hợp node diện tích quá nhỏ bị biến mất khi downsample
            grid = pos * 2.0 - 1.0  # [B, max_nodes, 2]
            grid = grid.unsqueeze(1) # [B, 1, N, 2]
            sampled = F.grid_sample(
                feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True
            ) # [B, C, 1, N]
            fallback_feat = sampled.squeeze(2).transpose(1, 2) # [B, N, C]

            # Resize mask xuống độ phân giải của Feature Map
            mask_small = F.interpolate(
                mask.unsqueeze(1).float(), 
                size=(H_feat, W_feat), 
                mode='nearest'
            ).squeeze(1).long() # [B, H_feat, W_feat]

            # Tạo one-hot vector cho các Node
            node_idx = torch.arange(max_nodes, device=mask.device).view(1, -1, 1, 1) # [1, N, 1, 1]
            is_node = (mask_small.unsqueeze(1) == node_idx) # [B, N, H_feat, W_feat]

            # Lọc và tính tổng Feature Map cho từng vùng mask
            masked_feat = feat_map.unsqueeze(1) * is_node.unsqueeze(2).float() # [B, N, C, H_feat, W_feat]
            sum_feat = masked_feat.sum(dim=(3, 4)) # [B, N, C]

            # Tính diện tích (trọng số)
            count = is_node.sum(dim=(2, 3)).unsqueeze(2).float() # [B, N, 1]

            # Chia trung bình (Average Pooling) hoặc Fallback
            out_features = torch.where(
                count > 0,
                sum_feat / torch.clamp(count, min=1.0),
                fallback_feat
            )
        else:
            # === Legacy Grid Sample ===
            grid = pos * 2.0 - 1.0  # [B, max_nodes, 2]
            grid = grid.unsqueeze(1) # [B, 1, N, 2]
            
            sampled = F.grid_sample(
                feat_map,               # [B, C, H_feat, W_feat]
                grid,                   # [B, 1, N, 2]
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ) # [B, C, 1, N]
            
            out_features = sampled.squeeze(2).transpose(1, 2) # [B, N, C]
        
        # 3. Project to embedding_dim
        out = self.projector(out_features) # [B, N, embed_dim]
        
        return out