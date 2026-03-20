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
        return grad_output * ctx.scale, None


def _grid_sample_from_map(feat_map, pos):
    """
    Lấy feature tại các vị trí centroids bằng bilinear interpolation.

    Args:
        feat_map: [B, C, H, W]
        pos: [B, N, 2] — (cx, cy) chuẩn hóa [0, 1]

    Returns:
        [B, N, C]
    """
    # Chuyển pos từ [0,1] → [-1,1] theo chuẩn của F.grid_sample
    grid = pos * 2.0 - 1.0          # [B, N, 2]
    grid = grid.unsqueeze(1)         # [B, 1, N, 2]

    sampled = F.grid_sample(
        feat_map,                    # [B, C, H, W]
        grid,                        # [B, 1, N, 2]
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )                                # [B, C, 1, N]

    return sampled.squeeze(2).transpose(1, 2)  # [B, N, C]


class OnlineCNNExtractor(nn.Module):
    """
    Multi-scale CNN Extractor dùng ResNet18 backbone với FPN-style feature sampling.

    Lấy đặc trưng từ 3 tầng của ResNet18 để nắm bắt cả thông tin texture/cạnh (tầng sớm)
    lẫn thông tin ngữ nghĩa (tầng sâu), giúp model nhận ra các pattern nhỏ như Hispa.

    Spatial resolution tại mỗi scale (ảnh đầu vào 224×224):
        layer2 → [B, 128, 28, 28]  stride 8x  — texture & cạnh mảnh (tốt cho Hispa!)
        layer3 → [B, 256, 14, 14]  stride 16x — đặc trưng trung cấp
        layer4 → [B, 512, 7,  7 ]  stride 32x — ngữ nghĩa toàn cục

    Args:
        embed_dim: Số chiều output sau projection.
        freeze_bn:  Nếu True, giữ BatchNorm ở eval mode để tránh feature shift.
        scales:     Tuple các tầng ResNet cần lấy (2, 3, 4). Giảm scales = nhẹ hơn nhưng
                    mất thông tin. Mặc định dùng cả 3 cho kết quả tốt nhất.
    """

    # Kênh đầu ra tương ứng với mỗi layer của ResNet18
    _LAYER_CHANNELS = {2: 128, 3: 256, 4: 512}

    def __init__(self, embed_dim=128, freeze_bn=True, scales=(2, 3, 4)):
        super().__init__()
        assert all(s in (2, 3, 4) for s in scales), "scales phải là tập con của {2, 3, 4}"
        self.scales = sorted(scales)

        resnet = models.resnet18(pretrained=True)

        # Tách ResNet18 thành các block riêng để truy cập feature map trung gian
        self.stem    = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1   # [B, 64,  56, 56]  stride 4x
        self.layer2  = resnet.layer2   # [B, 128, 28, 28]  stride 8x
        self.layer3  = resnet.layer3   # [B, 256, 14, 14]  stride 16x
        self.layer4  = resnet.layer4   # [B, 512, 7,  7 ]  stride 32x

        total_channels = sum(self._LAYER_CHANNELS[s] for s in self.scales)
        self.projector = nn.Linear(total_channels, embed_dim)

        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            self._freeze_bn_params()

    # ------------------------------------------------------------------
    # BN freeze helpers
    # ------------------------------------------------------------------
    def _freeze_bn_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad   = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _extract_feature_maps(self, raw_image):
        """
        Đưa ảnh qua ResNet18 và trả về dict {scale: feature_map}.

        raw_image: [B, 3, 224, 224] — đã chuẩn hóa ImageNet
        """
        x = self.stem(raw_image)    # [B, 64, 56, 56]
        x = self.layer1(x)          # [B, 64, 56, 56]
        l2 = self.layer2(x)         # [B, 128, 28, 28]
        l3 = self.layer3(l2)        # [B, 256, 14, 14]
        l4 = self.layer4(l3)        # [B, 512, 7,  7 ]
        return {2: l2, 3: l3, 4: l4}

    def forward(self, raw_image, pos, cnn_lr_scale=0.1):
        """
        Args:
            raw_image:    Tensor [B, 3, 224, 224]
            pos:          Tensor [B, max_nodes, 2] — centroids chuẩn hóa [0, 1]
            cnn_lr_scale: Nhân gradient ngược để giảm learning rate cho CNN backbone.

        Returns:
            features: Tensor [B, max_nodes, embed_dim]
        """
        # 1. Trích xuất feature maps từ các tầng đã chọn
        feat_maps = self._extract_feature_maps(raw_image)

        # 2. Áp dụng GradMultiply để scale learning rate cho backbone
        feat_maps = {
            s: GradMultiply.apply(fm, cnn_lr_scale)
            for s, fm in feat_maps.items()
        }

        # 3. Grid sample tại centroids cho mỗi scale, rồi concatenate
        sampled_list = [
            _grid_sample_from_map(feat_maps[s], pos)   # [B, N, C_s]
            for s in self.scales
        ]
        multi_scale = torch.cat(sampled_list, dim=-1)  # [B, N, sum(C_s)]

        # 4. Project về embed_dim
        out = self.projector(multi_scale)              # [B, N, embed_dim]
        return out
