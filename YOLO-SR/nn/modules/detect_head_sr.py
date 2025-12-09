import torch
import torch.nn as nn

class DetectSR(nn.Module):
    """
    Altitude-aware detection head for YOLOv8-SR.
    Outputs [B, na * no, H, W] per feature map (for use with v8DetectionLoss).
    """
    def __init__(self, nc=10, ch=(256, 512, 1024), reg_max=16, na=3):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.na = na
        self.no = nc + 4 * reg_max  # = 10 + 64 = 74
        self.stride = [8, 16, 32]
        self.ch = ch

        self.heads = nn.ModuleList([
            nn.Conv2d(c, self.na * self.no, kernel_size=1) for c in ch
        ])

    def forward(self, x, altitude=None):
        assert isinstance(x, list) and len(x) == len(self.ch), \
            f"[DetectSR] Expected {len(self.ch)} feature maps, got {len(x)}"

        outputs = []
        for i, feat in enumerate(x):
            if altitude is not None:
                bias = altitude[:, :feat.shape[1], ...]
                feat = feat + bias

            out = self.heads[i](feat)  # → (B, na * no, H, W)
            outputs.append(out)

        return outputs

