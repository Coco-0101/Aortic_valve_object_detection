import torch
import torch.nn as nn

class DetectAF(nn.Module):
    """Anchor-Free detection head compatible with YOLOv11"""
    def __init__(self, nc=1, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.ch = ch
        self.reg_max = 16
        self.no = nc + 4 * self.reg_max
        self.stride = torch.zeros(self.nl)
        self.cv2 = nn.ModuleList(nn.Conv2d(x, self.no, 1) for x in ch)

    def forward(self, x):
        if not isinstance(x, list): x = [x]
        out = [self.cv2[i](x[i]) for i in range(self.nl)]
        return out
