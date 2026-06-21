import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("weight_x", kernel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("weight_y", kernel_y.repeat(channels, 1, 1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channels = channels

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.channels)
        edge_prior = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        edge_gate = self.gate(torch.cat([x, edge_prior], dim=1))
        return x * (1.0 + edge_gate), edge_gate


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.edge_gate = SobelEdgeGate(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        feat = self.conv(x)
        feat, gate = self.edge_gate(feat)
        return feat, self.pool(feat), gate


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip, x):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MultiScaleContext(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 4, 8)):
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class BoundaryConstrainedGraphConv(nn.Module):
    def __init__(self, channels, node_dim=64, graph_channels=256, spatial_weight=0.4, boundary_weight=1.0):
        super().__init__()
        self.node_proj = nn.Conv2d(channels, node_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(channels, graph_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(graph_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_weight = spatial_weight
        self.boundary_weight = boundary_weight

    def spatial_distance(self, h, w, device, dtype):
        y_coords = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        try:
            y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        except TypeError:
            y, x = torch.meshgrid(y_coords, x_coords)
        coords = torch.stack([y.reshape(-1), x.reshape(-1)], dim=1)
        dist = torch.cdist(coords, coords, p=2)
        return dist / (dist.max() + 1e-6)

    def forward(self, x, boundary_logits):
        b, _, h, w = x.shape
        node = self.node_proj(x).flatten(2).transpose(1, 2)
        node = F.normalize(node, dim=-1)
        sim = torch.bmm(node, node.transpose(1, 2)) / math.sqrt(node.size(-1))
        dist = self.spatial_distance(h, w, x.device, x.dtype).unsqueeze(0)
        sim = sim - self.spatial_weight * dist
        boundary = torch.sigmoid(F.interpolate(boundary_logits, size=(h, w), mode="bilinear", align_corners=True))
        boundary = boundary.flatten(2).transpose(1, 2)
        barrier = 0.5 * (boundary + boundary.transpose(1, 2))
        sim = sim - self.boundary_weight * barrier
        adj = F.softmax(sim, dim=-1)
        value = self.value_proj(x).flatten(2).transpose(1, 2)
        graph = torch.bmm(adj, value).transpose(1, 2).reshape(b, -1, h, w)
        graph = self.out_proj(graph)
        return self.refine(x + graph)


class BoundaryHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class BCPNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.down1 = EncoderBlock(in_channels, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        self.down4 = EncoderBlock(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = BoundaryHead(64)
        self.context = MultiScaleContext(1024)
        self.graph = BoundaryConstrainedGraphConv(1024)
        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1, x, _ = self.down1(x)
        boundary_logits = self.boundary_head(x1)
        x2, x, _ = self.down2(x)
        x3, x, _ = self.down3(x)
        x4, x, _ = self.down4(x)
        x = self.middle_conv(x)
        x = self.context(x)
        x = self.graph(x, boundary_logits)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        return {"seg": self.final_conv(x), "edge": boundary_logits}

    def summary(self):
        return sum(np.prod(param.size()) for param in self.parameters() if param.requires_grad)
