"""
Backbones available in this codebase, all behind one `build_model(...)` call
site so any training method / dataset can use any of them interchangeably:

  - "smallcnn"   : compact 4-block CNN (~0.3-1.1M params), the default used
                   for the CPU-only compact-CNN protocol.
  - "wrn28_10"   : WideResNet-28-10 (~36.5M params), the paper's main-protocol
                   architecture (Zagoruyko & Komodakis, 2016).
  - "resnet18"   : a real residual network, standard CIFAR-style ResNet-18
                   (He et al., 2016) with BasicBlocks, ~11.2M params.
  - "vgg11"      : a real VGG-11-style stack (Simonyan & Zisserman, 2015) of
                   conv+BN+ReLU blocks with max-pooling, ~9.8M params.
  - "mobilenet"  : a real MobileNet-style network (Howard et al., 2017) built
                   from depthwise-separable convolutions, ~1.0M params -- the
                   lightest of the five, included to span the parameter-count
                   range from compact CNN to WideResNet-28-10.

Every backbone also exposes `forward_features(x)`, returning the pooled,
pre-classifier embedding (the same tensor `forward` feeds to its final linear
layer), used for the latent-space (t-SNE) visualizations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SmallCNN(nn.Module):
    """
    A compact, dataset-agnostic CNN:
      input (C,H,W) -> 4 conv blocks with downsampling -> global average pool
      -> linear classifier.
    Works for any input resolution/channel count via AdaptiveAvgPool2d.
    """
    def __init__(self, in_channels=3, num_classes=10, width=32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, width, stride=1),
            ConvBlock(width, width, stride=1),
            nn.MaxPool2d(2),
            ConvBlock(width, width * 2, stride=1),
            ConvBlock(width * 2, width * 2, stride=1),
            nn.MaxPool2d(2),
            ConvBlock(width * 2, width * 4, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        z = self.forward_features(x)
        return self.classifier(z)


class WideBasicBlock(nn.Module):
    """Pre-activation basic block used by WideResNet (Zagoruyko & Komodakis, 2016)."""
    def __init__(self, cin, cout, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_io = (cin == cout and stride == 1)
        if not self.equal_io:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = x if self.equal_io else self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    """
    Standard WideResNet-d-k (Zagoruyko & Komodakis, 2016), e.g. WRN-28-10 with
    depth=28, widen_factor=10. This is the backbone architecture specified for
    the paper's main CIFAR-10/CIFAR-100 protocol (Section IV-A); it is provided
    here, alongside SmallCNN, so the exact same architecture can also be run
    (at a necessarily reduced scale) inside this CPU-only sandbox as a code-
    verified feasibility check (Section IV-B4), not as a replacement for the
    paper's original full-scale, GPU-trained numbers in Tables II-IV.
    """
    def __init__(self, in_channels=3, num_classes=10, depth=28, widen_factor=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, stride=1, padding=1, bias=False)
        self.block1 = self._make_group(widths[0], widths[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_group(widths[1], widths[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_group(widths[2], widths[3], n, stride=2, drop_rate=drop_rate)
        self.bn_final = nn.BatchNorm2d(widths[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[3], num_classes)
        self.n_params = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _make_group(cin, cout, n_blocks, stride, drop_rate):
        layers = [WideBasicBlock(cin, cout, stride, drop_rate)]
        for _ in range(n_blocks - 1):
            layers.append(WideBasicBlock(cout, cout, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        z = self.conv1(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = F.relu(self.bn_final(z), inplace=True)
        return self.pool(z).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ----------------------------------------------------------------------------
# ResNet-18 (He et al., 2016), standard CIFAR-style variant: 3x3 stride-1 stem
# (no 7x7/maxpool stem, since inputs here are 28-64px, not ImageNet-scale),
# 4 stages of BasicBlock at channel widths [64,128,256,512], strides
# [1,2,2,2]. Real residual connections, real backward pass -- not a
# lightweight stand-in.
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return F.relu(out + shortcut, inplace=True)


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(64, 64, 2, stride=1)
        self.layer2 = self._make_stage(64, 128, 2, stride=2)
        self.layer3 = self._make_stage(128, 256, 2, stride=2)
        self.layer4 = self._make_stage(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(cin, cout, n_blocks, stride):
        layers = [BasicBlock(cin, cout, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(cout, cout, 1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        return self.pool(z).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ----------------------------------------------------------------------------
# VGG-11-style (Simonyan & Zisserman, 2015): stacked 3x3 conv+BN+ReLU blocks
# with max-pooling downsampling, ending in global average pooling (instead of
# the original's 3 large FC layers, to stay parameter-reasonable and
# resolution-agnostic across this codebase's 28-64px datasets).
# ----------------------------------------------------------------------------
VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        layers = []
        cin = in_channels
        for v in VGG11_CFG:
            if v == "M":
                # ceil_mode=True: VGG11_CFG has 5 stride-2 poolings, which
                # floor-divides a 28x28 input (Fashion-MNIST) down to 0x0 on
                # the 5th pooling (28->14->7->3->1->0, a crash) under the
                # default floor rounding. ceil_mode rounds up instead
                # (28->14->7->4->2->1), so every dataset's native resolution
                # in this codebase (28 through 96) survives all 5 poolings;
                # it is a no-op for sizes that already divide evenly (e.g.
                # CIFAR's 32x32), so CIFAR-scale behavior is unchanged.
                layers.append(nn.MaxPool2d(2, ceil_mode=True))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                cin = v
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ----------------------------------------------------------------------------
# MobileNet-style (Howard et al., 2017): a standard conv stem followed by
# depthwise-separable (depthwise 3x3 + pointwise 1x1) blocks -- real
# depthwise convolutions, not a simulated lightweight net. The smallest of
# this codebase's five backbones by parameter count.
# ----------------------------------------------------------------------------
class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)), inplace=True)
        x = F.relu(self.bn2(self.pointwise(x)), inplace=True)
        return x


class MobileNetSmall(nn.Module):
    MOBILE_CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        layers = []
        cin = 32
        for cout, stride in self.MOBILE_CFG:
            layers.append(DepthwiseSeparable(cin, cout, stride))
            cin = cout
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cin, num_classes)

    def forward_features(self, x):
        z = self.stem(x)
        z = self.blocks(z)
        return self.pool(z).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


BACKBONE_PARAM_COUNTS_APPROX = {
    "mobilenet": "~1.0M", "smallcnn": "~0.3-1.1M", "vgg11": "~9.8M",
    "resnet18": "~11.2M", "wrn28_10": "~36.5M",
}


BACKBONES = ["smallcnn", "wrn28_10", "resnet18", "vgg11", "mobilenet"]


def build_model(dataset_info, width=32, backbone="smallcnn"):
    """
    backbone: one of "smallcnn" (default), "wrn28_10", "resnet18", "vgg11",
    "mobilenet" -- see the module docstring for what each is. All five share
    this one call site and all expose forward_features(x) for latent-space
    extraction.
    """
    cin, ncls = dataset_info["channels"], dataset_info["num_classes"]
    if backbone == "smallcnn":
        return SmallCNN(in_channels=cin, num_classes=ncls, width=width)
    elif backbone == "wrn28_10":
        return WideResNet(in_channels=cin, num_classes=ncls, depth=28, widen_factor=10)
    elif backbone == "resnet18":
        return ResNet18(in_channels=cin, num_classes=ncls)
    elif backbone == "vgg11":
        return VGG11(in_channels=cin, num_classes=ncls)
    elif backbone == "mobilenet":
        return MobileNetSmall(in_channels=cin, num_classes=ncls)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


class NormalizedModel(nn.Module):
    """
    Wraps a backbone so that attacks / training can operate directly on raw
    [0,1]-range images (the standard convention for epsilon budgets like
    8/255), while the backbone still receives per-dataset normalized input.
    """
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)

    def forward_features(self, x):
        return self.backbone.forward_features((x - self.mean) / self.std)
