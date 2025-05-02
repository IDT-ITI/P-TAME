from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from ptame.utils.masking_utils import minmax_4d


class TtameAttention(nn.Module):
    """TtameAttention is a class that represents the attention mechanism used
    in TTAME."""

    def __init__(self, ft_size: List[torch.Size], num_classes=1000):
        super().__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.resolution = feat_height
        self.interpolate = lambda inp: F.interpolate(
            inp,
            size=(feat_height, feat_height),
            mode="bilinear",
            align_corners=False,
        )
        in_channels_list = [o[1] for o in ft_size]
        self.channels = in_channels_list
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
                for in_channels in in_channels_list
            ]
        )
        self.bn_channels = in_channels_list
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(channels) for channels in self.bn_channels]
        )
        self.relu = nn.ReLU()
        # for each extra layer we need 1000 more channels to input to the fuse
        # convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.num_classes = num_classes

    def forward(self, features, **kwargs):
        """Forward pass of the attention mechanism."""
        # Fusion Strategy
        feature_maps = features
        if kwargs.get("opticam", False):
            return features[0]
        # Now all feature map sets are of the same HxW
        # conv
        class_maps = [
            op(feature) for op, feature in zip(self.convs, feature_maps)
        ]
        # batch norm
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.relu(class_map) for class_map in class_maps]
        # upscale
        class_maps = [self.interpolate(feature) for feature in class_maps]
        # concat
        class_maps = torch.cat(class_maps, 1)
        # fuse into num_classes channels
        c = self.fuser(class_maps)  # batch_size x1xWxH
        if not self.training:
            return minmax_4d(c)
        else:
            return torch.sigmoid(c)

    @torch.no_grad()
    def get_contributions(self):
        """Calculate the contributions of the attention mechanism."""
        for name, param in self.fuser.named_parameters():
            if "weight" in name:
                weights = param.squeeze()
        channels = self.channels.copy()
        contribs = torch.stack(
            [
                a.sum(dim=1)
                for a in weights.softmax(dim=1).split(channels, dim=1)
            ],
            dim=1,
        )
        return [f"{i}" for i in range(len(self.channels))], contribs
