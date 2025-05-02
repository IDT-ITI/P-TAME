import math

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from ptame.utils.masking_utils import minmax_4d


class PTAMEAttention(nn.Module):
    """PTAMEAttention is a class that represents the attention mechanism based
    on a feature extractor and P-TAME."""

    def __init__(
        self,
        model: dict[str, list | nn.Module] = {
            "model": resnet18(weights="DEFAULT"),
            "layers": ["layer1", "layer2", "layer3", "layer4"],
        },
        input_dim: list[int] | None = [2, 3, 224, 224],
        scale_up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear"),
        activation: nn.Module = nn.ReLU(),
        fuser_bias: bool = True,
        num_classes: int = 1000,
        unfreeze: int = 0,
        cascading: bool = True,
        only_layer_train: bool = False,
        normalizers: list = [torch.sigmoid, minmax_4d],
    ):
        super().__init__()
        self.layer_names = model["layers"]
        fx_dict = self._make_fx(**model, input_dim=input_dim)
        self.attention = fx_dict["attention"]
        self.attention.requires_grad_(False).eval()
        self.ft_size = fx_dict["ft_size"]
        self.output_name = fx_dict["output_name"]
        self.scale_up = scale_up
        self.num_classes = num_classes

        self.resolution = self.ft_size[0][-1]
        channels = [ft[0] for ft in self.ft_size]
        self.channels = channels
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
                for c in channels
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(c) for c in channels])
        self.act = activation
        self.fuser = nn.Conv2d(
            in_channels=sum(channels),
            out_channels=self.num_classes,
            kernel_size=1,
            padding=0,
            bias=fuser_bias,
        )

        self.normalizers = normalizers
        self.trainable_layers = [self.convs, self.bns, self.fuser]

    @staticmethod
    def list_processor(
        ops_list: nn.ModuleList, feature_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Apply a list of operations to a list of feature maps.

        Args:
            ops_list (nn.ModuleList): List of operations to apply.
            feature_list (list[torch.Tensor]): List of feature maps.

        Returns:
            list[torch.Tensor]: List of feature maps with the operations applied.
        """
        return [op(feature) for op, feature in zip(ops_list, feature_list)]

    @staticmethod
    def skip_connection(
        a: list[torch.Tensor], b: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Add the feature maps of two lists.

        Args:
            a (list[torch.Tensor]): First list of feature maps.
            b (list[torch.Tensor]): Second list of feature maps.

        Returns:
            list[torch.Tensor]: List of feature maps with the two input lists added.
        """
        return [a + b for a, b in zip(a, b)]

    def list_activation(
        self, feature_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Apply the activation function to the feature maps.

        Args:
            feature_list (list[torch.Tensor]): List of feature maps.

        Returns:
            list[torch.Tensor]: List of feature maps with the activation function applied.
        """
        return [self.act(feature) for feature in feature_list]

    def upscale(self, feature_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Upscale the feature maps to the original resolution.

        Args:
            feature_list (list[torch.Tensor]): List of feature maps.

        Returns:
            list[torch.Tensor]: List of upscaled feature maps.
        """

        def up(map, times):
            return up(self.scale_up(map), times - 1) if times > 0 else map

        return [
            up(
                feature,
                int(math.log(self.resolution / feature.shape[-1], 2)),
            )
            for feature in feature_list
        ]

    def forward(
        self,
        images: torch.Tensor,
        with_outputs=False,
        return_cms=False,
        **kwargs,
    ) -> torch.Tensor:
        """Overwrite of attention forward method to include explanation
        head."""
        y, ft_maps = self.attention(images)

        class_maps = self.list_processor(self.convs, ft_maps)
        class_maps = self.list_processor(self.bns, class_maps)
        class_maps = self.skip_connection(class_maps, ft_maps)
        class_maps = self.list_activation(class_maps)
        class_maps = self.upscale(class_maps)
        class_maps = torch.cat(class_maps, 1)
        if return_cms:
            return class_maps
        c = self.fuser(class_maps)
        if with_outputs:
            return y, c
        if self.training or self.normalizers[1] is None:
            return self.normalizers[0](c)
        else:
            return self.normalizers[1](c)

    def _make_fx(self, model, layers, input_dim) -> dict:
        """Create the feature extractor from the model."""
        train_names, eval_names = get_graph_node_names(model)
        if not layers:
            print(train_names)
            quit()
        output = (train_names[-1], eval_names[-1])
        if output[0] != output[1]:
            print(
                "WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE"
            )
        output_name = output[0]
        attention = create_feature_extractor(
            model, return_nodes=layers + [output_name]
        )
        if input_dim is not None:
            inp = torch.randn(input_dim)
        else:
            inp = torch.randn(2, 3, 224, 224)
        attention.eval()
        outputs = attention(inp)
        outputs.pop(output_name)
        features = outputs.values()
        feature_size = [o.shape[1:] for o in features]

        def hook(
            module, inputs, outputs
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            y: torch.Tensor = outputs.pop(output_name)
            features = list(outputs.values())
            return y, features

        attention.register_forward_hook(hook)
        return {
            "attention": attention,
            "ft_size": feature_size,
            "output_name": output_name,
        }

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
        return self.layer_names, contribs


def minmax_compose(fn):
    def wrapper(*args, **kwargs):
        return minmax_4d(fn(*args, **kwargs))

    return wrapper
