from typing import List, Tuple, Type

import torch
from torch import nn
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from ptame.models.components.ptame_pipeline import PtamePipeline
from ptame.utils.map_printer import XAIModel


class TTAME(PtamePipeline, XAIModel):
    """Transformer-compatible Trainable Attention Mechanism for
    Explainability."""

    def __init__(
        self,
        backbone: nn.Module,
        attention: nn.Module,
        masking_procedure,
    ):
        super().__init__(backbone, attention, masking_procedure)

    def _forward_backbone(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of the backbone (feature extractor)."""
        y, features = self.backbone(x)
        return y, features

    def _forward_second(
        self, x: torch.Tensor, features: List[torch.Tensor], targets
    ):
        """Second forward pass, for the training stage only."""
        # Get the saliency maps from the attention mechanism
        saliency_maps = self.attention(features)
        # Apply the masks to the input tensor and get the output
        x_masked = self.masking_procedure(
            x, self._select_matching_maps(saliency_maps, targets)
        )
        y_masked, _ = self.backbone(x_masked)
        # Select the saliency maps which will be returned based on the targets and the stage
        saliency_maps = self._select_returned_maps(saliency_maps, targets)
        return y_masked, saliency_maps

    def forward(self, x: torch.Tensor):
        """Forward pass of the entire model.

        Depending on the stage, different outputs are returned.
        """
        y, features = self._forward_backbone(x.clone().detach())
        targets = y.argmax(dim=1)
        y_masked, maps = self._forward_second(x, features, targets)
        return {
            "logits": y,
            "logits_masked": y_masked,
            "targets": targets,
            "masks": maps,
        }

    def get_predictions(self, x):
        """Get the predictions of the model.

        Useful for the measures.
        """
        return self._forward_backbone(x)[0]

    def produce_map(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        image = image.unsqueeze(0)
        out, features = self._forward_backbone(image)
        maps = self.attention(features)
        prediction = out[0].argmax().item()
        map = maps[0, prediction]
        return map, prediction

    def produce_cdmaps(
        self, image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int]:
        image = image.unsqueeze(0)
        out, features = self._forward_backbone(image)
        maps = self.attention(features)
        predictions = out[0]
        return maps, predictions


class TTAMEBuilder:
    """Builder for the TTAME model."""

    def __init__(
        self,
        backbone: nn.Module,
        attention: Type[nn.Module],
        masking_procedure,
        layers: List[str],
        input_dim: List[int] = None,
        num_classes: int = 1000,
        **kwargs,
    ):
        self.backbone = backbone
        self.attention = attention
        self.masking_procedure = masking_procedure
        self.layers = layers
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.kwargs = kwargs

    @torch.no_grad()
    def build(self) -> TTAME:
        """Build the TTAME model."""
        # build the feature extractor
        self._build_fx()
        # build the attention mechanism
        self._build_attention()
        # build the model
        return TTAME(
            self.backbone,
            self.attention,
            self.masking_procedure,
        )

    @torch.no_grad()
    def build_pipeline(self) -> TTAME:
        """Build the TTAME model."""
        # build the feature extractor
        self._build_fx()
        # build the attention mechanism
        self._build_attention()

        def ttame_backbone_step(pipeline, _default_step):
            def backbone_step(x: dict) -> dict:
                """Forward pass of the backbone."""
                if (inp := x.get("masked_images", None)) is not None:
                    x["logits_masked"], _ = pipeline.backbone(inp)
                    x["targets_masked"] = x["logits_masked"].argmax(dim=1)
                else:
                    x["logits"], x["features"] = pipeline.backbone(x["images"])
                    x["targets"] = x["logits"].argmax(dim=1)
                return x

            return backbone_step

        self.kwargs["hooks"] = {
            "backbone": ttame_backbone_step,
            "masked_backbone": ttame_backbone_step,
        } | self.kwargs.get("hooks", {})
        # build the model
        return PtamePipeline(
            self.backbone,
            self.attention,
            self.masking_procedure,
            **self.kwargs,
        )

    def _build_fx(self):
        """Build feature extractor."""
        # if no layers are specified, print layers and quit
        train_names, eval_names = get_graph_node_names(self.backbone)
        if self.layers == [] or self.layers is None:
            print(train_names)
            quit()
        # get the output layer name
        output = (train_names[-1], eval_names[-1])
        if output[0] != output[1]:
            print(
                "WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE"
            )
        self.output_name = output[0]
        # get feature extractor
        self.backbone = create_feature_extractor(
            self.backbone, return_nodes=(self.layers + [self.output_name])
        )
        # Dry run to get number of channels of each layer for the attention mechanism
        if self.input_dim is not None:
            inp = torch.randn(self.input_dim)
        else:
            inp = torch.randn(1, 3, 224, 224)
        self.backbone.eval()
        outputs = self.backbone(inp)
        outputs.pop(self.output_name)
        features = outputs.values()
        self.feature_size = [o.shape for o in features]
        self.backbone.register_forward_hook(self._simplify_graph_outputs())

    def _build_attention(self):
        """Build the attention mechanism."""
        # check if the model is a transformer
        if self._is_transformer():
            feature_size = [
                torch.Size([2, ft[-1], 14, 14]) for ft in self.feature_size
            ]
            self.attention = self.attention(feature_size, self.num_classes)
            self.attention.register_forward_pre_hook(
                self._feature_adapter_vit_b_16(), with_kwargs=True
            )
        else:
            self.attention = self.attention(
                self.feature_size, self.num_classes
            )

    def _is_transformer(self) -> bool:
        """Check if the model is a transformer.

        Returns:
            bool: True if the model is a transformer, False otherwise.
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.MultiheadAttention):
                return True
        return False

    def _simplify_graph_outputs(self):
        """Returns a hook to simplify the outputs of the GraphModule feature
        extractor."""

        def hook(
            module, inputs, outputs
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            y: torch.Tensor = outputs.pop(self.output_name)
            features = list(outputs.values())
            return y, features

        return hook

    def _feature_adapter_vit_b_16(self):
        """Returns a hook to adapt the features of the Vision Transformer
        model."""

        def hook(module, args, kwargs) -> List[torch.Tensor]:
            seq_list = kwargs["features"]
            # discard class token
            seq_list = [seq[:, 1:, :] for seq in seq_list]
            # reshape
            seq_list = [
                seq.reshape(seq.size(0), 14, 14, seq.size(2))
                for seq in seq_list
            ]
            # bring channels after batch dimension
            seq_list = [
                seq.transpose(2, 3).transpose(1, 2) for seq in seq_list
            ]
            kwargs["features"] = seq_list
            return args, kwargs

        return hook


def build_ttame(**kwargs) -> TTAME:
    """Build the TTAME model.

    Args:
        Same as TTAMEBuilder.
    Returns:
        TTAME: The TTAME model.
    """
    builder = TTAMEBuilder(**kwargs)
    return builder.build()


def build_ttame_pipeline(**kwargs) -> TTAME:
    """Build the TTAME model.

    Args:
        Same as TTAMEBuilder.
    Returns:
        TTAME: The TTAME model.
    """
    builder = TTAMEBuilder(**kwargs)
    return builder.build_pipeline()
