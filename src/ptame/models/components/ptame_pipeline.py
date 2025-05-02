from collections.abc import Callable

import torch
from torch import nn

from ptame.utils.map_printer import XAIModel
from ptame.utils.masking_utils import (
    norm_resize_mask,
    random_map_select,
    single_map_select,
)


class PtamePipeline(nn.Module, XAIModel):
    """Perturbation-based Attention Mechanism for Explaining bLAck-box models
    (PAMELA)"""

    def __init__(
        self,
        backbone: nn.Module,
        attention: nn.Module,
        masking_procedure=norm_resize_mask,
        train_map_select=random_map_select,
        matching_map_select=single_map_select,
        eval_map_select=None,
        backbone_eval=True,
        return_keys: list[str] = [
            "logits",
            "masks",
            "logits_masked",
            "targets",
        ],
        hooks: dict[str, Callable] = {},
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        if backbone_eval:
            self.backbone.eval()
        self.attention = attention
        self.masking_procedure = masking_procedure
        self.train_map_select = train_map_select
        self.matching_map_select = matching_map_select
        self.eval_map_select = (
            a if (a := eval_map_select) is not None else matching_map_select
        )
        self.default_pipeline = {
            "backbone": self.backbone_step,
            "attention": self.attention_step,
            "masking": self.masking_step,
            "masked_backbone": self.backbone_step,
            "map_selection": self.selection_step,
        }
        phases = ["before_backbone"]
        phases += [f"after_{step}" for step in self.default_pipeline.keys()]
        self.hooks = {phase: [] for phase in phases}
        for phase, hook in hooks.items():
            if phase in self.default_pipeline.keys():
                # a total override of a step
                self.default_pipeline[phase] = hook(
                    self, self.default_pipeline[phase]
                )
            elif phase in phases:
                # a hook around a step
                self.hooks[phase].append(hook)
            else:
                raise ValueError(f"Unknown phase: {phase}")
        self.return_keys = return_keys

    def _select_returned_maps(self, saliency_maps, targets):
        """Select the saliency maps that will be returned based on the targets
        and the stage."""
        if self.training:
            return self.train_map_select(saliency_maps, targets)
        else:
            return self._select_matching_maps(saliency_maps, targets)

    def _select_matching_maps(self, saliency_maps, targets):
        """Get the saliency maps from the attention mechanism.

        Depending on the stage and, masks as selected using a different
        procedure.
        """
        if self.training:
            return self.matching_map_select(saliency_maps, targets)
        else:
            return self.eval_map_select(saliency_maps, targets)

    def backbone_step(self, x: dict) -> dict:
        """Forward pass of the backbone."""
        if (inp := x.get("masked_images", None)) is not None:
            x["logits_masked"] = self.backbone(inp)
            x["targets_masked"] = x["logits_masked"].argmax(dim=1)
        else:
            x["logits"] = self.backbone(x["images"])
            x["targets"] = x["logits"].argmax(dim=1)
        return x

    def attention_step(self, x: dict) -> dict:
        """Forward pass of the attention mechanism."""
        x["maps"] = self.attention(**x)
        return x

    def masking_step(self, x: dict) -> dict:
        """Apply the masks to the input tensor and get the output."""
        x["masked_images"] = self.masking_procedure(
            x["images"], self._select_matching_maps(x["maps"], x["targets"])
        )
        return x

    def selection_step(self, x: dict) -> dict:
        """Select the saliency maps that will be returned based on the targets
        and the stage."""
        x["masks"] = self._select_returned_maps(x["maps"], x["targets"])
        return x

    def run_hooks(self, phase: str, x: dict) -> dict:
        """Run the hooks for the given phase."""
        for hook in self.hooks[phase]:
            x = hook(self, x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the entire model.

        Depending on the stage, different outputs are returned.
        """
        data = {"images": x, "training": self.training}
        data = self.run_hooks("before_backbone", data)
        for step_name, step in self.default_pipeline.items():
            data = step(data)
            data = self.run_hooks("after_" + step_name, data)

        return {
            return_key: data[return_key] for return_key in self.return_keys
        }

    def produce_map(
        self, image: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, int]:
        """Produce a saliency map for the given image."""
        data = {"images": image.unsqueeze(0), "training": False}
        data = self.run_hooks("before_backbone", data)
        for step_name, step in self.default_pipeline.items():
            data = step(data)
            data = self.run_hooks("after_" + step_name, data)
        if classes := kwargs.get("classes", False):
            top_classes = torch.topk(data["logits"], classes)[1]
            maps = data["maps"].squeeze()[top_classes]
            return maps, top_classes
        else:
            map = data["masks"].squeeze()
            return map, data["targets"].squeeze()

    def produce_cdmaps(
        self, image: torch.Tensor
    ) -> tuple[list[torch.Tensor], int]:
        """Produce a list of saliency maps for the given image."""
        image = image.unsqueeze(0)
        out = self.backbone(image)
        maps = self.attention(image)
        predictions = out.squeeze()
        return maps, predictions

    def get_predictions(self, x):
        """Get the predictions of the model.

        Useful for the measures.
        """
        return self.default_pipeline["backbone"]({"images": x})["logits"]
