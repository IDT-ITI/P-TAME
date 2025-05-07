import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Literal, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchshow as ts
from importlib_resources import as_file, files
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms

import ptame
from ptame.data.imagenet_datamodule import ImageNetDataModule
from ptame.utils.masking_utils import gaussian_masking, norm_mask


class SampleSet(ABC):
    """Base class for sample sets.

    These are used for printing maps.
    """

    @abstractmethod
    def __iter__(
        self,
    ) -> Iterator[Tuple[str, Tuple[torch.Tensor, torch.Tensor, int]]]:
        """Iterate over the sample set.

        Yields:
            Iterator[Tuple[torch.Tensor, torch.Tensor, int]]: Tuple containing
                the original image, the normalized image, and the ground truth
                label.
        """
        pass

    @abstractmethod
    def id2label(self, id: int) -> str:
        """Return the label for the given id.

        Args:
            id (int): The id to get the label for.

        Returns:
            str: One of the labels for the given id.
        """
        pass


class XAIModel(ABC):
    """Base class for XAI models."""

    @abstractmethod
    @torch.inference_mode()
    def produce_map(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Produce an explanation map for the given image.

        Args:
            image (torch.Tensor): The image to produce the explanation map for.

        Returns:
            Tuple[torch.Tensor, id]: The explanation map and the model's
            prediction.
        """
        pass

    @abstractmethod
    @torch.inference_mode()
    def produce_cdmaps(
        self, image: torch.Tensor
    ) -> tuple[list[torch.Tensor], int]:
        pass


class MapPrinter:
    """Base mapping procedure for qualitative examination of attentions
    maps."""

    # TODO: perhaps use lightning fabric to make this code GPU compatible
    def __init__(
        self,
        save_path: str,
        set_name: str,
        model: XAIModel,
        sampleset: SampleSet,
        upscaling: Literal["bilinear", "nearest"] = "bilinear",
        **kwargs,
    ) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.set_name = set_name
        self.save_path = Path(save_path) / self.set_name
        self.model = model
        self.set = sampleset
        self.kwargs = kwargs
        self.upscaling = upscaling

    def save_maps(self, id, image, norm_image, map, model_truth, ground_truth):
        # Print explanation map on features-size
        ts.save(
            map,
            f"{self.save_path}/{id}/small_map.png",
        )
        map = F.interpolate(
            map.unsqueeze(0).unsqueeze(0),
            size=(image.size(-2), image.size(-1)),
            mode=self.upscaling,
            # align_corners=False,
        )
        # Print explanation map after up-scale to input image size.
        ts.save(
            map,
            f"{self.save_path}/{id}/big_map.png",
        )
        # Print mapped input image
        ts.save(
            map * image,
            f"{self.save_path}/{id}/mapped_image.png",
        )
        ts.save(
            norm_mask(masking=gaussian_masking)(norm_image.unsqueeze(0), map),
            f"{self.save_path}/{id}/blurred_image.png",
        )
        # Print norm input image
        ts.save(
            norm_image,
            f"{self.save_path}/{id}/norm_image.png",
        )

        # Soft mapping w/ heat map to visualize impact
        opencvImage = cv2.cvtColor(
            image.squeeze().permute(1, 2, 0).cpu().numpy(),
            cv2.COLOR_RGB2BGR,
        )
        opencvImage = (np.asarray(opencvImage, np.float32) * 255).astype(
            np.uint8
        )
        np_map = np.array(
            map.squeeze().detach().cpu().numpy() * 255, dtype=np.uint8
        )
        np_map_hist = cv2.equalizeHist(np_map)
        np_map = cv2.applyColorMap(np_map, cv2.COLORMAP_JET)
        np_map_hist = cv2.applyColorMap(np_map_hist, cv2.COLORMAP_JET)
        map_image = cv2.addWeighted(np_map, 0.5, opencvImage, 0.5, 0)
        map_image_hist = cv2.addWeighted(np_map_hist, 0.5, opencvImage, 0.5, 0)
        cv2.imwrite(
            f"{self.save_path}/{id}/mdl_{self.set.id2label(model_truth)}"
            f"({model_truth})_gr_{self.set.id2label(ground_truth)}({ground_truth}).png",
            map_image,
        )
        cv2.imwrite(
            f"{self.save_path}/{id}/mdl_{self.set.id2label(model_truth)}"
            f"({model_truth})_gr_{self.set.id2label(ground_truth)}({ground_truth})_hist.png",
            map_image_hist,
        )
        cv2.imwrite(
            f"{self.save_path}/{id}.png",
            map_image,
        )
        cv2.imwrite(
            f"{self.save_path}/{id}/heatmap.png",
            np_map,
        )

    def print_maps(self):
        """Save maps produced by model and visualize them as heatmaps."""
        for id, sample in self.set:
            image, norm_image, ground_truth_label = sample
            ts.save(
                image,
                f"{self.save_path}/{id}/image.png",
            )
            # Produce explanation map
            # Map is 2D
            map, model_truth = self.model.produce_map(
                norm_image, classes=self.kwargs.get("classes", None)
            )
            if map.ndim < 3:
                self.save_maps(
                    id, image, norm_image, map, model_truth, ground_truth_label
                )
            else:
                for map_i, model_truth_i in zip(map, model_truth):
                    self.save_maps(
                        id,
                        image,
                        norm_image,
                        map_i,
                        model_truth_i,
                        ground_truth_label,
                    )


class ImageNetSampleSet(SampleSet):
    """Sample set for the ImageNet dataset."""

    def __init__(
        self,
        datamodule: ImageNetDataModule,
        ids: List[int],
        split: Literal["val", "test"] = "val",
        **kwargs,
    ):
        self.datamodule = datamodule
        self.ids = ids
        self.split = split
        self.tsfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        self.norm_tsfm = transforms.Compose(
            [
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                )
            ]
        )
        # sub = torch.load(Path(self.datamodule.hparams.datalist) / "test_set.pt")
        # print("Corresponding ids are", [sub[i].item() for i in [1, 15, 22, 56, 165, 219, 309]])
        self.imagenet = ImageNet(
            self.datamodule.hparams.data_dir,
            split="val",
            transform=self.tsfm,
        )
        with as_file(files(ptame.datalists).joinpath(f"{split}_set.pt")) as f:
            self.dataset = Subset(
                Subset(
                    self.imagenet,
                    torch.load(
                        f,
                        weights_only=True,
                    ),
                ),
                ids,
            )

        if img_path := kwargs.get("single_image", False):
            self.ids = [0]
            with open(img_path, "rb") as f:
                img = Image.open(f)
                self.dataset = [(self.tsfm(img.convert("RGB")), 0)]

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.ids)

    def __iter__(
        self,
    ) -> Iterator[Tuple[str, Tuple[torch.Tensor, torch.Tensor, int]]]:
        """Iterate over the sample set.

        Yields:
            Iterator[Tuple[torch.Tensor, torch.Tensor, int]]: Tuple containing
                the original image, the normalized image, and the ground truth
                label.
        """
        for id, img_id in enumerate(self.ids):
            sample, target = self.dataset[id]
            norm_sample = self.norm_tsfm(sample)
            img_id = self.split + "-" + str(img_id)
            yield img_id, (sample, norm_sample, target)

    def id2label(self, id: int) -> str:
        """Return the first label for the given id.

        Args:
            id (int): The id to get the label for.

        Returns:
            str: One of the labels for the given id.
        """
        return self.imagenet.classes[id][0].replace(" ", "-")
