from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2 as transforms
from importlib_resources import as_file, files
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageNet, Imagenette

import ptame


def norm(x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """Normalize the input tensor.

    :param x: The input tensor.
    :param reverse: Whether to reverse the normalization. Defaults to `False`.
    :return: The normalized tensor.
    """
    if reverse:
        return transforms.Normalize(
            (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
            (1 / 0.229, 1 / 0.224, 1 / 0.225),
        )(x)
    else:
        return transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )(x)


class ImageNetDataModule(LightningDataModule):
    """`LightningDataModule` for the imagenet dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/ImageNet",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        test: bool = True,
        test_for_train: bool = False,
        imagenette: bool = False,
        imagenette_augs: bool = False,
        pct: float | None = None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_tsfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        self.imagenette_tsfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.AugMix(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        self.val_tsfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of imagenet classes (1000).
        """
        if self.hparams.imagenette:
            return 10
        else:
            return 1000

    def prepare_data(self):
        if self.hparams.imagenette:
            if not (p := Path(self.hparams.data_dir)).is_dir():
                p.mkdir(parents=False, exist_ok=False)
                Imagenette(self.hparams.data_dir, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
        dataset = Imagenette if self.hparams.imagenette else ImageNet
        train_tsfm = (
            self.train_tsfm
            if not (self.hparams.imagenette or self.hparams.imagenette_augs)
            else self.imagenette_tsfm
        )
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = dataset(
                self.hparams.data_dir, split="train", transform=train_tsfm
            )
            if self.hparams.pct is not None:
                self.data_train, _ = random_split(
                    dataset(
                        self.hparams.data_dir,
                        split="train",
                        transform=train_tsfm,
                    ),
                    [self.hparams.pct, 1 - self.hparams.pct],
                )
            if self.hparams.imagenette:
                self.data_val, self.data_test = random_split(
                    dataset(
                        self.hparams.data_dir,
                        split="val",
                        transform=self.val_tsfm,
                    ),
                    [0.5, 0.5],
                )
            else:
                datalist_source = files(ptame.datalists)
                with as_file(datalist_source.joinpath("val_set.pt")) as f:
                    self.data_val = Subset(
                        dataset(
                            self.hparams.data_dir,
                            split="val",
                            transform=self.val_tsfm,
                        ),
                        torch.load(
                            f,
                            weights_only=True,
                        ),
                    )
                with as_file(datalist_source.joinpath("test_set.pt")) as f:
                    self.data_test = Subset(
                        dataset(
                            self.hparams.data_dir,
                            split="val",
                            transform=self.val_tsfm,
                        ),
                        torch.load(
                            f,
                            weights_only=True,
                        ),
                    )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.hparams.test_for_train:
            return self.test_dataloader()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if not self.hparams.test:
            return self.val_dataloader()
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        :return: A dictionary containing the datamodule state that you want to
            save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule
        state given datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by
            `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ImageNetDataModule()
