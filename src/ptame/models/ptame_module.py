from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric, Metric
from torchmetrics.classification.accuracy import Accuracy

from ptame.utils.measures import Composer

from .components.loss import Loss


class PTAMELitModule(LightningModule):
    """`LightningModule` for PTAME.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: Loss,
        optimizer: Callable[
            [Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ],
        scheduler: dict[str, Any] | None,
        val_measures: dict[str, Metric] | None,
        test_measures: dict[str, Metric] | None,
        feature_contribution: bool = False,
        terminate_on_nan: bool = True,
        compile: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a `PTAMELitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training,
            together with the options (how often it should update etc.).
        :param val_measures: The explainability specific validation measures to
            use.
        :param test_measures: " test ".
        :param compile: Whether to compile the model before training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            ignore=[
                "net",
                "loss",
                "val_measures",
                "test_measures",
                "optimizer",
                "scheduler",
            ],
            logger=False,
        )

        self.net = net

        # cut off initialization for simple restore
        if loss is None:
            return

        # loss function
        self.criterion = loss

        # optimizer and scheduler
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        # metric objects are initialized in `setup` method
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None
        # for averaging loss across batches
        self.train_losses = nn.ModuleList(
            [MeanMetric() for _ in range(self.criterion.num_terms)]
        )
        self.val_losses = nn.ModuleList(
            [MeanMetric() for _ in range(self.criterion.num_terms)]
        )
        self.test_loss = MeanMetric()
        # metric object for calculating and averaging ADIC or ROAD or across batches
        self.val_measures = (
            Composer(nn.ModuleList(val_measures.values()), prefix="val/")
            if val_measures
            else None
        )
        self.test_measures = (
            Composer(nn.ModuleList(test_measures.values()), prefix="test/")
            if test_measures
            else None
        )

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if self.hparams.compile:
            return self.compiled_net(x)
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        [val_loss.reset() for val_loss in self.val_losses]
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y_ground = batch
        out = self.forward(x)
        losses = self.criterion(**out, epoch=self.trainer.current_epoch)
        return losses, out

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training
        set.

        :param batch: A batch of data (a tuple) containing the input tensor of
            images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, out = self.model_step(batch)
        preds_masked = out["logits_masked"]
        targets = out["targets"]
        # update and log metrics
        for i, metric in enumerate(self.train_losses):
            metric(losses[i])  # compute metric
            self.log(f"train/loss[{i}]", metric, prog_bar=True)
        self.train_acc(preds_masked, targets)
        self.log("train/acc", self.train_acc, prog_bar=True)

        # return loss or backpropagation will fail
        if self.hparams.terminate_on_nan and losses[0].isnan().any():
            raise ValueError("NaN detected in loss!")
        return losses[0]

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called before a validation epoch begins."""
        if self.val_measures:
            if self.hparams.compile:
                self.val_measures.register_net(self.compiled_net)
            else:
                self.val_measures.register_net(self.net)

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the
        validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of
            images and target labels.
        :param batch_idx: The index of the current batch.
        """
        losses, out = self.model_step(batch)
        preds = out["logits"]
        preds_masked = out["logits_masked"]
        targets = out["targets"]
        maps = out["masks"]

        # update and log metrics
        for i, metric in enumerate(self.val_losses):
            metric.update(losses[i])
            self.log(f"val/loss[{i}]", metric)
        self.val_acc.update(preds_masked, targets)
        self.log("val/acc", self.val_acc)
        if self.val_measures:
            self.val_measures.update(batch[0], preds, targets, maps)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        if self.val_measures:
            self.log_dict(self.val_measures.compute())
            self.val_measures.reset()

    def on_test_epoch_start(self) -> None:
        """Lightning hook that is called before a test epoch begins."""
        if self.test_measures:
            if self.hparams.compile:
                self.test_measures.register_net(self.compiled_net)
            else:
                self.test_measures.register_net(self.net)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of
            images and target labels.
        :param batch_idx: The index of the current batch.
        """
        losses, out = self.model_step(batch)
        preds = out["logits"]
        preds_masked = out["logits_masked"]
        targets = out["targets"]
        maps = out["masks"]

        # update and log metrics
        self.test_loss.update(losses[0])
        self.log("test/loss", self.test_loss)
        self.test_acc.update(preds_masked, targets)
        self.log("test/acc", self.test_acc)
        if save_masks := self.hparams.get("save_masks"):
            Path(save_masks).mkdir(parents=True, exist_ok=True)
            torch.save(maps, f"{save_masks}/{batch_idx}.pt")
        if self.test_measures:
            self.test_measures.update(batch[0], preds, targets, maps)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if self.hparams.feature_contribution:
            layer_names, contribs = self.net.attention.get_contributions()
            contribs_dict = {
                layer: contrib
                for layer, contrib in zip(layer_names, contribs.mean(dim=0))
            }
            self.log_dict(contribs_dict)

        if self.test_measures:
            self.log_dict(self.test_measures.compute())
            self.test_measures.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        if self.hparams.compile:
            self.compiled_net = torch.compile(self.net)
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.trainer.datamodule.num_classes
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=self.trainer.datamodule.num_classes
        )
        self.test_acc = Accuracy(
            task="multiclass", num_classes=self.trainer.datamodule.num_classes
        )

    def teardown(self, stage: str) -> None:
        """Lightning hook that is called at the end of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to clean something up after the run.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile:
            del self.compiled_net

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer_cfg(params=self.trainer.model.parameters())
        if (scheduler_info := self.scheduler_cfg) is not None:
            scheduler = scheduler_info["scheduler"]
            if scheduler_info["name"] == "OneCycleLR":
                scheduler = partial(
                    scheduler,
                    total_steps=self.trainer.estimated_stepping_batches,
                )
            scheduler = scheduler(optimizer=optimizer)
            scheduler_params = scheduler_info["params"]
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": scheduler_params["interval"],
                    "frequency": scheduler_params["frequency"],
                    "name": scheduler_params["name"],
                },
            }
        return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint):
        # pop the keys you are not interested by
        sd = checkpoint["state_dict"]
        names = list(sd.keys())
        for name in names:
            if "backbone" in name or "_orig_mod" in name:
                sd.pop(name)
            if "net.attention.attention" in name:
                sd.pop(name)


if __name__ == "__main__":
    _ = PTAMELitModule(None, None, None, None, None, None)
