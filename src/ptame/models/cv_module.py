from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class CVModule(LightningModule):
    """`LightningModule` for PAMELA.

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
        loss: Callable[[Tensor], Tensor],
        optimizer: Callable[
            [Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ],
        scheduler: dict[str, Any] | None,
        teacher: Callable[[Tensor], Tensor] | None = None,
        terminate_on_nan: bool = True,
        compile: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a `PAMELALitModule`.

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
            ignore=["net", "loss", "optimizer", "scheduler", "teacher"],
            logger=False,
        )

        self.net = net
        self.teacher = None
        if teacher is not None:
            self.teacher = teacher
            loss = self.kl_loss
            self.teacher.eval()
            self.teacher.requires_grad_(False)

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
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # metric object for calculating and averaging ADIC or ROAD or across batches

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    @staticmethod
    def mse_loss(student_logits, teacher_logits):
        return torch.nn.functional.mse_loss(student_logits, teacher_logits)

    @staticmethod
    def kl_loss(student_logits, teacher_logits, temperature=5.0):
        p_t = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        p_s = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
        return torch.nn.functional.kl_div(
            p_s.log(), p_t, reduction="batchmean"
        )

    def convert_fc(self) -> None:
        """Convert the last layer of the model to a fully connected layer."""
        if hasattr(self.net, "fc"):
            in_features = self.net.fc.in_features
            out_features = (
                self.trainer.datamodule.num_classes
                if self.teacher is None
                else self.teacher.fc.out_features
            )
            self.classes = out_features
            if self.net.fc.out_features == out_features:
                return
            self.net.fc = torch.nn.Linear(in_features, out_features)
            # initialize weights
            torch.nn.init.xavier_uniform_(self.net.fc.weight)
            if self.net.fc.bias is not None:
                torch.nn.init.zeros_(self.net.fc.bias)

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
        self.val_loss.reset()
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
        x, y = batch
        target = y
        if self.teacher is not None:
            if self.hparams.compile:
                y = self.compiled_teacher(x)
            else:
                y = self.teacher(x)
            target = y.argmax(dim=1)
        out = self.forward(x)
        loss = self.criterion(out, y)
        return loss, out, target

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
        loss, out, target = self.model_step(batch)
        preds = out.argmax(dim=1)
        # update and log metrics
        self.train_loss(loss)  # compute metric
        self.log("train/loss", self.train_loss, prog_bar=True)
        self.train_acc(preds, target)
        self.log("train/acc", self.train_acc, prog_bar=True)

        # return loss or backpropagation will fail
        if self.hparams.terminate_on_nan and loss.isnan().any():
            raise ValueError("NaN detected in loss!")
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the
        validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of
            images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, out, target = self.model_step(batch)
        preds = out.argmax(dim=1)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss)
        self.val_acc(preds, target)
        self.log("val/acc", self.val_acc)

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

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of
            images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, out, target = self.model_step(batch)
        preds = out.argmax(dim=1)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss)
        self.test_acc(preds, target)
        self.log("test/acc", self.test_acc)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.convert_fc()
        if self.hparams.compile:
            self.compiled_net = torch.compile(self.net)
            if self.teacher is not None:
                self.compiled_teacher = torch.compile(self.teacher)
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.classes)

    def teardown(self, stage: str) -> None:
        """Lightning hook that is called at the end of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to clean something up after the run.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile:
            del self.compiled_net
            if self.teacher:
                del self.compiled_teacher

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
            if any(x in name for x in ["_orig_mod", "fc", "teacher"]):
                sd.pop(name)


if __name__ == "__main__":
    _ = CVModule(None, None, None, None, None, None)
