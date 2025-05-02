# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
MemoryFormat
============

changes the model memory format
"""

from collections.abc import MutableSequence
from typing import Any, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class MemoryFormat(Callback):
    """The `MemoryFormat` callback changes the model memory format to
    `torch.channels_last` before training starts and returns the original when
    it ends.

    <https://\\pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_.
    Setting the memory format channels_last usually improves GPU utilization.
    Runs on setup, so it can set the memory format before the model is DDP wrapped.
    """

    def __init__(
        self,
        memory_format: torch.memory_format = torch.channels_last,
        convert_input: bool = False,
    ):
        self.memory_format = memory_format
        self.convert_input = convert_input

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        """
        Sets up the memory format for the given PyTorch Lightning module during the training process.
        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module to be configured.
            stage (Optional[str]): The stage of the training process. Defaults to None.
        Notes:
            - If the specified memory format (e.g., `torch.channels_last` or `torch.channels_last_3d`)
              does not benefit any layers in the model, a warning will be issued.
            - The `pl_module` is moved to the specified memory format.
        """

        if self.memory_format in (
            torch.channels_last,
            torch.channels_last_3d,
        ) and not self.has_layer_benefiting_from_channels_last(pl_module):
            rank_zero_warn(
                f"model does not have any layers benefiting from {self.memory_format} format",
                category=RuntimeWarning,
            )

        pl_module.to(memory_format=self.memory_format)

    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        """Handles the teardown process for the PyTorch Lightning module by
        converting the module's memory format to contiguous format.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module being trained.
            stage (Optional[str]): The stage of the training process (e.g., 'fit', 'test', etc.).
                Defaults to None.
        Returns:
            None
        """

        pl_module.to(memory_format=torch.contiguous_format)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook that is called at the start of each training batch.

        This method is used to optionally convert the input batch's tensors to a specified memory format.
        If the `convert_input` flag is set to `False`, the method exits early without performing any conversion.
        If the batch is not a `MutableSequence`, a warning is issued, and no conversion is performed.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
            batch (Any): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            None
        """
        if not self.convert_input:
            return

        if not isinstance(batch, MutableSequence):
            rank_zero_warn(
                f"batch is not a MutableSequence, cannot convert input to {self.memory_format}",
                category=RuntimeWarning,
            )
            return

        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                batch[i] = item.to(memory_format=self.memory_format)

    benefitial_layers = (
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )

    def has_layer_benefiting_from_channels_last(
        self, model: torch.nn.Module
    ) -> bool:
        """Determines if the given model contains any layer that would benefit
        from using the "channels_last" memory format.

        Args:
            model (torch.nn.Module): The PyTorch model to inspect.

        Returns:
            bool: True if at least one layer in the model is an instance of
            the types specified in `self.benefitial_layers`, otherwise False.
        """
        return any(
            isinstance(layer, self.benefitial_layers)
            for layer in model.modules()
        )
