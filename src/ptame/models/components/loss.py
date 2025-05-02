import abc
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch.nn.functional import cross_entropy


@dataclass
class Loss(abc.ABC):
    """Loss is an abstract class that represents a loss function for the TAME-
    based family of explainabiltiy methods."""

    num_terms: ClassVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        logits_masked: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        epoch: int,
    ) -> list[torch.Tensor]:
        """Calculates the loss based on the logits, targets, and masks.

        Args:
            logits (torch.Tensor): The predicted logits.
            logits_masked (torch.Tensor): The predicted logits with the masks applied.
            targets (torch.Tensor): The targets.
            masks (torch.Tensor): The masks tensor.

        Returns:
            List[torch.Tensor]: A list of calculated losses.
        """

    def area_loss(self, masks):
        """Calculates the area loss based on the masks.

        Args:
            masks (torch.Tensor): The masks tensor.

        Returns:
            torch.Tensor: The calculated area loss.
        """
        if self.area_loss_power != 1:
            # add e to prevent nan (derivative of sqrt at 0 is inf)
            masks = (masks + 0.0005) ** self.area_loss_power
        return torch.mean(masks)

    def smoothness_loss(self, masks):
        """Calculates the smoothness loss based on the masks.

        Args:
            masks (torch.Tensor): The masks tensor.

        Returns:
            torch.Tensor: The calculated smoothness loss.
        """
        B, _, _, _ = masks.size()
        border_penalty = self.smoothness_border_penalty
        power = self.smoothness_power
        x_loss = torch.sum(
            (torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])) ** power
        )
        y_loss = torch.sum(
            (torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])) ** power
        )
        if border_penalty > 0:
            border = float(border_penalty) * torch.sum(
                masks[:, :, -1, :] ** power
                + masks[:, :, 0, :] ** power
                + masks[:, :, :, -1] ** power
                + masks[:, :, :, 0] ** power
            )
        else:
            border = 0.0
        return (x_loss + y_loss + border) / float(power * B)


@dataclass
class ClassicLoss(Loss):
    """ClassicLoss is a class that represents the loss function used in TAME.

    It calculates the loss based on cross-entropy, area loss, and smoothness
    loss.

    :param ce_coeff: The coefficient for the cross-entropy loss.
    :param area_coeff: The coefficient for the area loss.
    :param smoothness_coeff: The coefficient for the smoothness loss.
    :param smoothness_power: The power for the smoothness loss.
    :param smoothness_border_penalty: The penalty for the smoothness loss at
        the border.
    :param area_loss_power: The power for the area loss.
    :param num_terms: The number of terms returned by the loss function.
    """

    ce_coeff: float = 1.5
    area_coeff: float = 2
    smoothness_coeff: float = 0.01
    smoothness_power: float = 2
    smoothness_border_penalty: float = 0.3
    area_loss_power: float = 0.3
    num_terms: ClassVar[int] = 4

    def __call__(
        self,
        logits_masked: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Calculates the overall loss based on the logits, targets, and masks.

        Args:
            logits (torch.Tensor): The predicted logits.
            logits_masked (torch.Tensor): The predicted logits with the masks applied.
            targets (torch.Tensor): The targets.
            masks (torch.Tensor): The masks tensor.

        Returns:
            List[torch.Tensor]: A list of calculated losses, including the overall loss,
            cross-entropy loss, area loss, and smoothness loss.
        """
        targets = targets.long()
        variation_loss = torch.tensor(0)
        area_loss = torch.tensor(0)
        ce_loss = torch.tensor(0)
        if self.smoothness_coeff > 0:
            variation_loss = self.smoothness_loss(masks)
        if self.area_coeff > 0:
            area_loss = self.area_loss(masks)
        if self.ce_coeff > 0:
            ce_loss = cross_entropy(logits_masked, targets)
        ce_loss = cross_entropy(logits_masked, targets)

        loss = (
            self.ce_coeff * ce_loss
            + self.area_coeff * area_loss
            + self.smoothness_coeff * variation_loss
        )

        return [loss, ce_loss, area_loss, variation_loss]
