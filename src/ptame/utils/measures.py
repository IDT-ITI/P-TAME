import abc
import math
from functools import partial
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Metric

from ptame.data.imagenet_datamodule import norm

from .masking_utils import binary_mask_thresholding, norm_mask, resize_thresh
from .noisy_linear_imputation import InpaintingImputer, NoisyLinearImputer


class ExplMetric(Metric, metaclass=abc.ABCMeta):
    """A metric that is used to evaluate the explainability of a model."""

    def register_net(self, net: nn.Module) -> None:
        self.net = net.get_predictions
        self.last_percent = -1

    @torch.no_grad()
    def _compute_masked_scores(
        self, x: Tensor, masks: Tensor, percent: float, force: bool = False
    ) -> Tensor:
        if force or not math.isclose(percent, self.last_percent):
            masks = self.thresholding_procedure(masks, percent)
            self.x_masked: Tensor = self.masking_procedure(x, masks)
            self.last_percent = percent
        return self.x_masked

    def calc_masked_scores(
        self,
        x: Tensor,
        masks: Tensor,
        targets: Tensor,
        percent: float,
        force: bool = False,
    ) -> Tensor:
        """Calculate the masked scores."""
        x_masked: Tensor = self._compute_masked_scores(
            x, masks, percent, force
        )
        return (
            self.net(x_masked).softmax(dim=1).gather(1, targets.unsqueeze(-1))
        )

    def calc_masked_accuracy(
        self, x: Tensor, masks: Tensor, targets: Tensor, percent: float
    ) -> Tensor:
        """Calculate the masked accuracy."""
        x_masked: Tensor = self._compute_masked_scores(x, masks, percent)
        return (self.net(x_masked).argmax(dim=1) == targets).float()

    def _get_ad(
        self, original_scores: Tensor, masked_scores: Tensor
    ) -> Tensor:
        """Calculate the AD."""
        return (
            ((original_scores - masked_scores).clip(min=0) / original_scores)
            .nan_to_num()
            .mean()
        )

    def _get_ic(
        self, original_scores: Tensor, masked_scores: Tensor
    ) -> Tensor:
        """Calculate the IC."""
        return ((masked_scores - original_scores) > 0).mean(dtype=torch.float)


class ADIC(ExplMetric):
    """Classic ADIC measure (100, 50, 15)"""

    def __init__(
        self,
        thresholding_procedure=resize_thresh(),
        masking_procedure=norm_mask(),
        std: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state(
            "AD", default=torch.tensor([0.0, 0.0, 0.0]), dist_reduce_fx="sum"
        )
        self.add_state(
            "IC", default=torch.tensor([0.0, 0.0, 0.0]), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.std = std
        if std:
            self.add_state(
                "AD_std",
                default=torch.tensor([0.0, 0.0, 0.0]),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "IC_std",
                default=torch.tensor([0.0, 0.0, 0.0]),
                dist_reduce_fx="sum",
            )
        self.masking_procedure = masking_procedure
        self.thresholding_procedure = thresholding_procedure
        self.net = None

    @torch.no_grad()
    def update(
        self, input: Tensor, scores: Tensor, targets: Tensor, masks: Tensor
    ) -> None:
        """Update the ADIC measure."""
        original_scores = scores.softmax(dim=1).gather(
            1, targets.unsqueeze(-1)
        )
        for i, percent in enumerate([1, 0.5, 0.15]):
            masked_scores = self.calc_masked_scores(
                input, masks, targets, percent
            )
            ad_sample = self._get_ad(original_scores, masked_scores)
            ic_sample = self._get_ic(original_scores, masked_scores)
            self.AD[i] += ad_sample
            self.IC[i] += ic_sample
            if self.std:
                self.AD_std[i] += ad_sample**2
                self.IC_std[i] += ic_sample**2

        self.total += 1

    def compute(self) -> Tensor:
        """Compute the ADIC measure."""
        pcts = [100, 50, 15]
        s0 = self.total.clone()
        ads = self.AD.clone() / s0
        ics = self.IC.clone() / s0

        AD_dict = {f"AD {pct}%": ad for ad, pct in zip(ads, pcts)}
        IC_dict = {f"IC {pct}%": ic for ic, pct in zip(ics, pcts)}
        if self.std:
            ads_sq = self.AD_std.clone() / s0
            ics_sq = self.IC_std.clone() / s0
            for ad, ad_s, ic, ic_s, pct in zip(ads, ads_sq, ics, ics_sq, pcts):
                AD_dict[f"AD {pct}% std"] = ((ad_s - ad**2) / s0).sqrt()
                IC_dict[f"IC {pct}% std"] = ((ic_s - ic**2) / s0).sqrt()
        return AD_dict | IC_dict


class ModifiedNoisyLinearImputer:
    """Modified Noisy Linear Imputer.

    This imputer is used for the ROAD metric. It is modified to change the
    default __call__ method to batched_call.
    """

    def __init__(self, legacy=False):
        super().__init__()
        if not legacy:
            self.imputer = InpaintingImputer()
        else:
            self.imputer = NoisyLinearImputer()

    def __call__(self, x: Tensor, masks: Tensor) -> Tensor:
        """Call batched_call of imputer."""
        _, _, H, W = x.size()
        x = norm(x, reverse=True)
        masks = masks.float()
        masks = F.interpolate(
            masks, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()
        return norm(self.imputer.batched_call(x, masks))


class ROAD(ExplMetric):
    """ROAD confidence scores (0, 10, 30, 50, 70, 90)"""

    def __init__(
        self,
        imputer: nn.Module = ModifiedNoisyLinearImputer(),
        percentages: List[float] = [0.1, 0.3, 0.5, 0.7],
        MoRF: bool = False,
        thresholding_procedure=None,
        accuracy: bool = True,
        mean: bool = True,
        std: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.percentages = percentages
        self.MoRF = MoRF
        self.accuracy = accuracy
        self.mean = mean
        self.std = std
        self.masking_procedure = imputer
        if thresholding_procedure is None:
            self.thresholding_procedure = resize_thresh(
                thresholding=partial(binary_mask_thresholding, most=MoRF)
            )
        else:
            self.thresholding_procedure = thresholding_procedure
        self.add_state(
            "scores",
            default=torch.zeros(len(percentages)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        if std:
            self.add_state(
                "squared_scores",
                default=torch.zeros(len(percentages)),
                dist_reduce_fx="sum",
            )
        self.net = None

    @torch.no_grad()
    def update(
        self, input: Tensor, scores: Tensor, targets: Tensor, masks: Tensor
    ) -> None:
        """Update the ROAD measures."""
        for i, percent in enumerate(self.percentages):
            if self.accuracy:
                sample = self.calc_masked_accuracy(
                    input, masks, targets, percent
                ).mean()
            else:
                sample = self.calc_masked_scores(
                    input, masks, targets, percent
                ).mean()
            self.scores[i] += sample
            if self.std:
                self.squared_scores[i] += sample**2
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the ROAD measures."""
        pcts = self.percentages
        metric_name = "MoRF" if self.MoRF else "LeRF"

        # Do not alter the state
        s0 = self.total.clone()
        s1 = self.scores.clone() / s0

        if not self.accuracy:
            metric_name = metric_name + "(conf)"
        scores_dict = {
            f"{metric_name} {pct:.0%}": value for value, pct in zip(s1, pcts)
        }
        if self.mean:
            # we will calculate the mean of the scores in a complicated way
            # so that it can be compared to the old implementation
            # to compare with older implementation, multiply old mean by 0.93
            # we will use the trapezoidal rule to calculate the area under the
            # curve and then divide by the length of the curve
            s1_ext = torch.cat(
                [s1.new_tensor([1.0]), s1, s1.new_tensor([0.0])]
            )
            pcts_ext = pcts.copy()
            pcts_ext.insert(0, 0.0)
            pcts_ext.append(1.0)
            auc = sum(
                (y1 + y2) * (x2 - x1) / 2
                for x1, x2, y1, y2 in zip(
                    pcts_ext[:-1], pcts_ext[1:], s1_ext[:-1], s1_ext[1:]
                )
            )
            scores_dict[f"{metric_name}"] = auc
        if self.std:
            s2 = self.squared_scores.clone() / s0
            ses = []
            for pct, score, sscore in zip(pcts, s1, s2):
                se = ((sscore - score**2) / s0).sqrt()
                scores_dict[f"{metric_name} {pct:.0%} std"] = se
                ses.append(se.clone())
            if self.mean:
                ses.insert(0, 0.0)
                ses.append(0.0)
                total_variance = sum(
                    ((x2 - x1) / 2) ** 2 * (se1**2 + se2**2)
                    for x1, x2, se1, se2 in zip(
                        pcts_ext[:-1], pcts_ext[1:], ses[:-1], ses[1:]
                    )
                )
                auc_standard_error = (total_variance).sqrt()
                scores_dict[f"{metric_name} std"] = auc_standard_error
        return scores_dict


class Composer(Metric):
    """Compose multiple metrics together."""

    def __init__(self, metrics: nn.ModuleList, prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        self.prefix = prefix

    def register_net(self, net: nn.Module) -> None:
        for metric in self.metrics:
            metric.register_net(net)

    def update(self, *args, **kwargs) -> None:
        """Update all the metrics."""
        for metric in self.metrics:
            metric.update(*args, **kwargs)

    def compute(self) -> Tensor:
        """Compute all the metrics."""
        dicts = [metric.compute() for metric in self.metrics]
        return {
            f"{self.prefix}{key}": value
            for d in dicts
            for key, value in d.items()
        }

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
