from typing import Any

import hydra
import torch
from lightning.pytorch.utilities import measure_flops
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from ptame.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> bool:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    try:
        trainer.loggers
    except AttributeError:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    # measure model FLOPs
    test_model = hydra.utils.instantiate(cfg["model"])
    num_batches = 1
    x = torch.randn(num_batches, 3, 224, 224)

    if cfg.get("log_flops", True):
        # measure the flops caused by the backbone
        bb_flops = 2 * measure_flops(
            test_model.net, lambda: test_model.net.get_predictions(x)
        )
        fwd_flops = measure_flops(test_model.net, lambda: test_model.net(x))
        hparams["model/gflops/fwd"] = fwd_flops / 1e9
        hparams["model/gflops/bb_fwd"] = bb_flops / 1e9
        if hparams["model/params/trainable"] != 0:
            fwd_and_bwd_flops = measure_flops(
                test_model.net,
                lambda: test_model.net(x),
                lambda out: test_model.criterion(**out)[0],
            )
            hparams["model/gflops/bwd"] = fwd_and_bwd_flops / 1e9
        else:
            hparams["model/gflops/bwd"] = 0

    # save explanation resolution
    if hasattr(model.net, "attention"):
        hparams["model/attention/resolution"] = model.net.attention.resolution
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
    return hparams["model/params/trainable"] != 0
