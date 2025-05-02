from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from ptame.utils import (  # noqa: E402
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if ckpt_path := cfg.get("ckpt_path"):
        log.info(f"Loading model from checkpoint path {ckpt_path}")
        incompatible_keys = model.load_state_dict(
            torch.load(ckpt_path, weights_only=True)["state_dict"],
            strict=False,
        )
        # get compatible keys
        concrete_keys = model.state_dict().keys()
        log.info(
            f"Compatible keys: {set(concrete_keys) - set(incompatible_keys[0])}"
        )
        log.info(f"Unexpected keys: {incompatible_keys[1]}")

    if aux_path := cfg.get("aux_ckpt_path"):
        log.info(f"Loading auxiliary model from checkpoint path {aux_path}")
        aux_ckpt = torch.load(aux_path, weights_only=True)["state_dict"]
        aux_ckpt = {
            k.replace("net.", ""): v
            for k, v in aux_ckpt.items()
            if "fc" not in k
        }
        incompatible_keys = model.net.attention.attention.load_state_dict(
            aux_ckpt,
            strict=False,
        )
        # check if the model is in eval mode and has no gradients
        log.info(
            f"Aux. model in eval mode: {not model.net.attention.attention.training}"
        )
        log.info(
            f"Aux. model has no gradients: {not any(p.requires_grad for p in model.net.attention.attention.parameters())}"
        )
        # get compatible keys
        concrete_keys = list(
            set(model.net.attention.attention.state_dict().keys())
            - set(incompatible_keys[0])
        )
        display_keys = (
            (
                "Compatible keys: "
                + " ".join(concrete_keys[:10])
                + f"... {len(concrete_keys) - 10} more items remaining"
                if len(concrete_keys) > 10
                else concrete_keys
            )
            if len(incompatible_keys[0]) > 10
            else f"Incompatible keys: {incompatible_keys[0]}"
        )
        log.info(display_keys)
        log.info(f"Unexpected keys: {incompatible_keys[1]}")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    hydra.utils.instantiate(cfg.matmul)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        trainable = log_hyperparameters(object_dict)
        if not trainable:
            log.warning("Exiting training due to non-trainable model.")
            return {}, object_dict

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
