import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from ax.service.ax_client import AxClient
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import torch

from ptame.utils import (  # noqa: E402
    RankedLogger,
    ax_wrapper,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


@ax_wrapper
def evaluation_function(
    cfg: DictConfig, **kwargs: Any
) -> Tuple[Dict[str, Any], bool]:
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
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, test=False
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if aux_path := cfg.get("aux_ckpt_path"):
        log.info(f"Loading auxiliary model from checkpoint path {aux_path}")
        aux_ckpt = torch.load(aux_path, weights_only=True)["state_dict"]
        aux_ckpt = {k.replace("net.", ""): v for k, v in aux_ckpt.items()}
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
                + concrete_keys[:10]
                + [f"... {len(concrete_keys) - 10} more items remaining"]
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
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    metric_dict = {}
    if trainer.is_last_batch:
        # evaluate the model with validation set
        trainer.test(model=model, datamodule=datamodule)
        metrics = trainer.callback_metrics
        # choose the metrics that are in the objectives_map list
        metric_dict = {
            name: (
                metrics[key].item(),
                None,
            )
            for name, key in cfg.hparams_search.objectives.mapping.items()
        }
    return metric_dict, trainer.is_last_batch


def make_params(params: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Creates a list of dictionaries with the parameters to optimize."""
    parameters = []
    for name, param in params.items():
        param["name"] = name
        parameters.append(param)
    return parameters


def merge_config(cfg, update):
    """Merges the config with the update."""
    if reparam := cfg.hparams_search.search_space.get(
        "composition_constraint", None
    ):
        total = reparam.total
        independent = reparam.independent
        target = reparam.target
        update[target] = total - sum(update[i] for i in independent)
    for key, value in update.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value


@hydra.main(
    version_base="1.3", config_path="configs", config_name="ax_sweep.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for sweeping with ax.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    worst_result = {
        key: tuple(value)
        for key, value in cfg.hparams_search.objectives.worst_result.items()
    }

    objectives = hydra.utils.instantiate(cfg.hparams_search.objectives.config)
    parameters = make_params(
        OmegaConf.to_container(
            hydra.utils.instantiate(cfg.hparams_search.search_space.params)
        )
    )
    status_quo = cfg.hparams_search.search_space.get("status_quo", None)
    parameter_constraints = cfg.hparams_search.search_space.get(
        "parameter_constraints", []
    )
    get_status_quo = False
    if cfg.get("resume_ax"):
        ax_client = AxClient.load_from_json_file(cfg.resume_ax)
        if (
            not (
                ax_client.get_trials_data_frame()["arm_name"] == "status_quo"
            ).any()
            and status_quo
        ):
            if ax_client.experiment.status_quo is None:
                ax_client.set_status_quo(status_quo)
            params, trial_index = ax_client.attach_trial(
                status_quo, arm_name="status_quo"
            )
            get_status_quo = True
        ax_client.set_optimization_config(objectives)
        ax_client.set_search_space(
            parameters, parameter_constraints=parameter_constraints
        )
    else:
        ax_client = AxClient(random_seed=cfg.get("seed"))
        ax_client.create_experiment(
            name=cfg.hparams_search.get("name"),
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            status_quo=status_quo,
            objectives=objectives,
            choose_generation_strategy_kwargs={"max_initialization_trials": 5},
            immutable_search_space_and_opt_config=False,
        )
        if status_quo:
            get_status_quo = True
            params, trial_index = ax_client.attach_trial(
                status_quo, arm_name="status_quo"
            )
    output_dir = cfg.paths.output_dir
    for i in range(cfg.hparams_search.get("max_trials")):
        if get_status_quo:
            get_status_quo = False
        else:
            params, trial_index = ax_client.get_next_trial()
        cfg.paths.output_dir = output_dir + f"/trial_{trial_index}"
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        merge_config(cfg, params)
        # train the model
        if cfg.get("logger") and cfg.logger.get("wandb"):
            cfg.logger.wandb.name = f"trial_{trial_index}"
        metric_dict, status = evaluation_function(
            cfg, trial=(trial_index, ax_client), output_dir=output_dir
        )
        if status:
            ax_client.complete_trial(trial_index, metric_dict)
        else:
            ax_client.stop_trial_early(trial_index)
            ax_client.update_trial_data(trial_index, worst_result)
        # save client
        ax_client.save_to_json_file(output_dir + "/ax_client.json")


if __name__ == "__main__":
    main()
