import time
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from ax.service.ax_client import AxClient
from omegaconf import DictConfig

from ptame.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info(
            "Disabling python warnings! <cfg.extras.ignore_warnings=True>"
        )
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info(
            "Printing config tree with Rich! <cfg.extras.print_config=True>"
        )
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the
    task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        exit_code = 1
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
            exit_code = 0
        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")
            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish(exit_code=exit_code)

        return metric_dict, object_dict

    return wrap


def ax_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the
    task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(
        cfg: DictConfig, trial: Tuple[int, AxClient], output_dir="./"
    ) -> Tuple[Dict[str, Any], bool]:
        metric_dict = {}
        status = False
        # execute the task
        try:
            metric_dict, status = task_func(cfg=cfg)

        except (KeyboardInterrupt, RuntimeError) as ex:
            log.exception(
                f"Prematurely killed at time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            # save experiment
            trial[1].log_trial_failure(trial[0])
            for i in range(5):
                try:
                    trial[1].save_to_json_file(
                        f"{output_dir}/killed_experiment.json"
                    )
                    break
                except Exception as ex:
                    if i == 4:
                        log.exception(
                            f"Could not save experiment to json file! <{type(ex).__name__}>"
                        )

            # the above needs some time to save the file, but it is not blocking
            # so we need to wait a bit before exiting
            time.sleep(5)
            raise ex
        # things to do if exception occurs
        except Exception as ex:
            log.exception(f"Exception occurred! type: <{type(ex).__name__}>")
            # save exception to `.log` file
            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    exit_code = 0 if status else 1
                    wandb.finish(exit_code=exit_code)

        return metric_dict, status

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
