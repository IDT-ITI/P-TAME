from typing import Any, Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig

from ptame.models.ptame_module import PTAMELitModule  # noqa: E402
from ptame.utils import RankedLogger, extras  # noqa: E402
from ptame.utils.map_printer import SampleSet  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def print_map(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prints maps for the given model and subset."""

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    sampleset: SampleSet = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if not cfg.get("cam"):
        model: PTAMELitModule = hydra.utils.instantiate(cfg.model)
        net = model.net

        if ckpt_path := cfg.get("ckpt_path"):
            log.info(f"Loading model from checkpoint path {ckpt_path}")
            loaded_sd = torch.load(ckpt_path, weights_only=True)["state_dict"]
            if any("_orig_mod" in k for k in loaded_sd.keys()):
                log.info("Loading original model weights")
                keys = list(loaded_sd.keys())
                for key in keys:
                    if "_orig_mod" in key:
                        loaded_sd[key.replace("_orig_mod.", "")] = (
                            loaded_sd.pop(key)
                        )
            incompatible_keys = model.load_state_dict(
                loaded_sd,
                strict=False,
            )
            log.info(
                "Incompatible keys: ",
                ", ".join(a),
            ) if len(
                a := [i for i in incompatible_keys[0] if "backbone" not in i]
            ) else log.info("No incompatible keys found in model state dict")
            log.info("Unexpected keys: ", incompatible_keys[1]) if len(
                incompatible_keys[1]
            ) > 0 else log.info("No unexpected keys found in model state dict")

        if aux_path := cfg.get("aux_ckpt_path"):
            log.info(
                f"Loading auxiliary model from checkpoint path {aux_path}"
            )
            aux_ckpt = torch.load(aux_path, weights_only=True)["state_dict"]
            aux_ckpt = {
                k.replace("net.", ""): v
                for k, v in aux_ckpt.items()
                if "fc" not in k
            }
            incompatible_keys = net.attention.attention.load_state_dict(
                aux_ckpt,
                strict=False,
            )
            # check if the model is in eval mode and has no gradients
            log.info(
                f"Aux. model in eval mode: {not net.attention.attention.training}"
            )
            log.info(
                f"Aux. model has no gradients: {not any(p.requires_grad for p in net.attention.attention.parameters())}"
            )
            # get compatible keys
            concrete_keys = list(
                set(net.attention.attention.state_dict().keys())
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
    else:
        net = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating map printer <{cfg.map_printer._target_}>")
    # map_printer = hydra.utils.instantiate()
    map_printer = hydra.utils.instantiate(
        cfg.map_printer, model=net.eval(), sampleset=sampleset
    )

    log.info("Starting printing maps!")
    try:
        map_printer.print_maps()
        log.info("Success!")
    except Exception as e:
        log.error("Failed to print maps!")
        raise e


@hydra.main(
    version_base="1.3", config_path="configs", config_name="print_map.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for printing maps.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)

    extras(cfg)

    print_map(cfg)


if __name__ == "__main__":
    main()
