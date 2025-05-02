import torch


def cascading_random_model(
    model: torch.nn.Module,
    rand_model: torch.nn.Module,
    layer_key: str = "layer1",
    random_before: bool = True,
    random_after: bool = False,
) -> torch.nn.Module:
    """
    Randomize all layers that contain the specified key in their name.
    """
    before = random_before
    encounter = False
    after = random_after
    mdl = model.state_dict()
    rand_mdl = rand_model.state_dict()
    for name in mdl.keys():
        if layer_key in name:
            mdl[name] = rand_mdl[name]
            before = False
            encounter = True
        elif before:
            mdl[name] = rand_mdl[name]
        elif encounter and after:
            mdl[name] = rand_mdl[name]
    model.load_state_dict(mdl)
    return model
