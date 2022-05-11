from hydra.utils import instantiate

from model_training.model.utils import load_from_lighting


def load_model(model_config, consts_config, map_location=None):
    model = instantiate(model_config, consts_config=consts_config)
    model_config = model_config["model_config"]
    if model_config.get("load_weights", False):
        load_from_lighting(model, model_config["ckpt_path"], map_location=map_location, strict=False)
    return model



