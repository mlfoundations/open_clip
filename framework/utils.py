import yaml
import importlib

module_config_map = {
    "img_enc": "model/config.yaml",
    "text_enc": "model/config.yaml",
    "preprocessor": "preprocessor/config.yaml",
    "dataset": "dataset/config.yaml",
}


def read_config(path: str) -> dict:
    with open(path, 'r') as f:
        contents = f.readlines()
    data = ''
    for c in contents:
        left = c.find('[')
        right = c.rfind(']')
        if left == -1 or right == -1 or right < left:
            data += c
            continue
        c0 = c[:left] + '"' + c[left:right+1] + '"' + c[right+1:]
        data += c0
    return yaml.safe_load(data)


def list_to_dict(args_list):
    result = {}
    for d in args_list:
        result.update(d)
    return result


def merge_config(default_config: dict, custom_config: dict) -> dict:
    return {**default_config, **custom_config}


def instantiate_module(module_name, config: dict, defaults: dict):
    # update module config with class config
    module_config_path = module_config_map[module_name]
    module_config = read_config(module_config_path)[config["class"]]
    module_config["args"] = merge_config(module_config["args"], config["args"])
    module_config["args"] = merge_config(module_config["args"], defaults)

    # instantiate class
    module_path, class_name = module_config["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(module_config["args"])
    return instance