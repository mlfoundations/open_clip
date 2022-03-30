# Code ported from https://github.com/openai/CLIP
import os
import warnings
from typing import Union, List

import torch

from .model import build_model
from .pretrained import get_pretrained_url, list_pretrained_tag_models, download_pretrained
from .transform import image_transform

__all__ = ["available_openai_models", "load_openai"]


def available_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models('openai')


def load_openai(
        name: str,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit=True,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained(get_pretrained_url(name, 'openai'))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_openai_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model(state_dict or model.state_dict()).to(device)
        except KeyError:
            sd = {k[7:]: v for k,v in state_dict["state_dict"].items()}
            model = build_model(sd).to(device)

        if str(device) == "cpu":
            model.float()
        return model, \
            image_transform(model.visual.image_size, is_train=True), \
            image_transform(model.visual.image_size, is_train=False)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    return model, \
        image_transform(model.image_size.item(), is_train=True), \
        image_transform(model.image_size.item(), is_train=False)
