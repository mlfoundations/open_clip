"""
Adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
"""
import logging
from torch import optim


def get_num_layer_for_transformer(var_name, num_max_layer):
    first_layer_var_names = [
        "visual.cls_token", 
        "visual.mask_token", 
        "visual.pos_embed", 
        "visual.positional_embedding", 
        "visual.patch_embed",
        "visual.conv1", # name of patch embed in CLIP
        "text.pos_embed", 
        "text.positional_embedding",
        "text.token_embedding",
        "text.transformer.embeddings.word_embeddings",
        "text.transformer.embeddings.position_embeddings",
        "text.transformer.embeddings.token_type_embeddings"
    ]
    for first_layer_var_name in first_layer_var_names:
        if var_name.startswith(first_layer_var_name):
            return 0
    if var_name.startswith("visual.rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("visual.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name.startswith("visual.transformer.resblocks"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    elif var_name.startswith("text.transformer.resblocks"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    elif var_name.startswith("text.transformer.encoder.layer"):
        layer_id = int(var_name.split('.')[4])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, layer_decay, num_layers):
        self.values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))

def param_groups_layer_decay(model_params, lr, weight_decay, lr_scale_assigner, tower):
    get_num_layer  = lr_scale_assigner.get_layer_id if lr_scale_assigner else None
    get_layer_scale = lr_scale_assigner.get_scale if lr_scale_assigner else None

    param_group_names = {} # NOTE for debugging
    param_group_vars = {}
    for name, param in model_params:
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or 'logit_scale' in name:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = tower + "_" + "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in param_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            param_group_names[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }
            param_group_vars[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }

        param_group_vars[group_name]["params"].append(param)
        param_group_names[group_name]["params"].append(name)
    return list(param_group_vars.values())

def get_all_param_groups(args, model, lr_scale_assigner_visual, lr_scale_assigner_text):
    visual_params, text_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if name.startswith('visual.'):
            visual_params.append([name, param])
        elif name.startswith('text.'):
            text_params.append([name, param])
        else:
            other_params.append([name, param])

    optim_params = []
    visual_optim_params = param_groups_layer_decay(
        visual_params, 
        args.visual_lr or args.lr, 
        args.visual_wd or args.wd, 
        lr_scale_assigner_visual, 
        'visual'
    )
    text_optim_params = param_groups_layer_decay(
        text_params, 
        args.text_lr or args.lr, 
        args.text_wd or args.wd, 
        lr_scale_assigner_text, 
        'text'
    )
    other_optim_params = param_groups_layer_decay(
        other_params, 
        args.lr, 
        args.wd, 
        None, 
        'other'
    )
    
    optim_params.extend(visual_optim_params)
    optim_params.extend(text_optim_params)
    optim_params.extend(other_optim_params)

    if len(optim_params) == 0:
        optim_params = model.parameters()
    return optim_params

def create_optimizer(args, model):
    lr_scale_assigner_visual, lr_scale_assigner_text = None, None
    
    if args.visual_ld and "vit" in args.model.lower():
        visual_num_layers = model.visual.get_num_layers()
        lr_scale_assigner_visual = LayerDecayValueAssigner(args.visual_ld, visual_num_layers)
        logging.info("Assigned visual lr scale values = %s" % str(lr_scale_assigner_visual.values))

    if args.text_ld:
        text_num_layers = model.text.get_num_layers()
        lr_scale_assigner_text = LayerDecayValueAssigner(args.text_ld, text_num_layers)
        logging.info("Assigned text lr scale values = %s" % str(lr_scale_assigner_text.values))

    optim_params = get_all_param_groups(args, model, lr_scale_assigner_visual, lr_scale_assigner_text)

    optim_args = dict(
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    optimizer = optim.AdamW(optim_params, **optim_args)
    return optimizer