from torch import optim

def get_num_layer_for_transformer(var_name, num_max_layer):
    first_layer_var_names = [
        "cls_token", 
        "mask_token",
        "pos_embed", 
        "positional_embedding",
        "token_embedding",
        "patch_embed",
        "conv1", # name of patch embedding in VisionTransformer
        "transformer.embeddings.word_embeddings",
        "transformer.embeddings.position_embeddings",
        "transformer.embeddings.token_type_embeddings"
    ]
    for first_layer_var_name in first_layer_var_names:
        if var_name == first_layer_var_name or var_name.startswith(first_layer_var_name):
            return 0

    if var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith("transformer.resblocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name.startswith("transformer.resblocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name.startswith("transformer.encoder.layer"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))

def get_parameters(args, model, assigner, tower):
    filter_parameters = []
    if tower == 'visual':
        lr = args.visual_lr if args.visual_lr is not None else args.lr
        weight_decay = args.visual_wd if args.visual_wd is not None else args.wd
        if hasattr(model, 'visual'):
            filter_parameters = model.visual.named_parameters()
    elif tower == 'text':
        lr = args.text_lr if args.text_lr is not None else args.lr
        weight_decay = args.text_wd if args.text_wd is not None else args.wd
        if hasattr(model, 'text'):
            filter_parameters = model.text.named_parameters()
    else:
        lr = args.lr
        weight_decay = args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'visual.' not in name and 'text.' not in name]

    get_num_layer  = assigner.get_layer_id if assigner is not None else None
    get_layer_scale = assigner.get_scale if assigner is not None else None


    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in filter_parameters:
        if not param.requires_grad:
            continue

        if param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or 'logit_scale' in name:
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

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }
            parameter_group_vars[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())

def get_all_parameters(args, model, assigner_visual, assigner_text):
    parameters = []
    visual_parameters = get_parameters(args, model, assigner_visual, 'visual')
    text_parameters = get_parameters(args, model, assigner_text, 'text')
    other_parameters = get_parameters(args, model, None, 'other')

    parameters.extend(visual_parameters)
    parameters.extend(text_parameters)
    parameters.extend(other_parameters)

    if len(parameters) == 0:
        parameters = model.parameters()
    return parameters

def create_optimizer(args, model, assigner_visual, assigner_text):
    parameters = get_all_parameters(args, model, assigner_visual, assigner_text)

    optimizer_args = dict(
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    optimizer = optim.AdamW(parameters, **optimizer_args)
    return optimizer