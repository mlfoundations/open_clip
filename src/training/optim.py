from torch import optim
import json
import logging

from training.distributed import is_master

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("visual.cls_token", 
                    "visual.mask_token", 
                    "visual.pos_embed", 
                    "visual.positional_embedding", 
                    "text.pos_embed", 
                    "text.positional_embedding",
                    "text.token_embedding",
                    "text.transformer.embeddings.word_embeddings",
                    "text.transformer.embeddings.position_embeddings",
                    "text.transformer.embeddings.token_type_embeddings"):
        return 0
    elif var_name.startswith("visual.patch_embed"):
        return 0
    elif var_name.startswith("visual.conv1"):
        return 0
    elif var_name.startswith("visual.rel_pos_bias"):
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
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_parameter_groups(args, model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, group='visual', **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    assert (group in ['text', 'visual', 'other']), print('group must be one of visual, text or other')
    if group == 'text':
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'text.' in name]
        skip_list = ['text.'+s for s in skip_list] if skip_list else skip_list
    elif group == 'visual':
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'visual.' in name]
        skip_list = ['visual.'+s for s in skip_list] if skip_list else skip_list
    else:
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'visual.' not in name and 'text.' not in name]

    for name, param in filter_parameters:
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    if is_master(args, local=args.log_local):
                        logging.info(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue

        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = group + "_" + "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None


        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            if group == 'visual':
                lr = args.visual_lr if args.visual_lr is not None else args.lr
            elif group == 'text':
                lr = args.text_lr if args.text_lr is not None else args.lr
            else:
                lr = args.lr

            parameter_group_names[group_name] = {
                "group": group,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }
            parameter_group_vars[group_name] = {
                "group": group,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if is_master(args, local=args.log_local):
        logging.info(f"Group = {group}")
        logging.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())

def get_parameters(args, model, skip_list, assigner, filter_bias_and_bn, group, **kwargs):
    if group == 'visual':
        visual_parameters = []
        visual_weight_decay = args.visual_wd if args.visual_wd else args.wd
        if visual_weight_decay and filter_bias_and_bn:
            skip = set()
            if skip_list is not None:
                skip = skip_list
            if hasattr(model.visual, 'no_weight_decay'):
                skip = set.union(skip, model.visual.no_weight_decay())
            get_num_layer  = assigner.get_layer_id if assigner is not None else None
            get_layer_scale = assigner.get_scale if assigner is not None else None
            visual_parameters = get_parameter_groups(args, model, visual_weight_decay, skip, get_num_layer, get_layer_scale, 'visual', **kwargs)
            if is_master(args, local=args.log_local):
                logging.info(f"Skip weight decay name marked in model.visual: {skip}")
                logging.info(f"len of visual_parameters group: {len(visual_parameters)}")
        return visual_parameters
        
    elif group == 'text':
        text_parameters = []
        text_weight_decay = args.text_wd if args.text_wd else args.wd
        if text_weight_decay  and filter_bias_and_bn:
            skip = set()
            if skip_list is not None:
                skip = skip_list
            if hasattr(model, 'text'):
                if hasattr(model.text, 'no_weight_decay'):
                    skip = set.union(skip, model.text.no_weight_decay())
            get_num_layer  = assigner.get_layer_id if assigner is not None else None
            get_layer_scale = assigner.get_scale if assigner is not None else None
            text_parameters = get_parameter_groups(args, model, text_weight_decay, skip, get_num_layer, get_layer_scale, 'text', **kwargs)
            if is_master(args, local=args.log_local):
                logging.info(f"Skip weight decay name marked in model.text: {skip}")
                logging.info(f"len of text_parameters group: {len(text_parameters)}")
        return text_parameters
    
    else:
        other_parameters = []
        other_weight_decay = args.wd
        if other_weight_decay  and filter_bias_and_bn:
            skip = set()
            if skip_list is not None:
                skip = skip_list
            if hasattr(model, 'no_weight_decay'):
                skip = set.union(skip, model.no_weight_decay())
            get_num_layer  = assigner.get_layer_id if assigner is not None else None
            get_layer_scale = assigner.get_scale if assigner is not None else None
            other_parameters = get_parameter_groups(args, model, other_weight_decay, skip, get_num_layer, get_layer_scale, 'other', **kwargs)
            if is_master(args, local=args.log_local):
                logging.info(f"Skip weight decay name marked in model: {skip}")
                logging.info(f"len of other_parameters group: {len(other_parameters)}")
        return other_parameters

def get_all_parameters(args, model, skip_list, assigner_visual, assigner_text, filter_bias_and_bn=True, **kwargs):
    parameters = []
    visual_parameters = get_parameters(args, model, skip_list, assigner_visual, filter_bias_and_bn, 'visual', **kwargs)
    text_parameters = get_parameters(args, model, skip_list, assigner_text, filter_bias_and_bn, 'text', **kwargs)
    other_parameters = get_parameters(args, model, skip_list, None, filter_bias_and_bn, 'other', **kwargs)

    parameters.extend(visual_parameters)
    parameters.extend(text_parameters)
    parameters.extend(other_parameters)

    if len(parameters) == 0:
        parameters = model.parameters()
    return parameters

def create_optimizer(args, model, skip_list, assigner_visual, assigner_text, filter_bias_and_bn=True, **kwargs):
    parameters = get_all_parameters(args, model, skip_list, assigner_visual, assigner_text, filter_bias_and_bn, **kwargs)

    optimizer_args = dict(
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    if is_master(args, local=args.log_local):
        logging.info(f'Optimizer config: {optimizer_args}')
    optimizer = optim.AdamW(parameters, **optimizer_args)
    return optimizer