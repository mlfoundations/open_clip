from torch import nn, optim


def create_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    text_lr_decay: float = 1.0,
    vision_lr_decay: float = 1.0,
):
    is_gain_or_bias = (
        lambda n, p: p.ndim < 2
        or 'bn' in n
        or 'ln' in n
        or 'bias' in n
        or 'logit_scale' in n
    )

    def is_text_module(n: str):
        return n.startswith('text') or n.startswith('transformer')

    def is_vision_module(n: str):
        return n.startswith('visual')

    params = []
    _text_lr = base_lr
    _vision_lr = base_lr

    for name, param in reversed(list(model.named_parameters())):
        if param.requires_grad:
            _weight_decay = 0.0 if is_gain_or_bias(name, param) else weight_decay

            lr = base_lr
            if is_text_module(name):
                lr = _text_lr
                _text_lr *= text_lr_decay
            elif is_vision_module(name):
                lr = _vision_lr
                _vision_lr *= vision_lr_decay

            params.append({'params': param, 'lr': lr, 'weight_decay': _weight_decay})

    return optim.AdamW(params, betas=(beta1, beta2), eps=eps)
