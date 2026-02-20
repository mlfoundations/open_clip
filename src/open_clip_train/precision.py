import torch
from contextlib import suppress
from functools import partial


def get_autocast(precision, device_type='cuda', fsdp=False):
    if fsdp:
        return suppress

    if precision =='amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        amp_dtype = torch.bfloat16
    else:
        return suppress

    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)