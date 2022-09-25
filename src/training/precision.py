import torch
from contextlib import suppress

# amp_bfloat16 is more stable than amp float16 for clip training
def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16':
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress