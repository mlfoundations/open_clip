from contextlib import suppress
import logging

import torch

def get_autocast(precision, device='cuda'):
    if precision.startswith('amp'):
        if device == 'mps':
            logging.warning(
                "MPS devices do not support AMP yet: "
                "https://github.com/pytorch/pytorch/issues/88415 "
                "Disabling AMP and falling back to FP32."
            )
            return suppress
        elif device == 'cpu':
            logging.warning(
                "CPU devices have limited AMP support and result in "
                "attn_mask.dtype: float and query.dtype: c10::BFloat16 mismatch. "
                "Disabling AMP and falling back to FP32."
            )
            return suppress
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
