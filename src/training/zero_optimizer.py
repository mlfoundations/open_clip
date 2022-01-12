import torch
from torch.optim import AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer


            # optimizer = optim.AdamW(
            #     [
            #         {"params": gain_or_bias_params, "weight_decay": 0.},
            #         {"params": rest_params, "weight_decay": args.wd},
            #     ],
            #     lr=args.lr,
            #     betas=(args.beta1, args.beta2),
            #     eps=args.eps,
            # )

class _AdamW(AdamW):
    def __init__(self,
                 params,
                 gain_or_bias_params,
                 rest_params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 amsgrad=False,
                 args=None):
        params_to_pass = [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ] 
        super(_AdamW, self).__init__(
            params_to_pass,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

def createZeroRedundancyAdamW(gain_or_bias_params, rest_params, args):
    return ZeroRedundancyOptimizer(gain_or_bias_params+rest_params,
                                   _AdamW,
                                   gain_or_bias_params=gain_or_bias_params,
                                   rest_params=rest_params,
                                   lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   eps=args.eps,
                                   args=args)
