import torch
import torch.nn.functional as F
import open_clip
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np

import logging

from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast

def run(model,dataloader,args):
    
        preds = []
        gt_scores = []

        for sent1,sent2,scores in tqdm(dataloader, unit_scale=args.batch_size):
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                a_features = model.encode_text(sent1.to(args.device))
                b_features = model.encode_text(sent2.to(args.device))

                pred = F.cosine_similarity(a_features,b_features,dim=-1).squeeze()

                preds.append(pred.cpu().detach().numpy())
                gt_scores.append(scores)

        preds = np.concatenate(preds,axis=0)
        gt_scores = np.concatenate(gt_scores,axis=0)

        return spearmanr(preds,gt_scores)[0], pearsonr(preds,gt_scores)[0]


def sts_eval(model, data, epoch, args):
    
    if 'sts-val' not in data:
        return {}

    logging.info('Starting STS-17 evaluation.')

    
    results = {}

    spearman_corr, pearson_corr = run(model,data['sts-val'].dataloader, args)
    results['sts-val-spearman-corr'] = spearman_corr
    results['sts-val-pearson-corr'] = pearson_corr

    logging.info('Finished STS-17')

    return results


    

