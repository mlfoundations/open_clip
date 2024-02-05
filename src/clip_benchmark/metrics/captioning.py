from typing import Union

import torch.utils.data
from open_clip.tokenizer import _tokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tqdm.auto import tqdm

"""
Code adapted from https://github.com/salaniz/pycocoevalcap/blob/master/eval.py
Thanks to @salaniz for the code!
"""


class COCOEvalCap:
    def __init__(self, results):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.results = results

    def evaluate(self):
        gts = {}
        res = {}
        for imgId, r in enumerate(self.results):
            gts[imgId] = r['true']
            res[imgId] = r['gen']
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
            (Spice(), 'SPICE'),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) is list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_img_to_eval_imgs(scs, gts.keys(), m)
                    print('%s: %0.3f' % (m, sc))
            else:
                self.set_eval(score, method)
                self.set_img_to_eval_imgs(scores, gts.keys(), method)
                print('%s: %0.3f' % (method, score))
        self.set_eval_imgs()

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, imgids, method):
        for imgId, score in zip(imgids, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]['image_id'] = imgId
            self.imgToEval[imgId][method] = score

    def set_eval_imgs(self):
        self.evalImgs = [_eval for _, _eval in self.imgToEval.items()]


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
):
    results = []
    image_id = 0
    for idx, (img, captions) in enumerate(tqdm(dataloader)):
        out = model.generate(img.to(device))
        decoded = [
            _tokenizer.decode(i)
            .split('<end_of_text>')[0]
            .replace('<start_of_text>', '')
            .strip()
            for i in out.cpu().numpy()
        ]
        for pred, true in zip(decoded, captions):
            true = [{'caption': t} for t in true]
            pred = [{'caption': pred}]
            results.append({'image_id': image_id, 'gen': pred, 'true': true})
            image_id += 1

    coco_eval = COCOEvalCap(results)
    coco_eval.evaluate()
    metrics = coco_eval.eval

    for metric, score in metrics.items():
        print(f'{metric}: {score:.3f}')
    return metrics
