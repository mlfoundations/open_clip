import sys
import numpy as np
import torch
import torch.nn as nn
from mteb import MTEB
from training.params import parse_args
from tqdm.autonotebook import trange
import open_clip
from open_clip.factory import get_tokenizer

class CLANP_For_MTEB(nn.Module):

    def __init__(self,args):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, jit=False, model_type='SiameseCLANP')
        self.tokenizer = get_tokenizer(args.model)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def encode(self, sentences, batch_size=128, show_progress_bar=True, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """


        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index:start_index+batch_size]
            features = self.tokenizer(sentences_batch)
            features = features.to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_text(features)
            
            all_embeddings.append(embeddings.detach())

        all_embeddings = torch.cat(all_embeddings,0)

        return all_embeddings




def run_mteb(args):

    model = CLANP_For_MTEB(args)
    evaluation = MTEB(tasks=['MSMARCO']) #task_types=['Clustering','retrieval']
    results = evaluation.run(model, eval_splits=["test"], output_folder=args.mteb_output)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    run_mteb(args)

    # /fsx/home-jianz/logs/2023_05_11-16_04_50-model_Siamese-pythia-410m-lr_0.0001-b_1024-j_6-p_amp_bfloat16/checkpoints/epoch_1.pt
    # srun --account=laion --partition=g40 nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --job-name=MTEB --pty bash -i
    # srun --account laion --partition=g40 --gpus=1 --cpus-per-gpu=4 --job-name=pty --pty bash -i
    # python3 src/training/mteb_eval.py --model="Siamese-xlm-roberta-large" --pretrained /fsx/home-jianz/logs/2023_05_25-09_21_35-model_Siamese-xlm-roberta-large-lr_1e-05-b_256-j_10-p_amp_bfloat16/checkpoints/epoch_latest.pt --mteb-output /fsx/home-jianz/logs/mteb --context-length=128
    