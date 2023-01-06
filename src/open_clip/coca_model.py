from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .generation_utils import top_a, top_k, top_p, prepare_inputs_for_generation
from transformers import BeamSearchScorer, LogitsProcessorList, HammingDiversityLogitsProcessor, \
                            MinLengthLogitsProcessor,StoppingCriteriaList
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.pytorch_utils import torch_int_div
import os
import gc
import warnings
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Dict

@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    latent_dim: int = 512

class CoCaEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.encoder.set_grad_checkpointing(enable)
        self.decoder.set_grad_checkpointing(enable)

def _build_encoder_decoder_tower(
    embed_dim,
    multimodal_cfg,
    text_cfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):

    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        
    encoder = _build_text_tower(
        multimodal_cfg.latent_dim, 
        text_cfg=text_cfg, 
        quick_gelu=quick_gelu, 
        cast_dtype=cast_dtype
    )
    
    vocab_size = (
        encoder.config.vocab_size # for hf models
        if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
        else multimodal_cfg.vocab_size
    )

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    
    return CoCaEncoderDecoder(encoder, decoder), multimodal_cfg, vocab_size
 
class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )

        self.text, multimodal_cfg, vocab_size = _build_encoder_decoder_tower(
            embed_dim, multimodal_cfg, text_cfg, quick_gelu, cast_dtype
        )
        self.visual = _build_vision_tower(
            multimodal_cfg.latent_dim, vision_cfg, quick_gelu, cast_dtype
        )

        self.to_logits = nn.Sequential(
            norm_layer(multimodal_cfg.width), nn.Linear(multimodal_cfg.width, vocab_size, bias=False)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, images, normalize=True, return_tokens=False):
        image_latent, tokens_embs = self.visual(images, output_tokens=True)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return (image_latent, tokens_embs) if return_tokens else image_latent

    def encode_text(self, text, normalize=True, return_tokens=False):
        text = text[:, :-1] # make space for CLS token
        text_latent, token_emb = self.text.encoder(text, output_tokens=True)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return (text_latent, token_emb) if return_tokens else text_latent

    def forward(self, image, text, output_dict=False, image_latent=None, image_embs=None, past_key_values=None):

        text_latent, token_embs = self.encode_text(text, return_tokens=True)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self.encode_image(image, return_tokens=True)

        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]
        
        token_embs = self.text.decoder(image_embs, token_embs)
        logits = self.to_logits(token_embs)
        if output_dict:
            return {
                "image_features": image_latent,
                "text_features": text_latent,
                "logits": logits,
                "labels": labels,
                "logit_scale": self.logit_scale.exp(),
                "past_key_values": past_key_values
            }

        return image_latent, text_latent, logits, labels, self.logit_scale.exp()

    def generate(
        self,
        image,
        text,
        seq_len,
        max_seq_len=77,
        mask_prob = 0.0,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        min_p_pow = 2.0,
        min_p_ratio = 0.02,
        ):

        assert mask_prob < 1, "mask_prob must be smaller than 1."

        was_training = self.training
        num_dims = len(text.shape)

        if num_dims == 1:
            text = text[None, :]

        _, t = text.shape
        self.eval()
        out = text
        image_latent, image_embs = self.encode_image(image, return_tokens=True)

        for _ in range(seq_len):
            x = out[:, -max_seq_len:]

            # TODO: adjust for dict output
            logits = self(image, x, image_latent=image_latent, image_embs=image_embs)[2][:, -1]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is top_a:
                filtered_logits = filter_logits_fn(
                    logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                )
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.train(was_training)
        return out

# TODO: check it works as usual
# modified version of https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/generation_utils.py
    def _update_model_kwargs_for_generation(self,
            outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs["past_key_values"]
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.memes
        elif "past_buckets_states" in outpus:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        return model_kwargs

    def generate_beamseach(
        self,
        image_inputs,
        max_length=None,
        pad_token_id=0,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        output_scores=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=True,
        synced_gpus=False,
        **kwargs,
        ):
        device = image_inputs.device
        image_inputs = image_inputs.repeat(num_beams, 1, 1, 1)
        image_latent, image_embs = self.encode_image(image_inputs, return_token=True)

        input_ids = torch.ones((num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        target_logits_processor_list = [
            MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)
        ]
        logits_processor = LogitsProcessorList(
            target_logits_processor_list
        )
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        # TODO: where it gets config
        pad_token_id = pad_token_id if pad_token_id is not None else self.text_cfg.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.text_cfg.pad_token_id

        # in HF reads output_scores from config when it is None
        output_scores = output_scores if output_scores is not None else False
        # in HF reads output_attention from config when it is None
        output_attention = output_attentions if output_attentions is not None else False
        # in HF reads return_dict_in_generate from config when it is None
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else False
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape

        if return_dict_in_generate and output_scores:
            beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
        else:
            beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and model.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False # used by synced_gpus only
        model_kwargs = {}
        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    break

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            # logits = self(image, x, attn_mask=self.attn_mask[:x_seq_len, :x_seq_len])[2][:, -1]

            outputs = self(model_inputs['images'],
                           model_inputs['text'],
                           attn_mask[:cur_len, :cur_len],
                           image_latent=image_latent,
                           image_embs=image_embs,
                           output_dict=True)

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            if output_scores:
                processed_score = torch.zeros_like(outputs['logits'][:, -1, :])

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]

                # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                # next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )# (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                if output_scores:
                    processed_score[batch_group_indices] = next_token_scores_processed

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch_int_div(next_tokens, vocab_size)
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                if return_dict_in_generate and output_scores:
                    beam_indices[beam_group_idx] = tuple(
                        beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                    )
                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch_int_div(beam_idx, group_size) + group_start_idx + (beam_idx % group_size)
                )

            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_score,)
                # TODO : deal with when output_attentions in next time
                # if output_attentions:
                #     decoder_attentions += (
                #         (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions, )
                #     )
                #     if model.config.is_encoder_decoder:
                #         cross_attention += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )
            # TODO: support it in next step
            # if model_kwargs["past"] is not None:
            #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], reordering_indices)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences'][0]