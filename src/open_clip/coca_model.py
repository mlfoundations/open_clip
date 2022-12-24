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
    AttentionalPooler,
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
    dim_latents: int = None


def _build_input_dependent_text_tower(
    embed_dim: int,
    multimodal_cfg: MultimodalCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
    multimodal:bool = True
):

    if not multimodal:
        return _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype
        )

    if isinstance(multimodal_cfg, dict):
        multimodal_cfg = MultimodalCfg(**multimodal_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    text = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return text, multimodal_cfg


class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        n_queries: int = 256,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()


        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )

        text = _build_input_dependent_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype, multimodal=False)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)
        self.context_length = self.positional_embedding.shape[0] - 1

        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        self.visual = _build_vision_tower(
            embed_dim, vision_cfg, quick_gelu, cast_dtype
        )
        self.heads = text_cfg["heads"]

        self.multimodal_decoder, multimodal_cfg = _build_input_dependent_text_tower(
            embed_dim, multimodal_cfg, quick_gelu, cast_dtype
        )

        self.img_attn_pool = AttentionalPooler(
            multimodal_cfg.width, multimodal_cfg.heads, n_queries=n_queries + 1
        )

        self.img_attn_pool_norm = norm_layer(embed_dim)

        self.dim_latents = multimodal_cfg.dim_latents if multimodal_cfg.dim_latents else multimodal_cfg.width
        self.to_text_latent = nn.Linear(embed_dim, self.dim_latents, bias=False)

        self.to_logits = nn.Sequential(
            norm_layer(embed_dim), nn.Linear(embed_dim, self.vocab_size, bias=False)
        )

        # tie embedding weights and projection
        self.to_logits[-1].weight = self.token_embedding.weight

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = 0

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable
        self.multimodal_decoder.grad_checkpointing = enable

    def encode_image(self, images, normalize=True, return_tokens=False):
        x = self.visual(images, output_tokens=True)

        if hasattr(self.visual, "ln_post"):
            x = self.visual.ln_post(x)

        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            x = x @ self.visual.proj

        x = self.img_attn_pool(x, x)
        x = self.img_attn_pool_norm(x)

        image_latent = x[:, 0]
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent

        return (image_latent, x[:, 1:]) if return_tokens else image_latent

    def _repeat(self, t, N):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def _build_cls_mask(self, text, cast_dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(*cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def encode_text(self, text, normalize=True, return_tokens=False):
        text = text[:, :-1] # make space for CLS token
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = torch.cat(
            [
                x + self.positional_embedding[:seq_len, :].to(cast_dtype), 
                self._repeat(self.cls_token + self.positional_embedding[-1, :], x.shape[0])
            ], 
            dim=1
        )
        seq_len += 1 # seq is 1 longer as we added CLS
        attn_mask = self.attn_mask[None, :seq_len, :seq_len].expand(
            text.shape[0] * self.heads, seq_len, seq_len
        )
        cls_mask = self._build_cls_mask(text, cast_dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask + cls_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[torch.arange(x.shape[0]), :] @ self.text_projection

        cls_emb = x[torch.arange(x.shape[0]), -1]
        token_emb = x[torch.arange(x.shape[0]), :-1]

        cls_emb = self.ln_final(cls_emb)
        text_latent = self.to_text_latent(cls_emb)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return (text_latent, token_emb) if return_tokens else text_latent

    def forward(self, image, text):
        labels = text[:, 1:]

        text_latents, text_tokens = self.encode_text(text, return_tokens=True)
        image_latents, image_tokens = self.encode_image(image, return_tokens=True)

        text_tokens = self.multimodal_decoder(image_tokens, text_tokens)
        logits = self.to_logits(text_tokens)

        return image_latents, text_latents, logits, labels, self.logit_scale.exp()

    def forward_return_dict(self, image, text, attn_mask=None):
        text_latents, text_tokens = self.encode_text(text, return_tokens=True)
        image_latents, image_tokens = self.encode_image(image, return_tokens=True)

        text_tokens = self.multimodal_decoder(image_tokens, text_tokens, attn_mask)
        logits = self.to_logits(text_tokens)
        # TODO: fix in future to get past_key_values
        past_key_values = None
        outputs = {'logits': logits,
                   'past_key_values': past_key_values

        }
        return outputs

    def generate(
        self,
        image,
        text,
        seq_len,
        max_seq_len=None,
        mask_prob = 0.0,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        min_p_pow = 2.0,
        min_p_ratio = 0.02,
        ):

        assert mask_prob < 1, "mask_prob must be smaller than 1."

        if max_seq_len is None:
            max_seq_len = self.context_length

        was_training = self.training
        num_dims = len(text.shape)

        if num_dims == 1:
            text = text[None, :]

        _, t = text.shape
        self.eval()
        out = text

        for _ in range(seq_len):
            x = out[:, -max_seq_len:]

            # TODO: adjust for dict output
            logits = self(image, x)[2][:, -1]

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

    def generate_beamsearch(
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
            outputs = self.forward_return_dict(model_inputs['images'], model_inputs['text'], attn_mask[:cur_len, :cur_len])

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