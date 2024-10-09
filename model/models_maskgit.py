from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import LlamaConfig
from transformers.utils import ModelOutput

from data.datasets import schedule_maskgit_ratio
from data.pdb_tokenizer import PDBTokenizer
from .maskgit_modules import MaskGITModel, MaskGITPreTrainedModel

# Confidence score for known tokens to avoid masking or repredicting them.
CONFIDENCE_OF_KNOWN_TOKENS = float('inf')


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    """
    Args:
      mask_len: the number of tokens to mask.
      probs: the probabilities associated with each entry.
      temperature: controls the randomness of masking.

    Returns:
      A binary masking map [batch_size, seq_len].
    """
    # Add Gumbel noise for randomization
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs)))
    confidence = torch.log(probs) + temperature * gumbel_noise
    # Sort the confidence scores
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    # Determine the cut-off threshold for masking
    cut_off = torch.gather(sorted_confidence, dim=-1, index=mask_len.long())
    # Generate the masking map
    masking = (confidence < cut_off)
    return masking


class VQAE(nn.Module):
    def __init__(self):
        super(VQAE, self).__init__()
        self.special_code = nn.Embedding(300, 128)

    def code2vec(self, code, max_code=255):
        special_embed = self.special_code(code)
        vec = special_embed / (special_embed.norm(dim=-1, keepdim=True) + 1e-6)
        return vec


MAX_CAHIN = 1000


@dataclass
class CustomMaskGITOutputWithPast(ModelOutput):
    def __init__(self,
                 loss: Optional[torch.FloatTensor] = None,
                 logits: Optional[torch.FloatTensor] = None,
                 logits_pos: Optional[torch.FloatTensor] = None,
                 logits_chain: Optional[torch.FloatTensor] = None,
                 hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
                 attentions: Optional[Tuple[torch.FloatTensor]] = None):
        self.loss = loss
        self.logits = logits
        self.logits_pos = logits_pos
        self.logits_chain = logits_chain
        self.hidden_states = hidden_states
        self.attentions = attentions


class MaskGIT(MaskGITPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vqae = VQAE()
        self.enc_model = MaskGITModel(config)  # 🔍
        self.lm_head = nn.Linear(config.hidden_size, 128, bias=False)
        self.vqid_encoding = nn.Linear(128, config.hidden_size)
        self.tokenizer = PDBTokenizer()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CustomMaskGITOutputWithPast]:
        key_embeds = self.vqid_encoding(self.vqae.code2vec(input_ids))
        input_ids = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.enc_model(  # 🔍
            input_ids=input_ids,
            inputs_embeds=key_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        centers = self.vqae.code2vec(torch.arange(269, device=logits.device)[None])[0]
        logits = torch.einsum('bld,kd->blk', logits, centers)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CustomMaskGITOutputWithPast(
            loss=loss,
            logits=logits,
            # logits_pos = logits_pos,
            # logits_chain = logits_chain,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
            self,
            inputs_ids,
            mask_token_id=-1,
            num_iter=12,
            start_iter=0,
            temperature=4.5,
            mask_scheduling_method="cosine",
            **kwargs,
    ):
        """🔍 Adapted from https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py
        Fast decoding for iterative generation.

        Args:
          inputs_ids: Tensor [batch_size, seq_length] input sequence of masked tokens.
          mask_token_id: Integer representing the mask token ID.
          num_iter: Number of decoding iterations (default is 12).
          start_iter: Starting iteration index (default is 0).
          emperature: Controls the randomness of masking.
          mask_scheduling_method: Method for scheduling masking.

        Returns:
          Tensor [num_iter + 1, batch_size, seq_length] output sequences for all iterations.
        """
        device = inputs_ids.device
        inputs_ids = inputs_ids.to(torch.int64)

        # Calculate the initial number of unknown (masked) tokens
        unknown_number_in_the_beginning = torch.sum(inputs_ids == mask_token_id, dim=-1)

        # Initialize the decoding state
        cur_ids = inputs_ids  # Current sequence of tokens
        final_ids = inputs_ids.unsqueeze(0).repeat(start_iter + 1, 1, 1)  # Final sequence after decoding, [num_iters, batch_size, seq_len].

        # Run the decoding loop
        for step in range(start_iter, num_iter):
            step_ratio = (step + 1) / num_iter

            """calculate the result of this round"""
            # Sample next tokens based on logits
            outputs = self.forward(cur_ids, **kwargs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)  # [batch_size, seq_length, num_classes]
            sampled_ids = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])  # [batch_size, seq_length]

            # Update only the masked tokens
            unknown_map = (cur_ids == mask_token_id)
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)

            """randomly mask tokens according to confidence"""
            # Calculate the mask ratio for the next round
            mask_ratio = schedule_maskgit_ratio(unknown_number_in_the_beginning, ratio=step_ratio, method=mask_scheduling_method)
            # Update the final sequences with the current sampled_ids
            final_ids = torch.cat([final_ids, sampled_ids.unsqueeze(0)], dim=0)

            # Get the probabilities for the selected tokens
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1).long()).squeeze(-1)
            # Assign high confidence to known tokens
            selected_probs = torch.where(unknown_map, selected_probs, torch.tensor(CONFIDENCE_OF_KNOWN_TOKENS))

            # Determine the number of tokens to mask in the next iteration
            mask_len = torch.maximum(
                torch.tensor(1, device=device),
                torch.minimum(
                    torch.sum(unknown_map, dim=-1, keepdim=True) - 1,  # 本轮输入的实际mask数量
                    torch.floor(unknown_number_in_the_beginning * mask_ratio).unsqueeze(1)  # 迭代到本轮时，根据最初输入的mask比率，计算出本轮应有的mask数量
                )  # 两者取其小，保证mask数量不会随着迭代而意外增加
            )

            # Add noise for randomness and generate the masking map
            masking = mask_by_random_topk(mask_len, selected_probs, temperature * (1. - step_ratio))
            # Apply the masking to the current sequence
            sampled_ids = torch.where(masking, torch.tensor(mask_token_id), sampled_ids)

            """update for the next iteration"""
            cur_ids = sampled_ids

        return final_ids

    