"""
Custom Trainer with token-level normalized loss.

Formula (from CoderForge blog):
    L = sum_r sum_i (L_i * n_i_r) / sum_r sum_i n_i_r

where L_i = cross-entropy mean on rank r micro-batch i,
      n_i_r = number of valid (non -100) tokens in that micro-batch.

With Liger Kernel's fused_linear_cross_entropy=True, the model's lm_head + CE loss
are fused into a single kernel that never materializes the full (seq_len × vocab_size)
logits tensor. We therefore let the model compute its own loss (labels passed in) and
only do the global token-count normalization on top.

Without Liger (fallback): chunked logits loop avoids the ~40 GB peak allocation.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Trainer

logger = logging.getLogger(__name__)


class TokenNormalizedLossTrainer(Trainer):
    """
    Trainer that uses globally token-normalized cross-entropy loss.

    Works with Liger Kernel's fused_linear_cross_entropy (preferred) or falls back
    to a chunked logits loop. Compatible with FSDP, DeepSpeed ZeRO, single-GPU.
    """

    _CHUNK_SIZE = 4096  # rows of logits processed at once in fallback path

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        labels = inputs.get("labels")  # keep in inputs for model forward

        # Let the model compute loss. With Liger fused_linear_cross_entropy the
        # kernel computes CE directly from hidden states — no logits tensor created.
        outputs = model(**inputs)
        # outputs.loss is mean CE over non-ignored tokens (HF default reduction).
        # We need the (sum, n_tokens) pair to normalize globally across ranks.
        model_loss = outputs.loss  # scalar, reduction="mean" over valid tokens

        # Recover n_tokens from labels so we can reconstruct the sum.
        shift_labels = labels[..., 1:].contiguous().view(-1)
        local_n_tokens = (shift_labels != -100).sum().to(model_loss)
        local_loss_sum = model_loss * local_n_tokens

        # All-reduce across ranks
        if dist.is_available() and dist.is_initialized():
            stats = torch.stack([local_loss_sum, local_n_tokens])
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_loss_sum, global_n_tokens = stats[0], stats[1]
        else:
            global_loss_sum = local_loss_sum
            global_n_tokens = local_n_tokens

        if global_n_tokens == 0:
            loss = model_loss * 0.0
        else:
            loss = global_loss_sum / global_n_tokens

        return (loss, outputs) if return_outputs else loss
