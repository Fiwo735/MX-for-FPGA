import math
from typing import Optional, Tuple, Set, List
import torch
import torch.nn as nn
from transformers.models.distilbert.modeling_distilbert import (
    MultiHeadSelfAttention,
    find_pruneable_heads_and_indices,
    prune_linear_layer
)

from .quant_utils import IntQuantizer, MXFPQuantizer, q_reg



class QuantMultiHeadSelfAttention(nn.Module):
    """
    A distilbert attention block with quantization inserted into forward pass.
    """
    def __init__(self, orig_attn: MultiHeadSelfAttention, q_config={}):
        super().__init__()
        self.config = orig_attn.config

        self.n_heads = orig_attn.n_heads
        self.dim = orig_attn.dim
        self.dropout = orig_attn.dropout
        self.is_causal = orig_attn.is_causal

        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = orig_attn.q_lin
        self.k_lin = orig_attn.k_lin
        self.v_lin = orig_attn.v_lin
        self.out_lin = orig_attn.out_lin

        self.pruned_heads: Set[int] = orig_attn.pruned_heads
        self.attention_head_size = orig_attn.attention_head_size

        # Use CLI quantizer configs if available
        self.init_quantizers(q_config)

        # # Ternary Quantization
        # self.k_quantizer = IntQuantizer(bit_w=2, symmetric=True)
        # self.q_quantizer = IntQuantizer(bit_w=2, symmetric=True)
        # self.v_quantizer = IntQuantizer(bit_w=2, symmetric=True)
        # self.p_quantizer = IntQuantizer(bit_w=8, signed=False)

        # # Int Quantization
        # self.k_quantizer = IntQuantizer(bit_w=q_config['man_w'])
        # self.q_quantizer = IntQuantizer(bit_w=q_config['man_w'])
        # self.s_quantizer = IntQuantizer(bit_w=q_config['man_w'])
        # self.v_quantizer = IntQuantizer(bit_w=q_config['man_w'])
        # self.p_quantizer = IntQuantizer(bit_w=8, signed=False)
        # self.o_quantizer = IntQuantizer(bit_w=q_config['man_w'])

        # # MXFP Quantization
        # self.k_quantizer = MXFPQuantizer(**q_config)
        # self.q_quantizer = MXFPQuantizer(**q_config)
        # self.s_quantizer = MXFPQuantizer(**q_config)
        # self.v_quantizer = MXFPQuantizer(**q_config)
        # self.p_quantizer = MXFPQuantizer(**q_config)
        # self.o_quantizer = MXFPQuantizer(**q_config)

    def init_quantizers(self, q_config):
        ''' Make quantizers from CLI config. '''

        if 'k_quantizer' in q_config.keys():
            quant_type = q_config['k_quantizer'].pop('quant')
            self.k_quantizer = q_reg[quant_type](**q_config['k_quantizer'])
        if 'q_quantizer' in q_config.keys():
            quant_type = q_config['q_quantizer'].pop('quant')
            self.q_quantizer = q_reg[quant_type](**q_config['q_quantizer'])
        if 's_quantizer' in q_config.keys():
            quant_type = q_config['s_quantizer'].pop('quant')
            self.s_quantizer = q_reg[quant_type](**q_config['s_quantizer'])
        if 'v_quantizer' in q_config.keys():
            quant_type = q_config['v_quantizer'].pop('quant')
            self.v_quantizer = q_reg[quant_type](**q_config['v_quantizer'])
        if 'p_quantizer' in q_config.keys():
            quant_type = q_config['p_quantizer'].pop('quant')
            self.p_quantizer = q_reg[quant_type](**q_config['p_quantizer'])
        if 'o_quantizer' in q_config.keys():
            quant_type = q_config['o_quantizer'].pop('quant')
            self.o_quantizer = q_reg[quant_type](**q_config['o_quantizer'])

    def prune_heads(self, heads: List[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)

        # Quantize keys and queries
        if hasattr(self, "k_quantizer"): k = self.k_quantizer(k)
        if hasattr(self, "q_quantizer"): q = self.q_quantizer(q)

        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)

        # Quantize attention scores
        if hasattr(self, "s_quantizer"): scores = self.s_quantizer(scores)

        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        # Quantize attention probabilities and values
        if hasattr(self, "p_quantizer"): weights = self.p_quantizer(weights)
        if hasattr(self, "v_quantizer"): v = self.v_quantizer(v.transpose(-1,-2)).transpose(-1,-2)

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        # Quantize attention outputs
        if hasattr(self, "o_quantizer"): context = self.o_quantizer(context)

        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)
