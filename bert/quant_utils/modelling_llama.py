import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, Cache, logger, repeat_kv, apply_rotary_pos_emb

from .quant_utils import IntQuantizer, MXFPQuantizer, q_reg



class QuantLlamaAttention(nn.Module):
    """
    A llama attention block with quantization inserted into forward pass.
    """
    def __init__(self, orig_attn: LlamaAttention, q_config={}):
        super().__init__()
        self.config = orig_attn.config
        self.layer_idx = orig_attn.layer_idx

        self.attention_dropout = orig_attn.attention_dropout
        self.hidden_size = orig_attn.hidden_size
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.num_key_value_heads = orig_attn.num_key_value_heads
        self.num_key_value_groups = orig_attn.num_key_value_groups
        self.max_position_embeddings = orig_attn.max_position_embeddings
        self.rope_theta = orig_attn.rope_theta
        self.is_causal = orig_attn.is_causal

        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        self.rotary_emb = orig_attn.rotary_emb

        # Use CLI quantizer configs if available
        self.init_quantizers(q_config)

        # # Ternary Quantization
        # self.k_thresh = IntQuantizer(bit_w=2, symmetric=True)
        # self.q_thresh = IntQuantizer(bit_w=2, symmetric=True)
        # self.v_thresh = IntQuantizer(bit_w=2, symmetric=True)
        # self.p_thresh = IntQuantizer(bit_w=8, signed=False)

        # # self.k_thresh = TVQuantizer()
        # # self.q_thresh = TVQuantizer()

        # # Int Quantization
        # self.k_quantizer = IntQuantizer(bit_w=8,static_scale=False)
        # self.q_quantizer = IntQuantizer(bit_w=4)
        # self.s_quantizer = IntQuantizer(bit_w=4)
        # self.v_quantizer = IntQuantizer(bit_w=4)
        # self.p_quantizer = IntQuantizer(bit_w=8, signed=False)
        # self.o_quantizer = IntQuantizer(bit_w=4)

        # # # MXFP Quantization
        # self.k_quantizer = MXFPQuantizer(man_w=1,exp_w=2,group_size=32,static_scale=False)
        # self.q_quantizer = MXFPQuantizer(man_w=1,exp_w=2,group_size=32,static_scale=False)
        # self.s_quantizer = MXFPQuantizer(man_w=1,exp_w=2,group_size=32,static_scale=False)
        # self.v_quantizer = MXFPQuantizer(man_w=1,exp_w=2,group_size=32,static_scale=False)
        # self.p_quantizer = MXFPQuantizer(man_w=1,exp_w=2,group_size=32,static_scale=False)
        # self.o_quantizer = MXFPQuantizer(man_w=1,exp_w=2,group_size=32,static_scale=False)

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
        if 'k_thresh' in q_config.keys():
            quant_type = q_config['k_thresh'].pop('quant')
            self.k_thresh = q_reg[quant_type](**q_config['k_thresh'])
        if 'q_thresh' in q_config.keys():
            quant_type = q_config['q_thresh'].pop('quant')
            self.q_thresh = q_reg[quant_type](**q_config['q_thresh'])
        if 's_thresh' in q_config.keys():
            quant_type = q_config['s_thresh'].pop('quant')
            self.s_thresh = q_reg[quant_type](**q_config['s_thresh'])
        if 'v_thresh' in q_config.keys():
            quant_type = q_config['v_thresh'].pop('quant')
            self.v_thresh = q_reg[quant_type](**q_config['v_thresh'])
        if 'p_thresh' in q_config.keys():
            quant_type = q_config['p_thresh'].pop('quant')
            self.p_thresh = q_reg[quant_type](**q_config['p_thresh'])
        if 'o_thresh' in q_config.keys():
            quant_type = q_config['o_thresh'].pop('quant')
            self.o_thresh = q_reg[quant_type](**q_config['o_thresh'])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Quantize keys and queries
        # if self.k_thresh.calibration == False: import pdb; pdb.set_trace()
        if hasattr(self, "k_thresh"): key_states = self.k_thresh(key_states)
        if hasattr(self, "q_thresh"): query_states = self.q_thresh(query_states)
        if hasattr(self, "k_quantizer"): key_states = self.k_quantizer(key_states)
        if hasattr(self, "q_quantizer"): query_states = self.q_quantizer(query_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Quantize attention scores
        if hasattr(self, "s_thresh"): attn_weights = self.s_thresh(attn_weights)
        if hasattr(self, "s_quantizer"): attn_weights = self.s_quantizer(attn_weights)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Quantize attention probabilities and values
        if hasattr(self, "p_thresh"): attn_weights = self.p_thresh(attn_weights)
        if hasattr(self, "v_thresh"): value_states = self.v_thresh(value_states.transpose(-1,-2)).transpose(-1,-2)
        if hasattr(self, "p_quantizer"): attn_weights = self.p_quantizer(attn_weights)
        if hasattr(self, "v_quantizer"): value_states = self.v_quantizer(value_states.transpose(-1,-2)).transpose(-1,-2)

        attn_output = torch.matmul(attn_weights, value_states)

        # Quantize attention outputs
        if hasattr(self, "o_thresh"): attn_output = self.o_thresh(attn_output)
        if hasattr(self, "o_quantizer"): attn_output = self.o_quantizer(attn_output)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

