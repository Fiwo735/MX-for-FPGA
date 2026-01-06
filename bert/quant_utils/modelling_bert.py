import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention

from .quant_utils import IntQuantizer, MXFPQuantizer, q_reg



class QuantBertSelfAttention(nn.Module):
    """
    A bert attention block with quantization inserted into forward pass.
    """
    def __init__(self, orig_attn: BertSelfAttention, q_config={}):
        super().__init__()

        self.num_attention_heads = orig_attn.num_attention_heads
        self.attention_head_size = orig_attn.attention_head_size
        self.all_head_size = orig_attn.all_head_size

        self.query = orig_attn.query
        self.key = orig_attn.key
        self.value = orig_attn.value

        self.dropout = orig_attn.dropout
        self.position_embedding_type = orig_attn.position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = orig_attn.max_position_embeddings
            self.distance_embedding = orig_attn.distance_embedding

        self.is_decoder = orig_attn.is_decoder

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

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Quantize keys and queries
        if hasattr(self, "k_quantizer"): key_layer = self.k_quantizer(key_layer)
        if hasattr(self, "q_quantizer"): query_layer = self.q_quantizer(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Quantize attention scores
        if hasattr(self, "s_quantizer"): attention_scores = self.s_quantizer(attention_scores)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Quantize attention probabilities and values
        if hasattr(self, "p_quantizer"): attention_probs = self.p_quantizer(attention_probs)
        if hasattr(self, "v_quantizer"): value_layer = self.v_quantizer(value_layer.transpose(-1,-2)).transpose(-1,-2)

        context_layer = torch.matmul(attention_probs, value_layer)

        # Quantize attention outputs
        if hasattr(self, "o_quantizer"): context_layer = self.o_quantizer(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
