
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertSelfAttention, BertForSequenceClassification



def max_float(
        exponent_bit_width: torch.Tensor,
        mantissa_bit_width: torch.Tensor,
        exponent_bias: torch.Tensor
    ) -> torch.Tensor:
    """
    Get the largest representable value for a given minifloat format.
    """

    exp = 2**exponent_bit_width - 1 - exponent_bias
    man = ((2**(mantissa_bit_width+1))-1) * (2**-mantissa_bit_width)

    value = man * 2**exp

    return value


class MXFPQuantizer(nn.Module):

    def __init__(self, exp_w=2, man_w=1, group_size=32):
        super().__init__()

        # Quantization configuration
        self.exp_w = exp_w
        self.man_w = man_w
        self.exp_bias = 2**(exp_w-1)-1
        self.group_size = group_size

        # Calibration and training
        self.calibration: bool = False
        self.samples: List[torch.Tensor] = []

    def post_calibration(self):

        # Stack samples along new dim before batch dimension
        samples_full = torch.stack(self.samples, 0)

        # Reshape into group size
        orig_shape = samples_full.shape
        reshape = list(orig_shape[:-1]) + [orig_shape[-1] // self.group_size, self.group_size]
        x_rs = samples_full.view(reshape)

        # Reshape to share scales along batch and sequence dimensions
        if len(x_rs.shape) == 6:
            samples_rs = x_rs.permute(2,4,0,1,3,5)
            samples_rs = samples_rs.reshape(samples_rs.shape[0],samples_rs.shape[1],-1)
            scales = self.compute_scale(samples_rs)[:,:,0]
            scales_rep = scales.view(1, scales.shape[0], 1, scales.shape[1], 1)
            scales_rep = scales_rep.expand(list(scales_rep.shape[:-1])+[self.group_size])
            scales_rep = scales_rep.reshape(list(scales_rep.shape[:-2])+[scales_rep.shape[-2]*scales_rep.shape[-1]])
        else:
            # Triggers if quantizer is used in the wrong place, not in attention
            import pdb; pdb.set_trace()

        self.scale_calib = scales_rep


    def start_calib(self):
        self.samples = []
        self.calibration = True

    def end_calib(self):
        self.post_calibration()
        self.samples = []
        self.calibration = False

    def forward(self, x):

        if self.calibration:
            self.samples.append(x)
            return x
        else:
            return self.quantize_tensor(x)


    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scales shared along innermost dimension.
        """

        orig_shape = x.shape

        # Get max values
        x_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_max = torch.where(x_max == 0, torch.ones_like(x_max), x_max)

        # Divide by largest representable power of 2
        max_pot = 2**(self.exp_w-1)

        # Restrict to power of 2
        x_pot = torch.log2(x_max) - max_pot
        x_pot = torch.floor(x_pot)
        x_pot = 2**x_pot

        # Clamp to UE8M0
        x_clamp = torch.clamp(x_pot, 0, 255)

        # Expand to original shape
        x_rep = x_clamp.expand(orig_shape)

        return x_rep

    def dynamic_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scales for each block in input tensor.
        """

        # Reshape into group size
        orig_shape = x.shape
        reshape = list(orig_shape[:-1]) + [orig_shape[-1] // self.group_size, self.group_size]
        x_rs = x.view(reshape)

        # Compute scales
        x_scales = self.compute_scale(x_rs)

        # Reshape to original shape
        x_rep = x_scales.reshape(orig_shape)

        return x_rep
    
    
    def to_minifloat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize values in input tensor to minifloat.
        """
        # Extract signs and zeros
        signs = x.sign()
        x_abs = x.abs()
        zeros = (x == 0)
        x_abs = torch.where(zeros, torch.ones_like(x_abs), x_abs)

        # Shift mantissas to keep man_w+1 bits before binary point
        exps = torch.floor(torch.log2(x_abs))
        mans = x_abs * (2 ** -exps)
        mans_shifted = mans * (2 ** self.man_w)

        # Round mantissas
        x_rnd = torch.round(mans_shifted)

        # Undo shifts
        x_rnd = x_rnd * (2 ** -self.man_w)
        x_rnd = x_rnd * (2 ** exps)

        # Clamp between max and min float values
        max_repr = max_float(self.exp_w, self.man_w, self.exp_bias)
        min_repr = 2**(-self.exp_bias)
        lim_zero = min_repr/2
        x_clamp = torch.clamp(x_rnd, min_repr, max_repr)
        x_clamp = torch.where(x_abs <= lim_zero, torch.zeros_like(x_clamp), x_clamp)

        # Reapply signs and zeros
        x_signed = x_clamp * signs
        x_signed = torch.where(zeros, torch.zeros_like(x_signed), x_signed)

        return x_signed

    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MX quantization to input tensor.
        """

        # scale = self.get_scale(x)
        scale = self.scale_calib

        # Unapply scales
        x_descale = x / scale

        # Round and clamp
        x_rnd = self.to_minifloat(x_descale)

        # Apply scales
        x_rescale = x_rnd * scale
        
        return x_rescale


class QuantBertSelfAttention(nn.Module):
    """
    A bert attention block with quantization inserted into forward pass.
    """
    def __init__(self, orig_attn: BertSelfAttention, quantizer=MXFPQuantizer, q_config={}):
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

        self.k_quantizer = quantizer(**q_config)
        self.q_quantizer = quantizer(**q_config)
        self.v_quantizer = quantizer(**q_config)
        self.s_quantizer = quantizer(**q_config)
        self.p_quantizer = quantizer(**q_config)
        self.o_quantizer = quantizer(**q_config)

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
        key_layer = self.k_quantizer(key_layer)
        query_layer = self.q_quantizer(query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Quantize attention scores
        attention_scores = self.s_quantizer(attention_scores)

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
        attention_probs = self.p_quantizer(attention_probs)
        value_layer = self.v_quantizer(value_layer)

        context_layer = torch.matmul(attention_probs, value_layer)

        # Quantize attention outputs
        context_layer = self.o_quantizer(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



def patch_bert_model(model: BertForSequenceClassification, q_config={}, attn_block=QuantBertSelfAttention, quantizer=MXFPQuantizer):
    """
    Replaces all BertSelfAttention modules with quantized attention.
    """
    quantizers = []

    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        original_attn = layer.attention.self
        quant_attn = attn_block(original_attn, quantizer, q_config)
        for child in quant_attn.children():
            if type(child) == quantizer:
                quantizers.append(child)
        layer.attention.self = quant_attn

    print("Model patched with quantized attention.")
    return model, quantizers
