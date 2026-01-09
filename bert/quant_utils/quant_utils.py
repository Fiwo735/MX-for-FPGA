from typing import List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn



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


class Quantizer(nn.Module, ABC):

    def __init__(self):
        super().__init__()

        self.calibration: bool = False
        self.samples: List[torch.Tensor] = []

    def start_calib(self):
        self.samples = []
        self.calibration = True

    def end_calib(self):
        if len(self.samples) != 0:
            self.post_calibration()
            self.samples = []
        self.calibration = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.calibration:
            self.samples.append(x.detach())
            return x
        else:
            return self.quantize_tensor(x)

    @abstractmethod
    def post_calibration(self):
        pass

    @abstractmethod
    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MXFPQuantizer(Quantizer):

    def __init__(self, exp_w=2, man_w=1, group_size=32, static_scale=False):
        super().__init__()

        # Quantization configuration
        self.exp_w = exp_w
        self.man_w = man_w
        self.exp_bias = 2**(exp_w-1)-1
        self.group_size = group_size
        self.static_scale = static_scale

        # Other members
        self.register_buffer("scale_calib", torch.tensor(1))

    def post_calibration(self):

        # Stack samples along new dim before batch dimension
        samples_full = torch.cat(self.samples, 0)

        # Reshape into group size
        orig_shape = samples_full.shape
        reshape = list(orig_shape[:-1]) + [orig_shape[-1] // self.group_size, self.group_size]
        x_rs = samples_full.view(reshape)

        # Reshape to share scales along batch and sequence dimensions
        if len(x_rs.shape) == 5: # [B, H, S, D', group_size]
            samples_rs = x_rs.permute(1,3,0,2,4)
            samples_rs = samples_rs.reshape(samples_rs.shape[0],samples_rs.shape[1],-1)
            scales = self.compute_scale(samples_rs)[:,:,0] # [H, D', B * S * group_size]
            scales_rep = scales.view(1, scales.shape[0], 1, scales.shape[1], 1) # [1, H, 1, D', 1]
            scales_rep = scales_rep.expand(list(scales_rep.shape[:-1])+[self.group_size])
            scales_rep = scales_rep.reshape(list(scales_rep.shape[:-2])+[scales_rep.shape[-2]*scales_rep.shape[-1]])
        else:
            # Triggers if quantizer is used in the wrong place, not in attention
            import pdb; pdb.set_trace()

        self.scale_calib = scales_rep

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scales shared along innermost dimension.
        """

        orig_shape = x.shape

        # Get max values
        x_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_max = torch.where(x_max == 0, torch.ones_like(x_max), x_max)

        # Divide by largest representable power of 2
        # 2^max_pot is the largest representable power of 2
        max_pot = 2**(self.exp_w-1)

        # Restrict to power of 2
        x_pot = torch.log2(x_max) - max_pot
        x_pot = torch.floor(x_pot)

        # Clamp to UE8M0
        x_clamp = torch.clamp(x_pot, -127, 128)

        x_scale = 2**x_clamp

        # Expand to original shape
        x_rep = x_scale.expand(orig_shape)

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

        if self.static_scale:
            scale = self.scale_calib
        else:
            scale = self.dynamic_scale(x)

        if (scale == 0).any():
            raise ValueError("A scale was set to 0, use torch.bfloat16 to try to avoid this.")

        # Unapply scales
        x_descale = x / scale

        # Round and clamp
        x_rnd = self.to_minifloat(x_descale)

        # Apply scales
        x_rescale = x_rnd * scale

        return x_rescale


class IntQuantizer(Quantizer):

    def __init__(self, bit_w=8, static_scale=True, symmetric=False, signed=True):
        super().__init__()

        # Quantization configuration
        self.bit_w = bit_w
        self.static_scale = static_scale
        self.symmetric = symmetric
        self.signed = signed

        # Other members
        self.register_buffer("scale_calib", torch.tensor(1))

    def post_calibration(self):

        # Stack samples and compute scale
        samples_full = torch.cat(self.samples, 0)
        self.scale_calib = self.compute_scale(samples_full)

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Get max value
        x_max = torch.max(torch.abs(x.flatten()))

        # Divide by largest representable power of 2
        if self.signed:
            max_mag = 2**(self.bit_w-1)
            # # This is the correct implementation but it performs worse with ternary quantization
            # if self.symmetric:
            #     max_mag = 2**(self.bit_w-1) - 1
            # else:
            #     max_mag = 2**(self.bit_w-1)
        else:
            max_mag = 2**(self.bit_w) - 1

        x_scale = x_max / max_mag

        return x_scale

    def to_int(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize values in input tensor to integers.
        """

        # Round mantissas
        x_rnd = torch.round(x)

        # Clamp between max and min float values
        if self.signed:
            max_repr = 2**(self.bit_w-1) - 1
            if self.symmetric:
                min_repr = - 2**(self.bit_w-1) + 1
            else:
                min_repr = - 2**(self.bit_w-1)
        else:
            max_repr = 2**(self.bit_w) - 1
            min_repr = 0
        x_clamp = torch.clamp(x_rnd, min_repr, max_repr)

        return x_clamp

    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply integer quantization to input tensor.
        """

        if self.static_scale:
            scale = self.scale_calib
        else:
            scale = self.compute_scale(x)

        # Unapply scales
        x_descale = x / scale

        # Round and clamp
        x_rnd = self.to_int(x_descale)

        # Apply scales
        x_rescale = x_rnd * scale
        
        return x_rescale


q_reg = {
    "MXFPQuantizer": MXFPQuantizer,
    "IntQuantizer": IntQuantizer,
}
