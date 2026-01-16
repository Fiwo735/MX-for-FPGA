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
        self.samples_other: List[torch.Tensor] = []
        self.calibrated = False

    def start_calib(self):
        self.samples = []
        self.samples_other = []
        self.calibration = True

    def end_calib(self):
        if len(self.samples) != 0:
            self.post_calibration()
        self.samples = []
        self.samples_other = []
        self.calibration = False
        self.calibrated = True

    def forward(self, x: torch.Tensor, other: torch.Tensor = None) -> torch.Tensor:

        if self.calibration:
            self.samples.append(x.detach())
            if other != None:
                self.samples_other.append(other.detach())
            return x
        elif self.calibrated:
            return self.quantize_tensor(x)
        else:
            return x

    @abstractmethod
    def post_calibration(self):
        pass

    @abstractmethod
    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MXINTQuantizer(Quantizer):

    def __init__(self, bit_w=2, group_size=32, static_scale=False, symmetric=True, signed=True):
        super().__init__()

        # Quantization configuration
        self.bit_w = bit_w
        self.group_size = group_size
        self.static_scale = static_scale
        self.signed = signed
        self.symmetric = symmetric

        # Set calibrated to true if no scale calibration required
        self.calibrated = not self.static_scale

        # Other members
        self.register_buffer("scale_calib", torch.tensor(1))

    def post_calibration(self):

        # Stack samples and compute scale
        samples_full = torch.cat(self.samples, 0)

        # Reshape to [variable_dims, const_dims]
        orig_shape = samples_full.shape
        if len(orig_shape) == 4: # Assume [B,H,S,D]
            B, H, S, D = samples_full.shape
            samples_full = samples_full.permute(0,2,1,3)
            samples_full = samples_full.reshape(B*S,H*D)
        else: # Assume [*,D]
            samples_full = samples_full.reshape(-1, samples_full.size(-1))
        B, D = samples_full.shape
        D_prime = 1 if (self.group_size == -1) else (D // self.group_size)
        samples_full = samples_full.reshape(B, D_prime, self.group_size)

        # Compute [D'] scales on [B,D',g] samples.
        x_scale = self.compute_scale(samples_full)

        if D_prime > 1:
            if len(orig_shape) == 4:
                B, H, S, D = orig_shape
                if x_scale.size(0) > orig_shape[1]:
                    x_scale = x_scale.reshape(1, H, 1, D_prime // H, 1)
                    x_scale = x_scale.expand(1, H, 1, D_prime // H, D // (D_prime // H))
                    x_scale = x_scale.reshape(1, H, 1, D)
                else:
                    x_scale = x_scale.reshape(1, D_prime, 1, 1, 1)
                    x_scale = x_scale.expand(1, D_prime, H // D_prime, 1, 1)
                    x_scale = x_scale.reshape(1, H, 1, 1)
            else:
                D = orig_shape[-1]
                x_scale = x_scale.reshape(1, D_prime, 1)
                x_scale = x_scale.expand(1, D_prime, D // D_prime)
                x_scale = x_scale.reshape(1, D)

        self.scale_calib = x_scale

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Get max value
        x_max = x.abs().amax(dim=(0,-1)) # [B,D',g] -> [D']
        x_max = torch.where(x_max == 0, torch.ones_like(x_max), x_max)

        # Divide by largest representable power of 2
        if self.signed:
            max_pot = self.bit_w-1
        else:
            max_pot = self.bit_w

        # Exception for ternary:
        if (self.bit_w == 2) and self.symmetric and self.signed:
            max_pot = 1

        # Restrict to power of 2
        x_pot = torch.log2(x_max) - max_pot
        x_pot = torch.floor(x_pot)

        # Clamp to UE8M0
        x_clamp = torch.clamp(x_pot, -127, 128)

        x_scale = 2**x_clamp

        return x_scale

    def dynamic_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape to [variable_dims, const_dims]
        orig_shape = x.shape
        if len(orig_shape) == 4: # Assume [B,H,S,D]
            B, H, S, D = x.shape
            x = x.permute(0,2,1,3)
            x = x.reshape(1,B*S*H*D)
        else: # Assume [*,D]
            x = x.reshape(1, -1)
        B, D = x.shape
        D_prime = 1 if (self.group_size == -1) else (D // self.group_size)
        x = x.reshape(B, D_prime, self.group_size)

        # Compute [D'] scale on [B,D',g] input
        x_scale = self.compute_scale(x)

        if D_prime > 1:
            if len(orig_shape) == 4:
                B, H, S, D = orig_shape
                for dim in [B, S, H, D]:
                    if x_scale.size(-1) > dim: # make the next dim.
                        x_scale = x_scale.reshape(list(x_scale.shape[:-1]) + [dim, x_scale.size(-1) // dim])
                    elif x_scale.size(-1) == dim: # break
                        break
                    elif x_scale.size(-1) > 1: # expand last dim to match dim.
                        x_scale = x_scale.unsqueeze(-1)
                        x_scale = x_scale.expand(list(x_scale.shape[:-1]) + [dim // x_scale.size(-2)])
                        x_scale = x_scale.reshape(list(x_scale.shape[:-2]) + [dim])
                        break
                    else:
                        break
                # make x_scale have 4 dims, add at the end.
                while x_scale.dim() < 4:
                    x_scale = x_scale.unsqueeze(-1)
                # permute S and H.
                x_scale = x_scale.permute(0,2,1,3)
            else:
                for dim in orig_shape:
                    if x_scale.size(-1) > dim: # make the next dim.
                        x_scale = x_scale.reshape(list(x_scale.shape[:-1]) + [dim, x_scale.size(-1) // dim])
                    elif x_scale.size(-1) == dim: # break
                        break
                    elif x_scale.size(-1) > 1: # expand last dim to match dim.
                        x_scale = x_scale.unsqueeze(-1)
                        x_scale = x_scale.expand(list(x_scale.shape[:-1]) + [dim // x_scale.size(-2)])
                        x_scale = x_scale.reshape(list(x_scale.shape[:-2]) + [dim])
                        break
                    else:
                        break
                # make x_scale have same amount of dims as orig_shape, add at the end.
                while x_scale.dim() < len(orig_shape):
                    x_scale = x_scale.unsqueeze(-1)

        return x_scale

    def to_int(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize values in input tensor to integers.
        """

        # Round
        x_rnd = torch.round(x)

        # Clamp between max and min values
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
        Apply quantization to input tensor.
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
        x_rnd = self.to_int(x_descale)

        # Apply scales
        x_rescale = x_rnd * scale

        return x_rescale

    def extra_repr(self) -> str:
        return (
            f"bit_w={self.bit_w}, "
            f"group_size={self.group_size}, "
            f"signed={self.signed}, "
            f"symmetric={self.symmetric}, "
            f"static_scale={self.static_scale}"
        )

class MXFPQuantizer(Quantizer):

    def __init__(self, exp_w=2, man_w=1, group_size=32, static_scale=False, signed=True):
        super().__init__()

        # Quantization configuration
        self.exp_w = exp_w
        self.man_w = man_w
        self.group_size = group_size
        self.static_scale = static_scale
        self.signed = signed

        # Inferred parameters:
        self.exp_bias = 2**(exp_w-1)-1
        # Set calibrated to true if no scale calibration required
        self.calibrated = not self.static_scale

        # Other members
        self.register_buffer("scale_calib", torch.tensor(1))

    def post_calibration(self):

        # Stack samples and compute scale
        samples_full = torch.cat(self.samples, 0)

        # Reshape to [variable_dims, const_dims]
        orig_shape = samples_full.shape
        if len(orig_shape) == 4: # Assume [B,H,S,D]
            B, H, S, D = samples_full.shape
            samples_full = samples_full.permute(0,2,1,3)
            samples_full = samples_full.reshape(B*S,H*D)
        else: # Assume [*,D]
            samples_full = samples_full.reshape(-1, samples_full.size(-1))
        B, D = samples_full.shape
        D_prime = 1 if (self.group_size == -1) else (D // self.group_size)
        samples_full = samples_full.reshape(B, D_prime, self.group_size)

        # Compute [D'] scales on [B,D',g] samples.
        x_scale = self.compute_scale(samples_full)

        if D_prime > 1:
            if len(orig_shape) == 4:
                B, H, S, D = orig_shape
                if x_scale.size(0) > orig_shape[1]:
                    x_scale = x_scale.reshape(1, H, 1, D_prime // H, 1)
                    x_scale = x_scale.expand(1, H, 1, D_prime // H, D // (D_prime // H))
                    x_scale = x_scale.reshape(1, H, 1, D)
                else:
                    x_scale = x_scale.reshape(1, D_prime, 1, 1, 1)
                    x_scale = x_scale.expand(1, D_prime, H // D_prime, 1, 1)
                    x_scale = x_scale.reshape(1, H, 1, 1)
            else:
                D = orig_shape[-1]
                x_scale = x_scale.reshape(1, D_prime, 1)
                x_scale = x_scale.expand(1, D_prime, D // D_prime)
                x_scale = x_scale.reshape(1, D)

        self.scale_calib = x_scale

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Get max value
        x_max = x.abs().amax(dim=(0,-1)) # [B,D',g] -> [D']
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

        return x_scale

    def dynamic_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape to [variable_dims, const_dims]
        orig_shape = x.shape
        if len(orig_shape) == 4: # Assume [B,H,S,D]
            B, H, S, D = x.shape
            x = x.permute(0,2,1,3)
            x = x.reshape(1,B*S*H*D)
        else: # Assume [*,D]
            x = x.reshape(1, -1)
        B, D = x.shape
        D_prime = 1 if (self.group_size == -1) else (D // self.group_size)
        x = x.reshape(B, D_prime, self.group_size)

        # Compute [D'] scale on [B,D',g] input
        x_scale = self.compute_scale(x)

        if D_prime > 1:
            if len(orig_shape) == 4:
                B, H, S, D = orig_shape
                for dim in [B, S, H, D]:
                    if x_scale.size(-1) > dim: # make the next dim.
                        x_scale = x_scale.reshape(list(x_scale.shape[:-1]) + [dim, x_scale.size(-1) // dim])
                    elif x_scale.size(-1) == dim: # break
                        break
                    elif x_scale.size(-1) > 1: # expand last dim to match dim.
                        x_scale = x_scale.unsqueeze(-1)
                        x_scale = x_scale.expand(list(x_scale.shape[:-1]) + [dim // x_scale.size(-2)])
                        x_scale = x_scale.reshape(list(x_scale.shape[:-2]) + [dim])
                        break
                    else:
                        break
                # make x_scale have 4 dims, add at the end.
                while x_scale.dim() < 4:
                    x_scale = x_scale.unsqueeze(-1)
                # permute S and H.
                x_scale = x_scale.permute(0,2,1,3)
            else:
                for dim in orig_shape:
                    if x_scale.size(-1) > dim: # make the next dim.
                        x_scale = x_scale.reshape(list(x_scale.shape[:-1]) + [dim, x_scale.size(-1) // dim])
                    elif x_scale.size(-1) == dim: # break
                        break
                    elif x_scale.size(-1) > 1: # expand last dim to match dim.
                        x_scale = x_scale.unsqueeze(-1)
                        x_scale = x_scale.expand(list(x_scale.shape[:-1]) + [dim // x_scale.size(-2)])
                        x_scale = x_scale.reshape(list(x_scale.shape[:-2]) + [dim])
                        break
                    else:
                        break
                # make x_scale have same amount of dims as orig_shape, add at the end.
                while x_scale.dim() < len(orig_shape):
                    x_scale = x_scale.unsqueeze(-1)

        return x_scale

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
        if not self.signed:
            x_signed = torch.clamp(x_signed, min=0)

        return x_signed

    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization to input tensor.
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

    def extra_repr(self) -> str:
        return (
            f"exp_w={self.exp_w}, "
            f"man_w={self.man_w}, "
            f"group_size={self.group_size}, "
            f"signed={self.signed}, "
            f"static_scale={self.static_scale}"
        )

class IntQuantizer(Quantizer):

    def __init__(self, bit_w=8, group_size=-1, static_scale=True, symmetric=False, signed=True):
        super().__init__()

        # Quantization configuration
        self.bit_w = bit_w
        self.group_size = group_size
        self.static_scale = static_scale
        self.symmetric = symmetric
        self.signed = signed

        # Other members
        self.register_buffer("scale_calib", torch.tensor(1))

    def post_calibration(self):

        # Stack samples and compute scale
        samples_full = torch.cat(self.samples, 0)

        # Reshape to [variable_dims, const_dims]
        orig_shape = samples_full.shape
        if len(orig_shape) == 4: # Assume [B,H,S,D]
            B, H, S, D = samples_full.shape
            samples_full = samples_full.permute(0,2,1,3)
            samples_full = samples_full.reshape(B*S,H*D)
        else: # Assume [*,D]
            samples_full = samples_full.reshape(-1, samples_full.size(-1))
        B, D = samples_full.shape
        D_prime = 1 if (self.group_size == -1) else (D // self.group_size)
        samples_full = samples_full.reshape(B, D_prime, self.group_size)

        # Compute [D'] scales on [B,D',g] samples.
        x_scale = self.compute_scale(samples_full)

        if D_prime > 1:
            if len(orig_shape) == 4:
                B, H, S, D = orig_shape
                if x_scale.size(0) > orig_shape[1]:
                    x_scale = x_scale.reshape(1, H, 1, D_prime // H, 1)
                    x_scale = x_scale.expand(1, H, 1, D_prime // H, D // (D_prime // H))
                    x_scale = x_scale.reshape(1, H, 1, D)
                else:
                    x_scale = x_scale.reshape(1, D_prime, 1, 1, 1)
                    x_scale = x_scale.expand(1, D_prime, H // D_prime, 1, 1)
                    x_scale = x_scale.reshape(1, H, 1, 1)
            else:
                D = orig_shape[-1]
                x_scale = x_scale.reshape(1, D_prime, 1)
                x_scale = x_scale.expand(1, D_prime, D // D_prime)
                x_scale = x_scale.reshape(1, D)

        self.scale_calib = x_scale

    def dynamic_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape to [variable_dims, const_dims]
        orig_shape = x.shape
        if len(orig_shape) == 4: # Assume [B,H,S,D]
            B, H, S, D = x.shape
            x = x.permute(0,2,1,3)
            x = x.reshape(1,B*S*H*D)
        else: # Assume [*,D]
            x = x.reshape(1, -1)
        B, D = x.shape
        D_prime = 1 if (self.group_size == -1) else (D // self.group_size)
        x = x.reshape(B, D_prime, self.group_size)

        # Compute [D'] scale on [B,D',g] input
        x_scale = self.compute_scale(x)

        if D_prime > 1:
            if len(orig_shape) == 4:
                B, H, S, D = orig_shape
                for dim in [B, S, H, D]:
                    if x_scale.size(-1) > dim: # make the next dim.
                        x_scale = x_scale.reshape(list(x_scale.shape[:-1]) + [dim, x_scale.size(-1) // dim])
                    elif x_scale.size(-1) == dim: # break
                        break
                    elif x_scale.size(-1) > 1: # expand last dim to match dim.
                        x_scale = x_scale.unsqueeze(-1)
                        x_scale = x_scale.expand(list(x_scale.shape[:-1]) + [dim // x_scale.size(-2)])
                        x_scale = x_scale.reshape(list(x_scale.shape[:-2]) + [dim])
                        break
                    else:
                        break
                # make x_scale have 4 dims, add at the end.
                while x_scale.dim() < 4:
                    x_scale = x_scale.unsqueeze(-1)
                # permute S and H.
                x_scale = x_scale.permute(0,2,1,3)
            else:
                for dim in orig_shape:
                    if x_scale.size(-1) > dim: # make the next dim.
                        x_scale = x_scale.reshape(list(x_scale.shape[:-1]) + [dim, x_scale.size(-1) // dim])
                    elif x_scale.size(-1) == dim: # break
                        break
                    elif x_scale.size(-1) > 1: # expand last dim to match dim.
                        x_scale = x_scale.unsqueeze(-1)
                        x_scale = x_scale.expand(list(x_scale.shape[:-1]) + [dim // x_scale.size(-2)])
                        x_scale = x_scale.reshape(list(x_scale.shape[:-2]) + [dim])
                        break
                    else:
                        break
                # make x_scale have same amount of dims as orig_shape, add at the end.
                while x_scale.dim() < len(orig_shape):
                    x_scale = x_scale.unsqueeze(-1)

        return x_scale

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:

        # Get max value
        x_max = x.abs().amax(dim=(0,-1)) # [B,D',g] -> [D']

        # Divide by largest representable power of 2
        if self.signed:
            if self.symmetric:
                max_mag = 2**(self.bit_w-1) - 1
            else:
                max_mag = 2**(self.bit_w-1)
        else:
            max_mag = 2**(self.bit_w) - 1

        x_scale = x_max / max_mag

        return x_scale

    def to_int(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize values in input tensor to integers.
        """

        # Round
        x_rnd = torch.round(x)

        # Clamp between max and min values
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
            scale = self.dynamic_scale(x)

        # Unapply scales
        x_descale = x / scale

        # Round and clamp
        x_rnd = self.to_int(x_descale)

        # Apply scales
        x_rescale = x_rnd * scale
        
        return x_rescale

    def extra_repr(self) -> str:
        return (
            f"bit_w={self.bit_w}, "
            f"group_size={self.group_size}, "
            f"symmetric={self.symmetric}, "
            f"signed={self.signed}, "
            f"static_scale={self.static_scale}"
        )


q_reg = {
    "MXINTQuantizer": MXINTQuantizer,
    "MXFPQuantizer": MXFPQuantizer,
    "IntQuantizer": IntQuantizer,
}
