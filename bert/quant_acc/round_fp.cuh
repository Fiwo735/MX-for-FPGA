#ifndef ROUNDING_FUNCTIONS_CUH
#define ROUNDING_FUNCTIONS_CUH

#include <torch/extension.h>

__forceinline__ __device__ float round_rne_fp_full(float value, int man_width, int exp_width){

    // Extract sign, exponent, and mantissa
    const int bits = __float_as_int(value);
    const int sign = bits & 0x80000000;
    int exp = (bits >> 23) & 0xFF;

    // Calculate max exponent for the narrow format
    const int max_exp = (1 << (exp_width - 1)) + 127;
    const int min_exp = -max_exp + 2 + 127 + 127;
    const int man_dif = 23 - man_width;

    // Clamp exponent
    if (exp > max_exp){  // Too big, return max val.
        return __int_as_float(sign | ((max_exp) << 23) | (((1 << man_width) - 1) << (man_dif)));

    }else if (exp < (min_exp - man_width - 1)){ // Too small, return 0.
        return __int_as_float(sign);

    }else if (exp < min_exp){ // Could be 0, subnormal or normal.
        int man = bits & 0x007FFFFF;
        // Round mantissa
        const int man_diff_sub = man_dif + min_exp - exp;
        const int man_mask = (1 << (man_diff_sub)) - 1;
        const int round_bit = man & (1 << (man_diff_sub - 1));
        const int sticky_bits = man & (man_mask >> 1);

        man = (man >> (man_diff_sub)) << (man_diff_sub);

        if (round_bit && (sticky_bits || (man & (1 << (man_diff_sub))))) {
            man += (1 << (man_diff_sub));
            // Check for mantissa overflow
            if (man & 0x00800000) {
                man = 0;
                exp++;
            }
        }

        if (exp < (min_exp - man_width)){ // Too small, return 0.
            return __int_as_float(sign);
        }

        // Reassemble the float
        return __int_as_float(sign | (exp << 23) | man);

    }else{ // Normal or max val.
        int man = bits & 0x007FFFFF;
        // Round mantissa
        const int man_mask = (1 << (man_dif)) - 1;
        const int round_bit = man & (1 << (man_dif - 1));
        const int sticky_bits = man & (man_mask >> 1);

        man = (man >> (man_dif)) << (man_dif);

        if (round_bit && (sticky_bits || (man & (1 << (man_dif))))) {
            man += (1 << (man_dif));
            // Check for mantissa overflow
            if (man & 0x00800000) {
                man = 0;
                exp++;
                // Check for exponent overflow after rounding
                if (exp > max_exp) {
                    return __int_as_float(sign | ((max_exp) << 23) | (((1 << man_width) - 1) << (man_dif)));  // Clamp to max val
                }
            }
        }

        // Reassemble the float
        return __int_as_float(sign | (exp << 23) | man);
    }
}


#endif // ROUNDING_FUNCTIONS_CUH
