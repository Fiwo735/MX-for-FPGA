from .quant_utils import Quantizer
from copy import deepcopy


def patch_bert_model(model, attn_block, quant_attn_block, q_config={}):
    """
    Replaces all instances of attn_block modules with quantized attention.
    """
    quantizers = []
    thresholds = []

    def replace_attention_recursive(parent_module, parent_name=''):
        """Recursively search and replace attention blocks."""
        for name, module in parent_module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Check if this module is an instance of attn_block
            if isinstance(module, attn_block):
                # Create quantized version
                quant_attn = quant_attn_block(module, deepcopy(q_config))

                # Collect quantizers and thresholds
                for child_name, child in quant_attn.named_children():
                    if 'thresh' in child_name:
                        thresholds.append(child)
                    elif isinstance(child, Quantizer):
                        quantizers.append(child)

                # Replace the module
                setattr(parent_module, name, quant_attn)
                print(f"Replaced {full_name} with quantized attention.")
            else:
                # Recursively search children
                replace_attention_recursive(module, full_name)

    replace_attention_recursive(model)

    print(f"Model patched with quantized attention. Total replacements: {len(quantizers) + len(thresholds)} quantizers found.")
    return model, quantizers, thresholds
