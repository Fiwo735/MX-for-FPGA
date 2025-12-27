import torch
import torch.nn as nn
from pathlib import Path



def find_qlayers(
    module,
    layers=[torch.nn.Linear, torch.nn.Embedding],
    name: str = "",
):
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embeddings": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


@torch.no_grad()
def save_weights(model, save_dir):
    """Save weights from BERT encoder layers"""
    for i, layer in enumerate(model.bert.encoder.layer):
        subset = find_qlayers(layer)
        for name in subset:
            W = subset[name].weight.data
            torch.save(W, f'{save_dir}/{i}_{name}_weight.pt')
        torch.cuda.empty_cache()


def capture_layer_io(layer, layer_input, attention_mask):
    """Capture input/output activations for BERT layer"""
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                # Handle tuple outputs (BERT attention returns tuples)
                if isinstance(output, tuple):
                    captured_vals[module_name].append(output[0].detach().cpu())
                else:
                    captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    captured_inputs = {
        "query": [],  # key, value have the same input as query
        "dense": [],  # attention output projection
    }

    captured_outputs = {
        "query": [],
        "key": [],
        "value": [],
        "dense": [],  # attention output
    }

    # Register hooks for attention layers
    for name in ["query", "key", "value"]:
        module = getattr(layer.attention.self, name)
        if name in captured_inputs:
            handles.append(
                module.register_forward_hook(hook_factory(name, captured_inputs, True))
            )
        handles.append(
            module.register_forward_hook(hook_factory(name, captured_outputs, False))
        )
    
    # Register hooks for attention output projection
    module = layer.attention.output.dense
    handles.append(
        module.register_forward_hook(hook_factory("dense", captured_inputs, True))
    )
    handles.append(
        module.register_forward_hook(hook_factory("dense", captured_outputs, False))
    )

    # Process each sequence in the batch one by one to avoid OOM
    for seq_idx in range(len(layer_input)):
        seq = layer_input[seq_idx].unsqueeze(0).to("cuda")
        mask = attention_mask[seq_idx].unsqueeze(0).to("cuda") if attention_mask is not None else None
        layer(seq, attention_mask=mask)

    # Concatenate accumulated inputs/outputs
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup
    for h in handles:
        h.remove()

    return {"input": captured_inputs, "output": captured_outputs}


@torch.no_grad()
def save_acts(model, val_loader, dev, output_dir, silent=False):
    """Save activations from BERT model using validation dataloader"""
    model.eval()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get first batch from dataloader
    batch = next(iter(val_loader))
    input_ids = batch['input_ids'].to(dev)
    attention_mask = None
    
    # Check if attention_mask is in the batch
    if 'attention_mask' in batch:
        attention_mask = batch['attention_mask'].to(dev)
    
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    
    if not silent:
        print(f"Processing batch with shape: {input_ids.shape}")

    # Move embeddings to device
    model.bert.embeddings = model.bert.embeddings.to(dev)
    layers = model.bert.encoder.layer
    layers[0] = layers[0].to(dev)

    # Capture the input to the first encoder layer
    inps = None
    cache = {"attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            nonlocal inps
            inps = hidden_states
            cache["attention_mask"] = kwargs.get("attention_mask")
            raise ValueError

    layers[0] = Catcher(layers[0])

    try:
        # Get embeddings
        embedding_output = model.bert.embeddings(input_ids)
        # Try to pass through first layer (will be caught)
        extended_attention_mask = model.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        ) if attention_mask is not None else None
        layers[0](embedding_output, attention_mask=extended_attention_mask)
    except ValueError:
        pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.bert.embeddings = model.bert.embeddings.cpu()
    
    attention_mask_extended = cache["attention_mask"]

    torch.cuda.empty_cache()

    # Save attention mask
    if attention_mask is not None:
        torch.save(attention_mask.cpu(), f'{output_dir}/attn_mask.pt')

    # Process each layer
    outs = None
    for i in range(len(layers)):
        if not silent:
            print(f"Processing Layer {i}...", flush=True)
        layer = layers[i].to(dev)

        # Capture layer input/output
        captured_io = capture_layer_io(layer, inps, attention_mask_extended)
        torch.save(captured_io, f'{output_dir}/{i}_act.pt')

        # for dir in captured_io.keys():
        #     for k in captured_io[dir].keys():
        #         torch.save(captured_io[dir][k], f'{output_dir}/{i}_{dir}_{k}.pt')

        # Forward pass through layer
        outs = layer(inps.to(dev), attention_mask=attention_mask_extended)[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps = outs.cpu()

    if not silent:
        print(f"Activation capture complete! Saved to {output_dir}")