import torch
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
import json

from dist_utils import save_weights, save_acts
from quant_utils.patch_utils import patch_bert_model
from quant_utils.modelling_bert import BertSelfAttention, QuantBertSelfAttention

def fetch_dataloader(tokenizer, num_samples=None, max_length=128, split="train", seed=42, batch_size=32):
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    dataset = load_dataset("glue", "sst2")
    samples = dataset[split].shuffle(seed=seed)
    if num_samples is not None:
        samples = samples.select(range(num_samples))
    data = samples.map(tokenize_function, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=split == "train")
    return dataloader

def get_logits_from_outputs(outputs):
    # Handle different output formats
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits']
    else:
        # If it's a tensor or other format, assume it's the logits directly
        return outputs

def validate_model(model, validation_loader, criterion=None, silent=False, mask=False):

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_loader, desc="Validating", disable=silent)):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            if mask:
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                outputs = model(input_ids)

            logits = get_logits_from_outputs(outputs)
            if criterion is not None:
                batch_loss = criterion(logits, labels)
                loss += batch_loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).to(dtype=torch.int64).sum().item()

    acc = correct / len(validation_loader.dataset) * 100

    return acc, loss

def create_tinybert_config():
    """Create TinyBERT configuration"""
    config = BertConfig(
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        hidden_act="relu",
        num_labels=2,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT on validation set')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: %(default)s)')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length (default: %(default)s)')
    parser.add_argument('--max_num_samples', type=int, default=None, help='Crop the validation set to a maximum number of samples. None=no cropping. (default: %(default)s)')
    parser.add_argument('--model_id', default='takedarn/bert-tiny-sst2', help='HF Model ID of target model to quantize (optional)')
    parser.add_argument('--silent', action='store_true', help='Silent mode (default: %(default)s)')
    parser.add_argument('--config', action='append', default=[], help='Config in the form name=json. Eg. --config k_quantizer=\{"quant":"MXFPQuantizer","man_w":8\}')

    args = parser.parse_args()

    configs = {}
    for item in args.config:
        name, json_str = item.split('=', 1)
        configs[name] = json.loads(json_str)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.silent:
        print(f"Using device: {device}")
    
    mask = True
    model_id = args.model_id
    # model_id = "takedarn/bert-tiny-sst2"
    # model_id = "M-FAC/bert-tiny-finetuned-sst2"
    # model_id = "gchhablani/bert-base-cased-finetuned-sst2"
    # model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        attn_implementation="eager",
        cache_dir='/data/models'
    )

    # Load calib data
    calib_loader = fetch_dataloader(
        tokenizer, 
        num_samples=128, 
        max_length=args.max_length, 
        split="train", 
        batch_size=args.batch_size
    )
    # Load validation data
    val_loader = fetch_dataloader(
        tokenizer, 
        num_samples=args.max_num_samples, 
        max_length=args.max_length, 
        split="validation", 
        batch_size=args.batch_size
    )


    # Patch model with quantized attention.
    model, quantizers, thresholds = patch_bert_model(
        model,
        attn_block=BertSelfAttention,
        quant_attn_block=QuantBertSelfAttention,
        q_config=configs,
    )
    model.to(device)
    model.eval()
    
    if not args.silent:
        print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Calibrate quantizers
    for q in quantizers:
        q.start_calib()
    for q in thresholds:
        q.start_calib()
    calib_acc, _ = validate_model(model, calib_loader, mask=mask)
    for q in quantizers:
        q.end_calib()
    for q in thresholds:
        q.end_calib()
    
    if not args.silent:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Evaluate model
    val_acc, _ = validate_model(model, val_loader, silent=args.silent, mask=mask)

    # weight_dir = pathlib.Path('saved_tensors/weights')
    # weight_dir.mkdir(parents=True, exist_ok=True)
    # save_weights(model, weight_dir)

    # act_dir = pathlib.Path('saved_tensors/acts')
    # act_dir.mkdir(parents=True, exist_ok=True)
    # save_acts(model, val_loader, model.device, act_dir)

    print(f"Validation accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
