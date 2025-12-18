import torch
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
import pathlib

from dist_utils import save_weights, save_acts

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
    data.set_format(type="torch", columns=["input_ids", "label"])
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

def validate_model(model, validation_loader, criterion=None, silent=False):

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_loader, desc="Validating", disable=silent)):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

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
    parser.add_argument('--model_path', default='bert/best_fp32_model.pth', help='Path to saved model weights (optional)')
    parser.add_argument('--silent', action='store_true', help='Silent mode (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.silent:
        print(f"Using device: {device}")
    
    # Load tokenizer and create model
    if not args.silent:
        print("Loading tokenizer and creating model...")
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    config = create_tinybert_config()
    model = BertForSequenceClassification(config)
    
    # Load saved weights if provided
    if args.model_path:
        if not args.silent:
            print(f"Loading model weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    
    if not args.silent:
        print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load validation data
    val_loader = fetch_dataloader(
        tokenizer, 
        num_samples=args.max_num_samples, 
        max_length=args.max_length, 
        split="validation", 
        batch_size=args.batch_size
    )
    
    if not args.silent:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Evaluate model
    val_acc, _ = validate_model(model, val_loader, silent=args.silent)

    weight_dir = pathlib.Path('bert/saved_tensors/weights')
    weight_dir.mkdir(parents=True, exist_ok=True)
    save_weights(model, weight_dir)

    act_dir = pathlib.Path('bert/saved_tensors/acts')
    act_dir.mkdir(parents=True, exist_ok=True)
    save_acts(model, val_loader, model.device, act_dir, silent=args.silent)

    print(f"Validation accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
