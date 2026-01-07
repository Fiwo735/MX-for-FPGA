import argparse
import json
import random

from tqdm import tqdm
import torch
import transformers
import datasets
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quant_utils.patch_utils import patch_bert_model
from quant_utils.modelling_llama import LlamaAttention, QuantLlamaAttention



def get_model(model_id, max_length, device, bf16_model=True):

    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir='/data/models/',
    )
    config._attn_implementation = "eager"

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if bf16_model else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        config=config,
        torch_dtype=dtype,
        cache_dir='/data/models/',
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model.to(device)

    model.seqlen = max_length

    print("Start to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
        model_max_length=max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    print("Complete tokenizer loading...")

    model.config.use_cache = False

    return tokenizer, model


def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    
    if eval_mode:
        testdata = datasets.load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            cache_dir="/data/datasets/"
        )["test"]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        
        # Convert to same format as train set
        testloader = []
        for i in range(0, testenc.input_ids.shape[1] - seqlen, seqlen):
            j = i + seqlen
            inp = testenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            testloader.append((inp, tar))
        return testloader
    else:
        traindata = datasets.load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            cache_dir="/data/datasets/"
        )["train"]
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


@torch.no_grad()
def evaluator(model, testenc, dev, batch_size):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    nlls = []
    num_batches = (len(testenc) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(testenc), batch_size), desc="Evaluating", total=num_batches):
        batch_samples = testenc[i:i + batch_size]
        # Stack inputs from the batch
        batch = torch.cat([sample[0] for sample in batch_samples], dim=0).to(dev)
        loss = model(batch, labels=batch).loss
        neg_log_likelihood = loss.float()
        nlls.append(neg_log_likelihood.unsqueeze(-1))
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    
    model.config.use_cache = use_cache
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT on validation set')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: %(default)s)')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length (default: %(default)s)')
    parser.add_argument('--max_num_samples', type=int, default=None, help='Crop the validation set to a maximum number of samples. None=no cropping. (default: %(default)s)')
    parser.add_argument('--model_id', default='meta-llama/Llama-3.2-1B', help='HF Model ID of target model to quantize (optional)')
    parser.add_argument('--config', action='append', default=[], help='Config in the form name=json. Eg. --config k_thresh=\{"quant":"IntQuantizer","bit_w":8\}')

    args = parser.parse_args()

    configs = {}
    for item in args.config:
        name, json_str = item.split('=', 1)
        configs[name] = json.loads(json_str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    tokenizer, model = get_model(args.model_id, args.max_length, device)

    calib_loader = get_wikitext2(
        nsamples=1,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=False,
    )
    val_loader = get_wikitext2(
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )


    # Patch model with quantized attention.
    model, quantizers, thresholds = patch_bert_model(
        model,
        attn_block=LlamaAttention,
        quant_attn_block=QuantLlamaAttention,
        q_config=configs,
    )
    model.to(device)
    model.eval()
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Calibrate quantizers
    for q in thresholds:
        q.start_calib()
    for q in quantizers:
        q.start_calib()
    calib_ppl = evaluator(model, calib_loader, model.device, args.batch_size)
    for q in thresholds:
        q.end_calib()
    for q in quantizers:
        q.end_calib()


    # Evaluate model
    print(f"Validation samples: {len(val_loader)}")
    dataset_ppl = evaluator(model, val_loader, model.device, args.batch_size)
    print(f"\nPerplexity: {dataset_ppl:.2f}")

if __name__ == "__main__":
    main()
