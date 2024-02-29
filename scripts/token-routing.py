import os
import json
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# mp.set_start_method("spawn", force=True)

from colossalai.moe.routers import Top2Router
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import tqdm

# # Install ColossalAI
# !git clone https://github.com/Orion-Zheng/ColossalAI.git
# !pip install ./ColossalAI
# !python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt


def set_router_hook(model):
    def create_hook(save_dict, layer_name):
        def hook(module, args, kwargs):  #
            inputs = kwargs["inputs"]
            # copied from https://github.com/hpcaitech/ColossalAI/blob/5d380a1a215204d827604c4797be12aad001424a/colossalai/moe/routers.py#L225
            probs = F.softmax(inputs, dim=-1)
            num_experts = probs.size(-1)

            top1_idx = torch.argmax(probs, dim=-1)
            mask1 = F.one_hot(top1_idx, num_classes=num_experts).to(
                torch.int32
            )
            logits_except1 = probs.masked_fill(mask1.bool(), float("-inf"))
            top2_idx = torch.argmax(logits_except1, dim=-1)
            chosen_experts = (
                torch.stack([top1_idx, top2_idx], dim=-1).detach().cpu()
            )
            save_dict[layer_name].append(chosen_experts)

        return hook

    hooks = []
    activated_experts = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, Top2Router):
            handle = module.register_forward_pre_hook(
                create_hook(activated_experts, name), with_kwargs=True
            )
            hooks.append(handle)

    return activated_experts, hooks


def get_preproc_funcs(tokenizer):
    def tokenization(example):
        return tokenizer(example["raw_content"])

    block_size = tokenizer.model_max_length

    def group_texts(example):
        needed_keys = ["doc_id", "url", "language", "source_domain"]
        batched_result = defaultdict(list)
        for i, ids in enumerate(example["input_ids"]):
            copy_num = int(np.ceil(len(ids) / block_size))

            meta = json.loads(example["meta"][i])
            meta["doc_id"] = example["doc_id"][i]
            for k in needed_keys:
                batched_result[k].extend([meta[k]] * copy_num)
            batched_ids = []
            attention_mask = []

            for i in range(copy_num):
                a = ids[i * block_size : (i + 1) * block_size]
                # left padding
                a = np.array(
                    [tokenizer.pad_token_id] * max(block_size - len(a), 0) + a
                )
                mask = np.ones(block_size)
                mask[a == tokenizer.pad_token_id] = 0
                batched_ids.append(a)
                attention_mask.append(mask)
            assert len(batched_ids) == copy_num

            batched_result["input_ids"].extend(batched_ids)
            batched_result["attention_mask"].extend(attention_mask)

        return batched_result

    return tokenization, group_texts


def get_batch_results(sample, activated_experts, batch_i, batch_size):
    res = {**sample}
    for key, expert_ids in activated_experts.items():
        res[key] = expert_ids[batch_i].reshape(batch_size, -1, 2)
    return res


def get_dataloader(args,tokenizer):
    ds = load_dataset(
        "togethercomputer/RedPajama-Data-V2", name="sample", languages=["en"]
    )
    ds_en = ds.filter(lambda r: json.loads(r["meta"])["language"] == "en")

    num_rows = int(ds_en["train"].num_rows * args.subset_size)
    ds_en = ds_en["train"].select(range(num_rows))

    tokenize, group_texts = get_preproc_funcs(tokenizer)
    tokenized = ds_en.map(tokenize, batched=True)
    samples = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized.column_names,
    )

    samples = samples.with_format("torch")

    return DataLoader(samples,batch_size=args.batch_size, num_workers=args.num_workers)

def run_inference(args):

    model_path = (
        f"OrionZheng/openmoe-{args.model}"  # "OrionZheng/openmoe-8b-1T" #
    )    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        model_max_length=args.seq_len,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        # device_map="auto",
        resume_download=True,
    )
    activated_experts, hooks = set_router_hook(model)
    dataloader = get_dataloader(args,tokenizer)

    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)    

    if accelerator.is_main_process:
        accelerator.print(f"Using {accelerator.num_processes} processes!") 

    result = defaultdict(list)
    for batch_i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):        
        outputs = model.generate(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            max_new_tokens=1,
        )
        batch_res = get_batch_results(
            sample, activated_experts, batch_i, args.batch_size
        )
        for k, v in batch_res.items():
            result[k].extend(v)
    for k,v in result.items():
        result[k] = torch.tensor(v,device=accelerator.device)
    assert len(list(result.values())[0]) == (batch_i + 1) * args.batch_size
    
    if accelerator.is_main_process():
        accelerator.wait_for_everyone()
        final_results = accelerator.gather(result)    
        args.output.parent.mkdir(parents=True,exist_ok=True)
        accelerator.save(final_results, args.output)        
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="base",
        type=str,
        help="model path",
        choices=["base", "8b-1T"],
    )
    parser.add_argument(
        "--output", default="output/experts.pt", type=Path, help="output path"
    )
    parser.add_argument(
        "--subset_size", default=1.0, type=float, help="Size (in percentage) of the subset of RedPajama to use."
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="batch size for model inference",
    )
    parser.add_argument(
        "--seq_len",
        default=512,
        type=int,
        help="max length of sample sequence length",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of workers for data loader",
    )
    args = parser.parse_args()

    print("Datasets cache path: ", os.environ.get("HF_DATASETS_CACHE", ""))
    print("Models cache path: ", os.environ.get("HF_HOME", ""))

    run_inference(args)