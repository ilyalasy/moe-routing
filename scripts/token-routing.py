import os
import json
import argparse
from typing import Dict, List
import numpy as np
from pathlib import Path
from datasets import load_dataset, load_from_disk
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

# mp.set_start_method("spawn", force=True)

from colossalai.moe.routers import Top2Router
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from accelerate import Accelerator,init_empty_weights
from accelerate.utils import tqdm, gather_object


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


def get_dataloader(args, tokenizer):
    data_dir = Path(os.environ.get("DATA")) / "datasets/redpajama-preproc"
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_path = data_dir / f"v0-{args.subset_size}-{args.seq_len}"

    if ds_path.exists():
        samples = load_from_disk(ds_path)
    else:
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            name="sample",
            languages=["en"],
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
        samples.save_to_disk(ds_path)

    samples = samples.with_format("torch")
    return DataLoader(
        samples, batch_size=args.batch_size, num_workers=args.num_workers
    )


def _gather_dict(
    num_processes: int, result_dict: Dict[str, List[str | torch.Tensor]]
):
    output_objects = [None for _ in range(num_processes)]
    dist.all_gather_object(output_objects, result_dict)
    gathered = defaultdict(list)
    for obj in output_objects:
        for k, v in obj.items():
            gathered[k].extend(v)
    return _stack_tensors(gathered, to_cpu=True)


def _stack_tensors(
    dict_of_tensors: Dict[str, List[str | torch.Tensor]], to_cpu=False
):
    for k, v in dict_of_tensors.items():
        if isinstance(v[0], torch.Tensor):
            v = [t.cpu() for t in v] if to_cpu else v
            dict_of_tensors[k] = torch.stack(v)
    return dict_of_tensors

def _print_vram_info():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free = free / 1024 ** 3
        total = total / 1024 ** 3
        print(f"GPU {i}: {total - free:.2f}/{total:.2f} GB")

def run_inference(args):
    accelerator = Accelerator()
    if accelerator.is_main_process:
        accelerator.print(
            "Datasets cache path: ", os.environ.get("HF_DATASETS_CACHE", "")
        )
        accelerator.print("Models cache path: ", os.environ.get("HF_HOME", ""))
        accelerator.print(f"Using {accelerator.num_processes} processes!")

    model_path = (
        f"OrionZheng/openmoe-{args.model}"  # "OrionZheng/openmoe-8b-1T" #
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        model_max_length=args.seq_len,
        use_fast=True,
    )

    dataloader = get_dataloader(args, tokenizer)

    # config = AutoConfig.from_pretrained(model_path)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        resume_download=True,
    )
    _print_vram_info()

    activated_experts, hooks = set_router_hook(model)    
    # model, dataloader = accelerator.prepare(model, dataloader)
    # accelerator.prepare_data_loader()
    
    result = defaultdict(list)
    for batch_i, sample in tqdm(
        iterable=enumerate(dataloader), total=len(dataloader)
    ):
        model(
            input_ids=sample["input_ids"].to(model.device),
            attention_mask=sample["attention_mask"].to(model.device),
        )
        batch_res = get_batch_results(
            sample, activated_experts, batch_i, args.batch_size
        )
        for k, v in batch_res.items():
            result[k].extend(v)
    result = _stack_tensors(result)

    assert len(list(result.values())[0]) == (batch_i + 1) * args.batch_size

    accelerator.wait_for_everyone()
    final_results = _gather_dict(accelerator.num_processes, result)
    if accelerator.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
        "--subset_size",
        default=0.1,
        type=float,
        help="Size (in percentage) of the subset of RedPajama to use.",
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

    run_inference(args)
