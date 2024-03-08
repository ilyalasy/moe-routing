import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from inference.hooks import set_router_hook
from inference.utils import print_vram_info, set_openmoe_args, stack_tensors
from models.modelling_openmoe import OpenMoeForCausalLM


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


def run_inference(args):
    print("Datasets cache path: ", os.environ.get("HF_DATASETS_CACHE", ""))
    print("Models cache path: ", os.environ.get("HF_HOME", ""))

    model_path = f"hpcai-tech/openmoe-{args.model}"

    tokenizer = AutoTokenizer.from_pretrained(
        "google/umt5-small",
        model_max_length=args.seq_len,
    )

    dataloader = get_dataloader(args, tokenizer)
    config = AutoConfig.from_pretrained(model_path)
    set_openmoe_args(
        config,
        num_experts=config.num_experts,
        moe_layer_interval=config.moe_layer_interval,
        enable_kernel=False,
    )
    model = OpenMoeForCausalLM.from_pretrained(
        model_path, config=config, device_map="auto"
    )
    print_vram_info()

    activated_experts, hooks = set_router_hook(model)

    result = defaultdict(list)
    for batch_i, sample in tqdm(
        iterable=enumerate(dataloader), total=len(dataloader)
    ):
        model(
            input_ids=sample["input_ids"].cuda(),
            attention_mask=sample["input_ids"].cuda(),
        )
        batch_res = get_batch_results(
            sample, activated_experts, batch_i, args.batch_size
        )
        for k, v in batch_res.items():
            result[k].extend(v)

        # save every 1000 batches
        if batch_i % 1000 == 0:
            batch_res = stack_tensors(result)
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(batch_res, args.output / f"experts-{batch_i}.pt")
            print(f"Saved to {args.output / f'experts-{batch_i}.pt'}")

    result = stack_tensors(result)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output / "experts.pt")
    print(f"Saved to {args.output / f'experts.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="8b",
        type=str,
        help="model path",
        choices=["base", "8b"],
    )
    parser.add_argument(
        "--output", default="output", type=Path, help="output path"
    )
    parser.add_argument(
        "--subset_size",
        default=0.01,
        type=float,
        help="Size (in percentage) of the subset of RedPajama to use.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size for model inference",
    )
    parser.add_argument(
        "--seq_len",
        default=256,
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
