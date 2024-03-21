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
from transformers import PreTrainedTokenizer

from inference.runner import MoERunner
from inference.utils import print_vram_info, stack_tensors
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


def get_batch_results(
    sample, activated_experts, batch_i, batch_size, start_batch=0
):
    res = {**sample}
    for key, expert_ids in activated_experts.items():
        res[key] = expert_ids[batch_i - start_batch].reshape(batch_size, -1, 2)
    return res


def get_dataloader(args, tokenizer, force_load=False):
    data_dir = Path(os.environ.get("DATA")) / "datasets/redpajama-preproc"
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_path = data_dir / f"v0-{args.model}-{args.subset_size}-{args.seq_len}"

    if ds_path.exists() and not force_load:
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


def run_test_sequence(
    tokenizer: PreTrainedTokenizer, model: OpenMoeForCausalLM
):
    input_str = """```
y = list(map(int, ['1', 'hello', '2']))
```
What error does this program produce?
ValueError: invalid literal for int() with base 10: 'hello'

```
sum = 0
for i in range(100):
        sum += i
```
What is the value of sum immediately after the 10th time line 3 is executed?"""

    input_ids = tokenizer(
        "<pad>" + input_str, return_tensors="pt", add_special_tokens=False
    )
    input_ids = input_ids.input_ids.cuda()
    generation_output = model.generate(
        input_ids, use_cache=True, do_sample=True, max_new_tokens=32
    )
    out = tokenizer.decode(generation_output[0], skip_special_tokens=False)
    print(f"output: \n{out}\n")


def run_inference(args, start_batch=0):

    print("Datasets cache path: ", os.environ.get("HF_DATASETS_CACHE", ""))
    print("Models cache path: ", os.environ.get("HF_HOME", ""))

    runner = MoERunner.from_name(args.model, args.seq_len)
    dataloader = get_dataloader(args, runner.tokenizer)
    print_vram_info()

    ## Uncomment to check whether the checkpoint is valid and model can produce reasonable text
    # run_test_sequence(runner.tokenizer,runner.model)

    result = defaultdict(list)
    save_step = int(len(dataloader) * 0.1)  # Save every 10% of the dataset
    for batch_i, sample in tqdm(
        iterable=enumerate(dataloader), total=len(dataloader)
    ):
        if batch_i < start_batch:
            continue
        activated_experts = runner(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
        )
        batch_res = get_batch_results(
            sample, activated_experts, batch_i, args.batch_size, start_batch
        )
        for k, v in batch_res.items():
            result[k].extend(v)

        if batch_i % save_step == 0:
            output_file = f"{args.model}-experts-{batch_i}.pt"
            batch_res = stack_tensors(result, to_cpu=True)
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(batch_res, args.output / output_file)
            print(f"Saved to {args.output / output_file}")

    result = stack_tensors(result, to_cpu=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = f"{args.model}-experts.pt"
    torch.save(result, args.output / output_file)
    print(f"Saved to {args.output / output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="deepseek",
        type=str,
        help="MoE Model name",
        choices=["openmoe", "mixtral", "deepseek"],
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
        default=32,
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
