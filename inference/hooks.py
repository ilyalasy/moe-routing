from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from colossalai.moe.routers import Top2Router
from torch.utils.hooks import RemovableHandle


def set_router_hook(
    model: torch.nn.Module,
) -> Tuple[Dict, List[RemovableHandle]]:
    def create_hook(save_dict: Dict, layer_name: str) -> Callable:
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
