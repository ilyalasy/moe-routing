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
        def hook(module: Top2Router, args, kwargs, output):  #
            # see https://github.com/hpcaitech/ColossalAI/blob/5d380a1a215204d827604c4797be12aad001424a/colossalai/moe/layers.py#L172
            experts_mask = output[1]
            experts_per_token = []
            for tok_experts in experts_mask:
                full_experts = torch.full(
                    (module.k_value,), -1
                )  # keep shape even if token is overflowed
                selected_experts = tok_experts.nonzero(as_tuple=True)[0]
                full_experts[: len(selected_experts)] = selected_experts
                experts_per_token.append(full_experts)
            experts_per_token = torch.stack(experts_per_token).detach().cpu()
            save_dict[layer_name].append(experts_per_token)

        return hook

    hooks = []
    activated_experts = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, Top2Router):
            handle = module.register_forward_hook(
                create_hook(activated_experts, name), with_kwargs=True
            )
            hooks.append(handle)

    return activated_experts, hooks
