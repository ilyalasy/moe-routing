from collections import defaultdict
from typing import Callable, Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
from colossalai.moe.routers import Top2Router
from models.modelling_deepseek import MoEGate
from torch.utils.hooks import RemovableHandle


def _create_hook(
    save_dict: Dict,
    layer_name: str,
    model_type: Literal["openmoe", "deepseek"],
) -> Callable:
    def hook_openmoe(module: Top2Router, args, kwargs, output):  #
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

    def hook_deepseek(module: MoEGate, args, kwargs, output):
        experts_per_token = output[0]
        save_dict[layer_name].append(experts_per_token.detach().cpu())

    if model_type == "openmoe":
        return hook_openmoe
    if model_type == "deepseek":
        return hook_deepseek


ROUTER_CLASSES = {"openmoe": Top2Router, "deepseek": MoEGate}


def set_router_hook(
    model: torch.nn.Module, model_type: Literal["openmoe", "deepseek"]
) -> Tuple[Dict, List[RemovableHandle]]:
    router_class = ROUTER_CLASSES[model_type]
    hooks = []
    activated_experts = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, router_class):
            handle = module.register_forward_hook(
                _create_hook(activated_experts, name, model_type),
                with_kwargs=True,
            )
            hooks.append(handle)

    return activated_experts, hooks
