from collections import defaultdict
from copy import copy
from typing import Callable, Dict, List

import torch
import torch.distributed as dist
from colossalai.moe.utils import set_moe_args
from transformers import LlamaConfig
import http.client


def set_openmoe_args(
    config: LlamaConfig,
    num_experts: int,
    moe_layer_interval: int,
    router_topk: int = 2,
    router_capacity_factor_train: float = 1.25,
    router_capacity_factor_eval: float = 2.0,
    router_min_capacity: int = 4,
    router_noisy_policy: str = None,
    router_drop_tks: bool = True,
    router_aux_loss_factor: float = 0.01,
    router_z_loss_factor: float = 0.0001,
    mlp_gated: bool = True,
    label_smoothing: float = 0.001,
    z_loss_factor: float = 0.01,
    enable_load_balance: bool = False,
    load_balance_tolerance: float = 0.1,
    load_balance_beam_width: int = 8,
    load_balance_group_swap_factor: float = 0.4,
    enable_kernel: bool = False,
    enable_comm_overlap: bool = False,
    enable_hierarchical_alltoall: bool = False,
) -> None:
    """
    MoE related arguments.
    It inserts the MoE arguments into the Llama config.

    Args:
        config (LlamaConfig): Transformers Llama config.
        num_experts (int, optional): Number of experts.
        moe_layer_interval (int, optional): The interval moe layer.
        router_topk (int, optional): Moe router top k. Defaults to 2.
        router_capacity_factor_train (float, optional): Moe router max capacity for train. Defaults to 1.25.
        router_capacity_factor_eval (float, optional): Moe router max capacity for eval. Defaults to 2.0.
        router_min_capacity (int, optional): Moe router min capacity. Defaults to 4.
        router_noisy_policy (str, optional): Moe router noisy policy. You can choose [Jitter, Gaussian, None]. Defaults to None.
        router_drop_tks (bool, optional): Whether moe router drop tokens which exceed max capacity. Defaults to True.
        router_aux_loss_factor (float, optional): Moe router aux loss. You can refer to STMoE for details. Defaults to 0.01.
        router_z_loss_factor (float, optional): Moe router z loss. You can refer to STMoE for details. Defaults to 0.01.
        mlp_gated (bool, optional): Use gate in mlp. Defaults to True.
        label_smoothing (float, optional): Label smoothing. Defaults to 0.001.
        z_loss_factor (float, optional): The final outputs' classification z loss factor. Defaults to 0.01.
        enable_load_balance (bool, optional): Expert load balance. Defaults to False.
        load_balance_tolerance (float, optional): Expert load balance search's difference tolerance. Defaults to 0.1.
        load_balance_beam_width (int, optional): Expert load balance search's beam width. Defaults to 8.
        load_balance_group_swap_factor (float, optional): Expert load balance group swap factor. Longer value encourages less swap. Defaults to 0.4.
        enable_kernel (bool, optional): Use kernel optimization. Defaults to False.
        enable_comm_overlap (bool, optional): Use communication overlap for MoE. Recommended to enable for muiti-node training. Defaults to False.
        enable_hierarchical_alltoall (bool, optional): Use hierarchical alltoall for MoE. Defaults to False.
    """
    moe_args = dict(
        num_experts=num_experts,
        moe_layer_interval=moe_layer_interval,
        router_topk=router_topk,
        router_capacity_factor_train=router_capacity_factor_train,
        router_capacity_factor_eval=router_capacity_factor_eval,
        router_min_capacity=router_min_capacity,
        router_noisy_policy=router_noisy_policy,
        router_drop_tks=router_drop_tks,
        router_aux_loss_factor=router_aux_loss_factor,
        router_z_loss_factor=router_z_loss_factor,
        mlp_gated=mlp_gated,
        label_smoothing=label_smoothing,
        z_loss_factor=z_loss_factor,
        enable_load_balance=enable_load_balance,
        load_balance_tolerance=load_balance_tolerance,
        load_balance_beam_width=load_balance_beam_width,
        load_balance_group_swap_factor=load_balance_group_swap_factor,
        enable_kernel=enable_kernel,
        enable_comm_overlap=enable_comm_overlap,
        enable_hierarchical_alltoall=enable_hierarchical_alltoall,
    )
    set_moe_args(config, moe_args)


def gather_dict(
    num_processes: int, result_dict: Dict[str, List[str | torch.Tensor]]
):
    output_objects = [None for _ in range(num_processes)]
    dist.all_gather_object(output_objects, result_dict)
    gathered = defaultdict(list)
    for obj in output_objects:
        for k, v in obj.items():
            gathered[k].extend(v)
    return stack_tensors(gathered, to_cpu=True)


def stack_tensors(
    dict_of_tensors: Dict[str, List[str | torch.Tensor]], to_cpu=False
):
    dict_of_tensors = copy(dict_of_tensors)
    for k, v in dict_of_tensors.items():
        if isinstance(v[0], torch.Tensor):
            v = [t.cpu() for t in v] if to_cpu else v
            dict_of_tensors[k] = torch.stack(v)
    return dict_of_tensors


def print_vram_info():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free = free / 1024**3
        total = total / 1024**3
        print(f"GPU {i}: {total - free:.2f}/{total:.2f} GB")


def run_with_retries(
    func: Callable, exception=http.client.IncompleteRead, retries=10, **kwargs
):
    for _ in range(retries):
        try:
            return func(**kwargs)
        except exception as e:
            print(e)
            print("Retrying...")
            continue
