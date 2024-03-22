from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

import torch
from inference.hooks import set_router_hook
from inference.utils import set_openmoe_args
from models.modelling_openmoe import OpenMoeForCausalLM
from models.modelling_deepseek import DeepseekForCausalLM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    MixtralForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
import torch.nn.functional as F

# Dictionary where keys are layer names and values are batched selected experts
LayerSelectedExperts = Dict[str, List[torch.LongTensor]]


class MoERunner:
    """
    Wrapper class for outputing routed experts during MoE inference
    """

    activated_experts = defaultdict(list)
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwds: Any,
    ) -> LayerSelectedExperts:
        raise NotImplementedError()

    def check_model_loaded(self):
        if self.model is None:
            raise Exception(
                "Runner was initialised with 'tokenizer_only', it is not suitable for model call."
            )

    @classmethod
    def from_name(
        cls,
        name: Literal["openmoe", "mixtral", "deepseek"],
        seq_len: int,
        tokenizer_only=False,
        use_quant=True,
        **kwargs,
    ) -> "MoERunner":
        if name == "openmoe":
            return OpenMoERunner(
                seq_len, use_quant, tokenizer_only=tokenizer_only, **kwargs
            )
        if name == "mixtral":
            return MixtralRunner(
                seq_len, use_quant, tokenizer_only=tokenizer_only, **kwargs
            )
        if name == "deepseek":
            return DeepSeekRunner(
                seq_len, use_quant, tokenizer_only=tokenizer_only, **kwargs
            )
        raise NotImplementedError()


class OpenMoERunner(MoERunner):
    """
    Wrapper class for outputing routed experts during Mixtral inference
    """

    def __init__(
        self, seq_len: int, use_quant=True, tokenizer_only=False, **kwargs
    ) -> None:
        self.model_name = f"OrionZheng/openmoe-8b-1T"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/umt5-small", model_max_length=seq_len
        )
        config = AutoConfig.from_pretrained(self.model_name)
        set_openmoe_args(
            config,
            num_experts=config.num_experts,
            moe_layer_interval=config.moe_layer_interval,
            enable_kernel=False,
        )
        self.model = None
        if not tokenizer_only:
            self.model = OpenMoeForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=self.quant_config if use_quant else None,
            )
            self.model.eval()
            self.activated_experts, hooks = set_router_hook(
                self.model, "openmoe"
            )

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LayerSelectedExperts:
        self.check_model_loaded()
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return self.activated_experts


class MixtralRunner(MoERunner):
    """
    Wrapper class for outputing routed experts during Mixtral inference
    """

    def __init__(
        self, seq_len: int, use_quant=True, tokenizer_only=False, **kwargs
    ) -> None:
        if use_quant:
            self.model_name = "TheBloke/mixtral-8x7b-v0.1-AWQ"
        else:
            self.model_name = "mistralai/Mixtral-8x7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, model_max_length=seq_len
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        if not tokenizer_only:
            self.model = MixtralForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                resume_download=True,
            )
            self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LayerSelectedExperts:
        self.check_model_loaded()
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        outputs: MoeCausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
            **kwargs,
        )
        for i, layer_router in enumerate(outputs.router_logits):
            routing_weights = F.softmax(layer_router, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(
                routing_weights, self.model.config.num_experts_per_tok, dim=-1
            )
            self.activated_experts[f"layer_{i}.router"].append(
                selected_experts
            )
        return self.activated_experts


class DeepSeekRunner(MoERunner):
    """
    Wrapper class for outputing routed experts during DeepSeek inference
    """

    def __init__(
        self, seq_len: int, use_quant=True, tokenizer_only=False, **kwargs
    ) -> None:
        self.model_name = "deepseek-ai/deepseek-moe-16b-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, model_max_length=seq_len
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

        if not tokenizer_only:
            self.model = DeepseekForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                resume_download=True,
                quantization_config=self.quant_config if use_quant else None,
            )
            self.model.eval()
            self.activated_experts, hooks = set_router_hook(
                self.model, "deepseek"
            )

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LayerSelectedExperts:
        self.check_model_loaded()
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return self.activated_experts
