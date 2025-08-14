from typing import Optional, Union, Callable, Protocol, TYPE_CHECKING

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput, GenerationMixin
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
import torch
from torch import nn

from src.types.models import LanguageModelConfig


class LanguageModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        step_model: PreTrainedModel,
        embed_layer: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()

        self.model = model
        self.step_model = step_model
        self.embed_layer = embed_layer
        self.tokenizer = tokenizer

    @property
    def config(self) -> LanguageModelConfig:
        return self.get_config()

    @staticmethod
    def get_config() -> LanguageModelConfig:
        raise NotImplementedError("Child did not implement the method")

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(
        self,
        inputs: torch.FloatTensor,  # [batch, seq, hidden]
        attention_mask: torch.Tensor,  # [batch, seq]
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.step_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        ).last_hidden_state

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        # others
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], list[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional[BaseStreamer] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        gen_model: GenerationMixin = self.model  # type: ignore

        return gen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            custom_generate=custom_generate,
            **kwargs,
        )


if TYPE_CHECKING:

    class LanguageModelProtocol(LanguageModel, Protocol):  # type: ignore
        def __init__(self, device: torch.device = torch.device("cpu")) -> None: ...

else:

    class LanguageModelProtocol(Protocol):
        def __init__(self, device: torch.device = torch.device("cpu")) -> None: ...
