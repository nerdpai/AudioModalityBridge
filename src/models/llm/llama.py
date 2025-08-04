import torch
from transformers.models.llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.models.language_model import LanguageModel, LanguageModelConfig
from src.constants.models import MODELS


class LLama3Model(LanguageModel):
    def __init__(self, device: torch.device = torch.device("cpu")):
        config = self.get_config()
        card = config.model_card.asdict()

        model = LlamaForCausalLM.from_pretrained(
            **card,
            device_map=device,
        )

        tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
            **card
        )
        tokenizer.pad_token = tokenizer.eos_token

        super().__init__(
            model,
            model.model,
            model.model.embed_tokens,
            tokenizer,
        )

    @staticmethod
    def get_config() -> LanguageModelConfig:
        return MODELS.llm.llama3
