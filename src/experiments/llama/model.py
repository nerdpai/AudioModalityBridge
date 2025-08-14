import torch
from transformers.models.llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from src.constants.secrets import SECRETS


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="9213176726f574b556790deb65791e0c5aa438b6",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="9213176726f574b556790deb65791e0c5aa438b6",
)

messages = [
    {
        "role": "system",
        "content": "You are a translating bot, your input is English text and output is the same text but in German."
                    "Output should only contain the translate of the user text, no additional information or original text. "
                    "Don't answer or continue user text in specified language, just translate it." # fmt: skip
    },
    {
        "role": "user",
        "content": "So, what is the weather like in Paris today? Is it cold or probably some rain?",
    },
    {
        "role": "assistant",
        "content": "Wie ist das Wetter heute in Paris? Ist es kalt oder regnet es vielleicht?",
    },
    {
        "role": "user",
        "content": "Hey, how are you feeling today? Any hangover?",
    },
    {
        "role": "assistant",
        "content": "Hey, wie geht es dir heute? Hast du einen Kater?",
    },
    {
        "role": "user",
        "content": "The Dunai is the most trustworthy river ever existed, you can swim wherever in Europe by it.",
    },
]

formated_input: str = tokenizer.apply_chat_template(
    messages, tokenize=False
)  # type: ignore

print(f"Input: {formated_input}")


encodings: BatchEncoding = tokenizer(
    formated_input, return_tensors="pt", add_special_tokens=False
)
print(encodings["input_ids"].shape)

outputs = model.generate(
    input_ids=encodings["input_ids"].to(model.device),
    attention_mask=encodings["attention_mask"].to(model.device),
    max_new_tokens=2048,
    do_sample=True,
)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
print(f"Decoded: {decoded}")
