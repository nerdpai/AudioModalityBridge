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
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

formated_input: str = tokenizer.apply_chat_template(
    messages, tokenize=False
)  # type: ignore

print(f"Input: {formated_input}")


encodings: BatchEncoding = tokenizer(formated_input, return_tensors="pt")

outputs = model.generate(
    input_ids=encodings["input_ids"].to(model.device),
    attention_mask=encodings["attention_mask"].to(model.device),
    max_new_tokens=256,
    do_sample=True,
)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(f"Decoded: {decoded}")
