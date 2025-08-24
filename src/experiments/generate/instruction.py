from src.constants.template import GENERATION_TEMPLATE
from src.constants.train import PRESETS_FACTORY

template = GENERATION_TEMPLATE

voicelm = PRESETS_FACTORY["classification/wav2vec2"]()

eos_token: str = voicelm.tokenizer.eos_token  # type: ignore
print(f"EOS token: {eos_token}")

instruction: str = voicelm.tokenizer.apply_chat_template(template, tokenize=False)  # type: ignore
instruction = instruction.removesuffix(eos_token)
transcription = "It was seen early in the morning, rushing over eastward."

result = voicelm.generate(
    [
        [instruction, transcription],
    ],
    max_new_tokens=512,
    do_sample=True,
)
print(result)
print(voicelm.tokenizer.batch_decode(result))
