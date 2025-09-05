from src.constants.template import GENERATION_TEMPLATE
from src.constants.train import PRESETS_FACTORY

template = GENERATION_TEMPLATE

voicelm = PRESETS_FACTORY["classification/ast"]()

eos_token: str = voicelm.tokenizer.eos_token  # type: ignore
print(f"EOS token: {eos_token}")

instruction: str = voicelm.tokenizer.apply_chat_template(template, tokenize=False)  # type: ignore
instruction = instruction.removesuffix(eos_token)
prompt1 = "We are in a deep shit. So, what is the goddamn plan?!"
prompt2 = "I have no idea what you are talking about Jimmy, the thing was left on your table yesterday at 8pm"

result = voicelm.generate(
    [
        [instruction, prompt1, eos_token],
        [instruction, prompt2, eos_token],
    ],
    max_new_tokens=100,
    do_sample=True,
)
print(result)
print(voicelm.tokenizer.batch_decode(result))
