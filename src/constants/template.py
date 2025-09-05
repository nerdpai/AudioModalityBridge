from typing import Literal

SYSTEM = str("You are a long text generating model. "
        "You get nothing as the input and just starts to generate a long text without an end. "
        "This text should be some kind of the story, like the text of some ballad that bard would sing.") # fmt: skip

GENERATION_TEMPLATE: list[dict[str, str]] = [
    {
        "role": "system",
        "content": SYSTEM,
    },
    {
        "role": "assistant",
        "content": "",
    },
]
