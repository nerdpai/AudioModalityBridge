from typing import Literal

Languages = Literal["german", "french", "spanish", "italian"]

SYSTEM = str("You are a translating bot, your input is English text and output is the same text but in {language}."
        "Output should only contain the translate of the user text, no additional information or original text. "
        "Don't answer or continue user text in specified language, just translate it.") # fmt: skip
FIRST_USER = (
    "So, what is the weather like in Paris today? Is it cold or probably some rain?"
)
SECOND_USER = "Hey, how are you feeling today? Any hangover?"

FEW_SHOT_TEMPLATES: dict[Languages, list[dict[str, str]]] = {
    "german": [
        {
            "role": "system",
            "content": SYSTEM.format(language="German"),
        },
        {
            "role": "user",
            "content": FIRST_USER,
        },
        {
            "role": "assistant",
            "content": "Wie ist das Wetter heute in Paris? Ist es kalt oder regnet es vielleicht?",
        },
        {
            "role": "user",
            "content": SECOND_USER,
        },
        {
            "role": "assistant",
            "content": "Hey, wie geht es dir heute? Hast du einen Kater?",
        },
        {
            "role": "user",
            "content": "",
        },
    ],
    "french": [
        {
            "role": "system",
            "content": SYSTEM.format(language="French"),
        },
        {
            "role": "user",
            "content": FIRST_USER,
        },
        {
            "role": "assistant",
            "content": "Alors, quel temps fait-il à Paris aujourd'hui? Il fait froid ou il va probablement pleuvoir?",
        },
        {
            "role": "user",
            "content": SECOND_USER,
        },
        {
            "role": "assistant",
            "content": "Salut, comment tu te sens aujourd'hui? Tu as la gueule de bois?",
        },
        {
            "role": "user",
            "content": "",
        },
    ],
    "spanish": [
        {
            "role": "system",
            "content": SYSTEM.format(language="Spanish"),
        },
        {
            "role": "user",
            "content": FIRST_USER,
        },
        {
            "role": "assistant",
            "content": "Entonces, ¿qué tiempo hace hoy en París? ¿Hace frío o probablemente llueva?",
        },
        {
            "role": "user",
            "content": SECOND_USER,
        },
        {
            "role": "assistant",
            "content": "Hola, ¿cómo te sientes hoy? ¿Tienes resaca?",
        },
        {
            "role": "user",
            "content": "",
        },
    ],
    "italian": [
        {
            "role": "system",
            "content": SYSTEM.format(language="Italian"),
        },
        {
            "role": "user",
            "content": FIRST_USER,
        },
        {
            "role": "assistant",
            "content": "Allora, che tempo fa oggi a Parigi? Fa freddo o forse pioverà?",
        },
        {
            "role": "user",
            "content": SECOND_USER,
        },
        {
            "role": "assistant",
            "content": "Ehi, come ti senti oggi? Hai i postumi della sbornia?",
        },
        {
            "role": "user",
            "content": "",
        },
    ],
}
