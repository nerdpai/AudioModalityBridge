import re


def title_print(message: str) -> None:
    prefix_matches = re.match(r"^(\n*).*", message)
    sufix_matches = re.match(r"^(\n*).*", message[::-1])

    prefix = prefix_matches.group(1)  # type: ignore
    sufix = sufix_matches.group(1)  # type: ignore

    print(f"{prefix}{'-' * 10}{message.strip()}{'-' * 10}{sufix}")
