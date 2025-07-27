from typing import Union


def parse_bool(value: Union[str, bool]) -> bool:
    return str(value).lower() in ["true", "1", "t", "y", "yes"]
