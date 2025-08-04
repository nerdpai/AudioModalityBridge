from typing import Any
from itertools import product


def named_product(
    data: list[list[Any]], field_names: list[str]
) -> list[dict[str, Any]]:
    return [dict(zip(field_names, combination)) for combination in product(*data)]
