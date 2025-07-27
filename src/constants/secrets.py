from typing import Literal
from pathlib import Path
import json


SecretsSchema = Literal["HF_ACCESS_TOKEN"]

SECRETS: dict[SecretsSchema, str] = json.load(
    Path("secrets.json").open("r", encoding="utf-8")
)
