from typing import Literal

import pandera.pandas as pa
from pandera.typing.pandas import Series


Splits = Literal["train", "dev", "test"]


class CommonVoiceModel(pa.DataFrameModel):
    client_id: Series[str]
    path: Series[str]
    sentence: Series[str]
    up_votes: Series[int] = pa.Field(ge=0)
    down_votes: Series[int] = pa.Field(ge=0)
    age: Series[str]
    gender: Series[str]
    accents: Series[str]
    locale: Series[str]
    segment: Series[str]
