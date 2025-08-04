from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    num_atten_heads: int
    translate_chunk_seconds: float
    in_out_rel: float
    overlap_audio_chunks: bool

    @classmethod
    def default(cls) -> "Preset":
        return cls(
            translate_chunk_seconds=5.0,
            in_out_rel=1.0,
            num_atten_heads=8,
            overlap_audio_chunks=True,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Preset":
        return cls(**data)

    @staticmethod
    def fields() -> list[str]:
        return [
            "num_atten_heads",
            "translate_chunk_seconds",
            "in_out_rel",
            "overlap_audio_chunks",
        ]
