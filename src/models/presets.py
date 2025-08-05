from typing import Callable, TypeVar, Type, Optional

from src.types.preset import Preset, PresetsTypes
from src.constants.presets import PRESETS
from .llm.llama import LLama3Model
from .voicelm import VoiceLM
from .audio_model import AudioModelProtocol
from .modal_translator import ModalTranslator
from .audio_bridge import AudioBridge
from .language_model import LanguageModelProtocol


A = TypeVar("A", bound=AudioModelProtocol)
L = TypeVar("L", bound=LanguageModelProtocol)


VoiceLMGen = Callable[[], VoiceLM]
PresetFactory = tuple[Preset, VoiceLMGen]
PresetsFactory = dict[PresetsTypes, list[PresetFactory]]


def create_factory(
    preset: Preset,
    audio_t: Type[A],
    llm_t: Type[L],
) -> Optional[VoiceLMGen]:
    a_config = audio_t.get_config()
    if a_config.embed_dim % preset.num_atten_heads != 0:
        return None

    if preset.translate_chunk_seconds > preset.in_out_rel:
        return None

    def preset_fn() -> VoiceLM:
        audio_model = audio_t()
        llm_model = llm_t()

        in_seq = int(audio_model.config.seq_per_second * preset.translate_chunk_seconds)
        out_seq = int(in_seq / preset.in_out_rel)
        overlap_inputs = preset.overlap_audio_chunks
        num_heads = preset.num_atten_heads
        bridge_model: ModalTranslator = ModalTranslator(
            input_dim=audio_model.config.embed_dim,
            output_dim=llm_model.config.embed_dim,
            #
            in_seq=in_seq,
            out_seq=out_seq,
            overlap_inputs=overlap_inputs,
            num_heads=num_heads,
            #
            torch_dtype=audio_model.config.model_card.torch_dtype,
        )

        bridge = AudioBridge(audio_model, bridge_model)

        return VoiceLM(bridge, llm_model)

    return preset_fn


def create_factories(
    audio_t: Type[A],
    llm_t: Type[L],
    presets: list[Preset],
) -> list[PresetFactory]:
    factories = [(preset, create_factory(preset, audio_t, llm_t)) for preset in presets]
    return [f for f in factories if f[1] is not None]  # type: ignore


def classification_wav2vec2() -> PresetsFactory:
    from .classification.wav2vec2 import Wav2Vec2Model

    return {
        "classification/wav2vec2": create_factories(
            Wav2Vec2Model, LLama3Model, PRESETS
        ),
    }


def classification_ast() -> PresetsFactory:
    from .classification.ast import ASTModel

    return {
        "classification/ast": create_factories(ASTModel, LLama3Model, PRESETS),
    }


def asr_wav2vec2() -> PresetsFactory:
    from .asr.wav2vec2 import Wav2Vec2Model

    return {
        "asr/wav2vec2": create_factories(Wav2Vec2Model, LLama3Model, PRESETS),
    }


def asr_s2t() -> PresetsFactory:
    from .asr.s2t import S2TModel

    return {
        "asr/s2t": create_factories(S2TModel, LLama3Model, PRESETS),
    }


def asr_whisper() -> PresetsFactory:
    from .asr.whisper import WhisperModel

    return {
        "asr/whisper": create_factories(WhisperModel, LLama3Model, PRESETS),
    }


PRESETS_FACTORY: PresetsFactory = {
    **classification_wav2vec2(),
    **classification_ast(),
    **asr_wav2vec2(),
    **asr_s2t(),
    **asr_whisper(),
}
