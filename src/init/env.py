import os


def set_huggingface_env():
    os.environ["HF_HOME"] = "./.hf"
    os.environ["HF_HUB_VERBOSITY"] = "warning"

    # windows specific
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # bloatware
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def set_env():
    set_huggingface_env()
