# AudioModalityBridge

## Environment Setup

Python version management using conda or mamba is recommended. The required Python version is specified in the `.python-version` file.

## Configuration

A `secrets.json` file is required with the structure provided in `example.secrets.json`.

## Model Access

Access to the LLAMA3.2 model repository on Hugging Face is required, along with a configured read token for repository access.

## Project Structure

The project contains multiple branches, each with its own `results` folder. Branches differ in approaches, architectures, and training pipelines.

## Execution

Run the project using `python -m src.main` from within the configured Python environment with all dependencies installed.

## Modules

The project provides three main modules:
- `prepare_dataset`
- `train_models` 
- `validate_presets`

Module selection is prompted at runtime with corresponding numbers. Exit by providing any number outside the presented options.

## Hardware Requirements

The code was primarily developed and tested on H200 NVL with 140 GB VRAM. GPU configuration adjustments can be made through the `src.constants` module for compatibility with other hardware configurations.