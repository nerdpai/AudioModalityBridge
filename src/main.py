from typing import Callable

from src import (
    prepare_dataset,
    validate_presets,
    train_models,
)

MODULES: dict[str, Callable[[], None]] = {
    "Prepare COMMON_VOICE Dataset": prepare_dataset.run,
    "Validate Model Presets": validate_presets.run,
    "Train Models": train_models.run,
}


def menu(modules: dict[str, Callable]) -> None:
    i: int = 0
    print("Module Selection:")
    for i, module_name in enumerate(modules.keys()):
        print(f"{i}. {module_name}")
    print(f"(any other value). Exit")


def options_loop(modules: list[Callable]):
    while True:
        try:
            choice = int(input(f"\nSelect module to run (0-{len(modules)-1}): "))

            if choice < 0 or choice >= len(modules):
                print("Exiting...")
                break

        except ValueError as e:
            print(f"Invalid input: {e}")
            print("Please enter a valid number.")
            continue

        modules[choice]()


def main():
    menu(MODULES)
    options_loop(list(MODULES.values()))


if __name__ == "__main__":
    main()
