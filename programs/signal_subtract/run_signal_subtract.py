"""Signal subtraction program to remove template-based signal from micrographs."""

import time

from leopard_em.pydantic_models.managers.signal_subtract_manager import (
    SignalSubtractManager,
)

#######################################
### Editable parameters for program ###
#######################################

# Path to the YAML configuration file for signal subtraction
YAML_CONFIG_PATH = "programs/signal_subtract/signal_subtract_example_config.yaml"

# Path where the subtracted micrograph will be saved
OUTPUT_PATH = "path/to/output/directory"

# Settings for signal subtraction
PREFER_REFINED_ANGLES = True
PREFER_REFINED_POSITIONS = True

###############################################################
### Main function called to run the signal subtract program ###
###############################################################


def main() -> None:
    """Main function for running the signal subtraction program."""
    print(f"Loading configuration from {YAML_CONFIG_PATH}...")

    # Initialize the manager with parameters from the YAML file
    manager = SignalSubtractManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print("Running signal subtraction...")
    start_time = time.time()

    # Run signal subtraction to get the subtracted micrograph
    subtracted_image = manager.run_signal_subtract(
        prefer_refined_angles=PREFER_REFINED_ANGLES,
        prefer_refined_positions=PREFER_REFINED_POSITIONS,
    )

    # Print the wall time in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Signal subtraction wall time: {elapsed_time_str}")

    # Save the result to a file
    print(f"Saving subtracted image to {OUTPUT_PATH}...")
    manager.save_subtracted_image(
        subtracted_image=subtracted_image, output_path=OUTPUT_PATH
    )

    print("Done!")


# NOTE: Invoking program under `if __name__ == "__main__"` necessary for multiprocessing
if __name__ == "__main__":
    main()
