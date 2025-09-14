"""Local orientation refinement around sets of match template results."""

import time

from leopard_em.pydantic_models.managers import InspectPeaksManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the refine template program.
# Needs to reference the outputs from a match template run.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "/path/to/inspect-peaks-configuration.yaml"

# Path to where the dataframe with refined peak parameters will be output.
DATAFRAME_OUTPUT_PATH = "/path/to/inspect-peaks-results.csv"

# Number of particles to refine simultaneously. Will need to tune this parameter
# based on the memory & computational resources available.
PARTICLE_BATCH_SIZE = 80

###############################################################
### Main function called to run the inspect peaks program ###
###############################################################


def main() -> None:
    """Main function for running the refine template program."""
    inspect_manager = InspectPeaksManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print("Running inspect peaks...")

    start_time = time.time()

    inspect_manager.run_inspect_peaks(DATAFRAME_OUTPUT_PATH, PARTICLE_BATCH_SIZE)

    print("Finished core inspect_peaks call.")

    # Print the wall time of the search in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Inspect peaks wall time: {elapsed_time_str}")

    print("Done!")


# NOTE: Invoking  program under `if __name__ == "__main__"` necessary for multiprocesing
if __name__ == "__main__":
    main()
