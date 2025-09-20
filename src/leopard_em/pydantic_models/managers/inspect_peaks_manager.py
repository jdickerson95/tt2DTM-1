"""Pydantic model for running the inspect peaks program."""

from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import ConfigDict

from leopard_em.backend.core_inspect_peaks import core_inspect_peaks
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.formats import INSPECT_PEAKS_DF_COLUMN_ORDER
from leopard_em.pydantic_models.utils import setup_particle_backend_kwargs
from leopard_em.utils.data_io import load_mrc_volume, load_template_tensor


class InspectPeaksManager(BaseModel2DTM):
    """Model holding parameters necessary for running the refine template program.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    template_volume : ExcludedTensor
        The template volume tensor (excluded from serialization).

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the inspect peaks manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend inspect peaks core function.
    run_inspect_peaks(self, correlation_batch_size: int = 32) -> None
        Run the inspect peaks program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig

    # Excluded tensors
    template_volume: ExcludedTensor

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    def make_backend_core_function_kwargs(
        self, prefer_refined_angles: bool = True
    ) -> dict[str, Any]:
        """Create the kwargs for the backend inspect peaks core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use the refined angles from the particle stack. Defaults to
            False.
        """
        # Ensure the template is loaded in as a Tensor object
        template = load_template_tensor(
            template_volume=self.template_volume,
            template_volume_path=self.template_volume_path,
        )

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)

        # The relative Euler angle offsets to search over (none for now)
        euler_angle_offsets = torch.zeros((1, 3))

        # The relative defocus values to search over
        defocus_offsets = torch.tensor([0.0])

        # No pixel size refinement
        pixel_size_offsets = torch.tensor([0.0])

        # Use the common utility function to set up the backend kwargs
        # pylint: disable=duplicate-code
        return setup_particle_backend_kwargs(
            particle_stack=self.particle_stack,
            template=template,
            preprocessing_filters=self.preprocessing_filters,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            device_list=self.computational_config.gpu_devices,
        )

    def run_inspect_peaks(
        self, output_dataframe_path: str, correlation_batch_size: int = 32
    ) -> None:
        """Run the inspect peaks program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the inspect peaks particle data.
        correlation_batch_size : int
            Number of cross-correlations to process in one batch, defaults to 32.
        """
        backend_kwargs = self.make_backend_core_function_kwargs()

        result = self.get_inspect_peaks_result(backend_kwargs, correlation_batch_size)

        self.inspect_peaks_result_to_dataframe(
            output_dataframe_path=output_dataframe_path, result=result
        )

    def get_inspect_peaks_result(
        self, backend_kwargs: dict, correlation_batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Get inspect peaks result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        correlation_batch_size : int
            Number of orientations to process at once. Defaults to 32.

        Returns
        -------
        dict[str, np.ndarray]
            The result of the inspect peaks program.
        """
        # pylint: disable=duplicate-code
        result: dict[str, np.ndarray] = {}
        result = core_inspect_peaks(
            batch_size=correlation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            **backend_kwargs,
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}

        return result

    def inspect_peaks_result_to_dataframe(
        self, output_dataframe_path: str, result: dict[str, np.ndarray]
    ) -> None:
        """Convert inspect peaks result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the inspect peaks particle data.
        result : dict[str, np.ndarray]
            The result of the inspect peaks program.
        """
        # pylint: disable=duplicate-code
        df_inspect_peaks = self.particle_stack._df.copy()  # pylint: disable=protected-access

        df_inspect_peaks["inspect_peaks_scaled_mip"] = result["average_z_score"]
        df_inspect_peaks["inspect_peaks_mip"] = result["average_surrounding_cc"]

        # Reorder the columns
        df_inspect_peaks = df_inspect_peaks.reindex(columns=INSPECT_PEAKS_DF_COLUMN_ORDER)

        # Save the inspect peaks DataFrame to disk
        df_inspect_peaks.to_csv(output_dataframe_path)
