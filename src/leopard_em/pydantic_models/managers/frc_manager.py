"""Pydantic model for running the FRC program."""

import os
from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import ConfigDict

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import setup_particle_backend_kwargs
from leopard_em.utils.data_io import load_mrc_volume, load_template_tensor
from leopard_em.pydantic_models.formats import REFINED_DF_COLUMN_ORDER


class FRCManager(BaseModel2DTM):
    """Model holding parameters necessary for running the frc program.

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
        Initialize the frc manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend frc core function.
    run_frc(self, output_text_path: str) -> None
        Run the frc program.
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
        """Create the kwargs for the backend refine_template core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use refined angles or not. Defaults to True.
        """
        # Ensure the template is loaded in as a Tensor object
        template = load_template_tensor(
            template_volume=self.template_volume,
            template_volume_path=self.template_volume_path,
        )

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)

        # The relative Euler angle offsets to search over (none for optimization)
        euler_angle_offsets = torch.zeros((1, 3))

        # The relative defocus values to search over (none for optimization)
        defocus_offsets = torch.tensor([0.0])

        # The relative pixel size values to search over (none for optimization)
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

    def run_frc(self) -> None:
        """Run the frc program.

        Parameters
        ----------
        """
        backend_kwargs = self.make_backend_core_function_kwargs()

        result = self.get_refine_result(backend_kwargs)

    def get_refine_result(
        self, backend_kwargs: dict, correlation_batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Get refine template result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        correlation_batch_size : int
            Number of orientations to process at once. Defaults to 32.

        Returns
        -------
        dict[str, np.ndarray]
            The result of the refine template program.
        """
        # pylint: disable=duplicate-code
        result: dict[str, np.ndarray] = {}
        result = core_refine_template(
            batch_size=correlation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            **backend_kwargs,
            frc=True,
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}

        return result

    def refine_result_to_dataframe(
        self, output_dataframe_path: str, result: dict[str, np.ndarray]
    ) -> None:
        """Convert refine template result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        result : dict[str, np.ndarray]
            The result of the refine template program.
        """
        # pylint: disable=duplicate-code
        df_refined = self.particle_stack._df.copy()  # pylint: disable=protected-access

        # x and y positions
        pos_offset_y = result["refined_pos_y"]
        pos_offset_x = result["refined_pos_x"]
        pos_offset_y_ang = pos_offset_y * df_refined["pixel_size"]
        pos_offset_x_ang = pos_offset_x * df_refined["pixel_size"]

        df_refined["refined_pos_y"] = pos_offset_y + df_refined["pos_y"]
        df_refined["refined_pos_x"] = pos_offset_x + df_refined["pos_x"]
        df_refined["refined_pos_y_img"] = pos_offset_y + df_refined["pos_y_img"]
        df_refined["refined_pos_x_img"] = pos_offset_x + df_refined["pos_x_img"]
        df_refined["refined_pos_y_img_angstrom"] = (
            pos_offset_y_ang + df_refined["pos_y_img_angstrom"]
        )
        df_refined["refined_pos_x_img_angstrom"] = (
            pos_offset_x_ang + df_refined["pos_x_img_angstrom"]
        )

        # Euler angles
        df_refined["refined_psi"] = result["refined_euler_angles"][:, 2]
        df_refined["refined_theta"] = result["refined_euler_angles"][:, 1]
        df_refined["refined_phi"] = result["refined_euler_angles"][:, 0]

        # Defocus
        # Check if refined_relative_defocus already exists in the dataframe
        if "refined_relative_defocus" in df_refined.columns:
            df_refined["refined_relative_defocus"] = (
                result["refined_defocus_offset"]
                + df_refined["refined_relative_defocus"]
            )
        else:
            # If not, create it from relative_defocus
            df_refined["refined_relative_defocus"] = (
                result["refined_defocus_offset"] + df_refined["relative_defocus"]
            )

        # Pixel size
        df_refined["refined_pixel_size"] = (
            result["refined_pixel_size_offset"] + df_refined["pixel_size"]
        )

        # Cross-correlation statistics
        # Check if correlation statistic files exist and use them if available
        # This allows for shifts during refinement

        # if (
        #    "correlation_average_path" in df_refined.columns
        #    and "correlation_variance_path" in df_refined.columns
        # ):
        # Check if files exist for at least the first entry
        #    if (
        #        df_refined["correlation_average_path"].iloc[0]
        #        and df_refined["correlation_variance_path"].iloc[0]
        #    ):
        # Load the correlation statistics from the files
        #        correlation_average = read_mrc_to_numpy(
        #            df_refined["correlation_average_path"].iloc[0]
        #        )
        #        correlation_variance = read_mrc_to_numpy(
        #            df_refined["correlation_variance_path"].iloc[0]
        #        )
        #        df_refined["correlation_mean"] = correlation_average[
        #            df_refined["refined_pos_y"], df_refined["refined_pos_x"]
        #           ]
        #        df_refined["correlation_variance"] = correlation_variance[
        #            df_refined["refined_pos_y"], df_refined["refined_pos_x"]
        #        ]
        refined_mip = result["refined_cross_correlation"]
        refined_scaled_mip = result["refined_z_score"]
        df_refined["refined_mip"] = refined_mip
        df_refined["refined_scaled_mip"] = refined_scaled_mip

        # Reorder the columns
        df_refined = df_refined.reindex(columns=REFINED_DF_COLUMN_ORDER)

        # Save the refined DataFrame to disk
        df_refined.to_csv(output_dataframe_path)
