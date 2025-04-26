"""Pydantic model for running the signal subtract program."""

import os
from typing import Any, ClassVar

import torch
from pydantic import ConfigDict, field_validator

from leopard_em.backend.core_signal_subtract import core_signal_subtract
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    preprocess_image,
    setup_images_filters_particle_stack,
)
from leopard_em.utils.data_io import (
    load_mrc_image,
    load_mrc_volume,
    write_mrc_from_numpy,
    write_mrc_from_tensor,
)


class SignalSubtractManager(BaseModel2DTM):
    """Model holding parameters necessary for running the signal subtract program.

    This model handles the process of subtracting template projections from a micrograph
    at particle positions. It uses the refined Euler angles and positions if available,
    to ensure accurate template projections.

    Attributes
    ----------
    micrograph_path : str
        Path to the micrograph MRC file.
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
    micrograph : ExcludedTensor
        Image to run template matching on. Not serialized.

    Methods
    -------
    __init__(self, preload_mrc_files: bool = False, **data: Any)
        Initialize the signal subtract manager.
    make_backend_core_function_kwargs(self, prefer_refined_angles: bool = True,
                                prefer_refined_positions: bool = True) -> dict[str, Any]
        Create the kwargs for the backend signal_subtract core function.
    run_signal_subtract(self, prefer_refined_angles: bool = True,
                       prefer_refined_positions: bool = True) -> torch.Tensor
        Run the signal subtract program to create a subtracted particle image.
    save_subtracted_image(self, subtracted_image: torch.Tensor,
                                output_path: str) -> None
        Save the subtracted micrograph as an MRC file.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    micrograph_path: str
    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig

    # Excluded tensors
    micrograph: ExcludedTensor = None
    template_volume: ExcludedTensor = None

    ###########################
    ### Pydantic Validators ###
    ###########################

    @field_validator("micrograph_path")  # type: ignore
    def validate_micrograph_path(cls, v) -> str:
        """Ensure the micrograph file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for micrograph does not exist.")

        return str(v)

    @field_validator("template_volume_path")  # type: ignore
    def validate_template_volume_path(cls, v) -> str:
        """Ensure the template volume file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for template volume does not exist.")

        return str(v)

    def __init__(self, preload_mrc_files: bool = False, **data: Any):
        super().__init__(**data)

        if preload_mrc_files:
            # Load the data from the MRC files
            self.micrograph = load_mrc_image(self.micrograph_path)
            self.template_volume = load_mrc_volume(self.template_volume_path)

    def make_backend_core_function_kwargs(
        self, prefer_refined_angles: bool = True, prefer_refined_positions: bool = True
    ) -> dict[str, Any]:
        """Create the kwargs for the backend signal subtract core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use the refined angles from the particle stack. Defaults to
            True.
        prefer_refined_positions : bool
            Whether to use the refined positions from the particle stack. Defaults to
            True.

        Returns
        -------
        dict[str, Any]
            Dictionary of keyword arguments for the core signal subtract function.
        """
        # Ensure the micrograph and template are loaded and in the correct format
        if self.micrograph is None:
            self.micrograph = load_mrc_image(self.micrograph_path)
        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)

        # Ensure the micrograph and template are both Tensors before proceeding
        if not isinstance(self.micrograph, torch.Tensor):
            image = torch.from_numpy(self.micrograph)
        else:
            image = self.micrograph

        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        # Fourier transform the image (RFFT, unshifted)
        image_dft = torch.fft.rfftn(image)  # pylint: disable=E1102
        image_dft[0, 0] = 0 + 0j  # zero out the constant term

        # Get the bandpass filter individually
        bp_config = self.preprocessing_filters.bandpass_filter
        bandpass_filter = bp_config.calculate_bandpass_filter(image_dft.shape)

        # Calculate the cumulative filters for both the image and the template.
        cumulative_filter_image = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=image_dft.shape,
        )

        # Apply the pre-processing and normalization
        image_preprocessed_dft = preprocess_image(
            image_rfft=image_dft,
            cumulative_fourier_filters=cumulative_filter_image,
            bandpass_filter=bandpass_filter,
        )

        # get projective filters
        (_, template_dft, projective_filters) = setup_images_filters_particle_stack(
            particle_stack=self.particle_stack,
            preprocessing_filters=self.preprocessing_filters,
            template=template,
        )

        # Get Euler angles using particle stack's method
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)

        # Get particle positions using particle stack's method
        pos_x, pos_y = self.particle_stack.get_positions(prefer_refined_positions)

        defocus_u, defocus_v = self.particle_stack.get_absolute_defocus()
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])

        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            particle_stack=self.particle_stack,
            template_shape=(template.shape[-2], template.shape[-1]),
        )

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft": template_dft,
            "ctf_kwargs": ctf_kwargs,
            "projective_filters": projective_filters,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "defocus_angle": defocus_angle,
            "euler_angles": euler_angles,
            "positions_x": pos_x,
            "positions_y": pos_y,
            "device": self.computational_config.gpu_devices,
        }

    def run_signal_subtract(
        self, prefer_refined_angles: bool = True, prefer_refined_positions: bool = True
    ) -> torch.Tensor:
        """Run the signal subtract program.

        This method preprocesses the micrograph and the template volume according to
        the preprocessing filters. It extracts particle positions and orientations from
        the particle stack , projects the template at each particle position
        and subtracts these projections from the micrograph.

        Parameters
        ----------
        prefer_refined_angles : bool, optional
            Whether to use the refined angles, by default True.
        prefer_refined_positions : bool, optional
            Whether to use the refined positions, by default True.

        Returns
        -------
        torch.Tensor
            The signal-subtracted micrograph.
        """
        # Create kwargs for backend function call
        backend_kwargs = self.make_backend_core_function_kwargs(
            prefer_refined_angles=prefer_refined_angles,
            prefer_refined_positions=prefer_refined_positions,
        )

        # Run the core signal subtract function
        subtracted_image = core_signal_subtract(**backend_kwargs)

        # Return the subtracted image
        return subtracted_image

    def save_subtracted_image(
        self, subtracted_image: torch.Tensor, output_path: str
    ) -> None:
        """Save the subtracted micrograph as an MRC file.

        Parameters
        ----------
        subtracted_image : torch.Tensor
            The signal-subtracted micrograph to save.
        output_path : str
            Path where to save the subtracted micrograph. If a directory is provided,
            a filename will be generated based on the original micrograph name.
        """
        # Check if output_path is a directory
        if os.path.isdir(output_path):
            # Get base name of original micrograph
            micrograph_name = os.path.basename(self.micrograph_path)
            micrograph_name = os.path.splitext(micrograph_name)[0]

            # Create output filename
            output_path = os.path.join(output_path, f"{micrograph_name}_subtracted.mrc")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Check the type of subtracted_image and use the appropriate function
        if isinstance(subtracted_image, torch.Tensor):
            write_mrc_from_tensor(subtracted_image, output_path, overwrite=True)
        else:
            write_mrc_from_numpy(subtracted_image, output_path, overwrite=True)


# Usage example:
