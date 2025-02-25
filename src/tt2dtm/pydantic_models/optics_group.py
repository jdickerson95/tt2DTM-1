"""Microscope optics group model for micrograph parameters."""

from os import PathLike
from typing import Annotated, Optional, Union

from pydantic import Field

from tt2dtm.pydantic_models.types import BaseModel2DTM


class OpticsGroup(BaseModel2DTM):
    """Stores optics group parameters for the imaging system on a microscope.

    Currently utilizes the minimal set of parameters for calculating a
    contrast transfer function (CTF) for a given optics group. Other parameters
    for future use are included but currently unused.

    Attributes
    ----------
    label : str
        Unique string (among other optics groups) for the optics group.
    pixel_size : float
        Pixel size in Angstrom.
    voltage : float
        Voltage in kV.
    spherical_aberration : float
        Spherical aberration in mm. Default is 2.7.
    amplitude_contrast_ratio : float
        Amplitude contrast ratio as a unitless percentage in [0, 1]. Default
        is 0.07.
    phase_shift : float
        Additional phase shift of the contrast transfer function in degrees.
        Default is 0.0 degrees.
    defocus_u : float
        Defocus (underfocus) along the major axis in Angstrom.
    defocus_v : float
        Defocus (underfocus) along the minor axis in Angstrom.
    astigmatism_angle : float
        Angle of defocus astigmatism relative to the X-axis in degrees.
    ctf_B_factor : float
        B-factor to apply in the contrast transfer function in A^2. Default
        is 0.0.

    Unused Attributes:
    ------------------
    chromatic_aberration : float
        Chromatic aberration in mm. Default is ???.
    mtf_reference : str | PathLike
        Path to MTF reference file.
    mtf_values : list[float]
        list of modulation transfer functions values on evenly spaced
        resolution grid [0.0, ..., 0.5].
    beam_tilt_x : float
        Beam tilt X in mrad.
    beam_tilt_y : float
        Beam tilt Y in mrad.
    odd_zernike : list[float]
        list of odd Zernike moments.
    even_zernike : list[float]
        list of even Zernike moments.
    zernike_moments : list[float]
        list of Zernike moments.

    Methods
    -------
    model_dump()
        Returns a dictionary of the model parameters.
    """

    # Currently implemented parameters
    label: str
    pixel_size: Annotated[float, Field(ge=0.0)]
    voltage: Annotated[float, Field(ge=0.0)]
    spherical_aberration: Annotated[float, Field(ge=0.0, default=2.7)] = 2.7
    amplitude_contrast_ratio: Annotated[float, Field(ge=0.0, le=1.0, default=0.07)] = (
        0.07
    )
    phase_shift: Annotated[float, Field(default=0.0)] = 0.0
    defocus_u: float
    defocus_v: float
    astigmatism_angle: float
    ctf_B_factor: Annotated[float, Field(ge=0.0, default=0.0)] = 0.0

    chromatic_aberration: Optional[Annotated[float, Field(ge=0.0)]] = 0.0
    mtf_reference: Optional[Union[str, PathLike]] = None
    mtf_values: Optional[list[float]] = None
    beam_tilt_x: Optional[float] = None
    beam_tilt_y: Optional[float] = None
    odd_zernike: Optional[list[float]] = None
    even_zernike: Optional[list[float]] = None
    zernike_moments: Optional[list[float]] = None
