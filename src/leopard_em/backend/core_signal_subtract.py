"""Backend functions related to signal subtraction from micrographs."""

# Following pylint error ignored because torch.fft.* is not recognized as callable
# pylint: disable=E1102

import roma
import torch
import tqdm
from torch_fourier_slice import extract_central_slices_rfft_3d

from leopard_em.pydantic_models.utils import calculate_ctf_filter_stack_full_args

# This is assuming the Euler angles are in the ZYZ intrinsic format
EULER_ANGLE_FMT = "ZYZ"


def normalize_projection(projection: torch.Tensor) -> torch.Tensor:
    """Normalize a projection by subtracting edge mean and scaling to unit variance.

    Parameters
    ----------
    projection : torch.Tensor
        The projection to normalize. Shape (batch, h, w).

    Returns
    -------
    torch.Tensor
        The normalized projection.
    """
    # Extract edges while preserving batch dimensions
    top_edge = projection[..., 0, :]  # shape: (..., w)
    bottom_edge = projection[..., -1, :]  # shape: (..., w)
    left_edge = projection[..., 1:-1, 0]  # shape: (..., h-2)
    right_edge = projection[..., 1:-1, -1]  # shape: (..., h-2)
    edge_pixels = torch.concatenate(
        [top_edge, bottom_edge, left_edge, right_edge], dim=-1
    )

    # Subtract the edge pixel mean
    projection = projection - edge_pixels.mean(dim=-1)[..., None, None]

    # Calculate variance and normalize
    variance = torch.var(projection, dim=(-2, -1), keepdim=True)
    normalized_projection = projection / torch.sqrt(variance)

    return normalized_projection


def calculate_valid_regions(
    position_y: int,
    position_x: int,
    image_shape: tuple[int, int],
    projection_shape: tuple[int, int],
) -> tuple[slice, slice, slice, slice]:
    """Calculate valid regions for image and projection for boundary-safe subtraction.

    Parameters
    ----------
    position_y : int
        Y position (top coordinate) of the projection in the image.
    position_x : int
        X position (left coordinate) of the projection in the image.
    image_shape : tuple[int, int]
        Shape of the image (height, width).
    projection_shape : tuple[int, int]
        Shape of the projection (height, width).

    Returns
    -------
    tuple[slice, slice, slice, slice]
        Slices for image and projection regions:
        (img_y_slice, img_x_slice, proj_y_slice, proj_x_slice)
    """
    im_h, im_w = image_shape
    proj_h, proj_w = projection_shape

    # Calculate valid regions for image
    y_start = max(0, position_y)
    y_end = min(im_h, position_y + proj_h)
    x_start = max(0, position_x)
    x_end = min(im_w, position_x + proj_w)

    # Calculate corresponding regions in the projection
    proj_y_start = max(0, -position_y)
    proj_y_end = proj_h - max(0, (position_y + proj_h) - im_h)
    proj_x_start = max(0, -position_x)
    proj_x_end = proj_w - max(0, (position_x + proj_w) - im_w)

    # Create slices for easier indexing
    img_y_slice = slice(y_start, y_end)
    img_x_slice = slice(x_start, x_end)
    proj_y_slice = slice(proj_y_start, proj_y_end)
    proj_x_slice = slice(proj_x_start, proj_x_end)

    return img_y_slice, img_x_slice, proj_y_slice, proj_x_slice


def extract_and_process_projection(
    template_dft: torch.Tensor,
    rot_matrix: torch.Tensor,
    ctf_filter: torch.Tensor,
    proj_filter: torch.Tensor,
) -> torch.Tensor:
    """Extract and process a projection from the template.

    Parameters
    ----------
    template_dft : torch.Tensor
        The template volume in Fourier space. Shape (d, h, w//2+1).
    rot_matrix : torch.Tensor
        The rotation matrix to use for the projection. Shape (1, 3, 3).
    ctf_filter : torch.Tensor
        The CTF filter to apply to the projection. Shape (1, 1, h, w//2+1).
    proj_filter : torch.Tensor
        The projective filter to apply to the projection. Shape (1, h, w//2+1).

    Returns
    -------
    torch.Tensor
        The processed and normalized projection. Shape (1, h, w).
    """
    # Extract template dimensions from the template_dft tensor
    _, template_h, template_w_half = template_dft.shape

    # Extract Fourier slice from the template
    # Input: template_dft (d, h, w//2+1), rot_matrix (1, 3, 3)
    # Output: fourier_slice (1, h, w//2+1)
    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        image_shape=(template_h,) * 3,
        rotation_matrices=rot_matrix,
    )

    # Apply required transformations
    # Shape: (1, h, w//2+1)
    fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
    fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component
    fourier_slice *= -1  # flip contrast

    # Apply combined filters
    # ctf_filter: (1, 1, h, w//2+1), proj_filter: (1, h, w//2+1)
    # Combined shape after broadcasting: (1, h, w//2+1)
    combined_filter = proj_filter * ctf_filter
    fourier_slice *= combined_filter

    # Convert to real space
    # fourier_slice (1, h, w//2+1) -> projection (1, h, w)
    projection = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    projection = torch.fft.ifftshift(projection, dim=(-2, -1))

    # Normalize the projection - input/output shape: (1, h, w)
    normalized_projection = normalize_projection(projection)

    return normalized_projection


def core_signal_subtract(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,
    defocus_u: torch.Tensor,
    defocus_v: torch.Tensor,
    defocus_angle: torch.Tensor,
    euler_angles: torch.Tensor,
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    device: torch.device | list[torch.device] = None,
) -> torch.Tensor:
    """Subtract the signal of particle projections from a micrograph.

    Parameters
    ----------
    image_dft : torch.Tensor
        The preprocessed micrograph in Fourier space. Shape (H, W//2+1).
    template_dft : torch.Tensor
        The template volume in Fourier space. Shape (d, h, w//2+1).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filters : torch.Tensor
        The filters to apply to each projection. Shape (N, h, w//2+1).
    defocus_u : torch.Tensor
        The defocus along the major axis for each particle. Shape (N,).
    defocus_v : torch.Tensor
        The defocus along the minor axis for each particle. Shape (N,).
    defocus_angle : torch.Tensor
        The defocus angle for each particle. Shape (N,).
    euler_angles : torch.Tensor
        The Euler angles (phi, theta, psi) for each particle. Shape (N, 3).
    positions_x : torch.Tensor
        The x positions of particles (left coordinate of template). Shape (N,).
    positions_y : torch.Tensor
        The y positions of particles (top coordinate of template). Shape (N,).
    device : torch.device | list[torch.device], optional
        The device or list of devices to use, by default None

    Returns
    -------
    torch.Tensor
        The signal-subtracted micrograph. Shape (H, W).
    """
    # If no device specified, use the device gpu 0
    if device is None:
        device = torch.device("cuda:0")

    # Convert to list if single device
    if not isinstance(device, list):
        device = [device]

    # Choose first device for operations
    primary_device = device[0]

    # Convert image back to real space for subtraction
    # image_dft (H, W//2+1) -> image (H, W)
    image = torch.fft.irfftn(image_dft)

    # Create a copy of the image for subtraction - Shape: (H, W)
    subtracted_image = image.clone()

    # Get the shape of the template
    # template_dft shape: (d, h, w//2+1)
    _, template_h, template_w = template_dft.shape
    # account for RFFT in template width
    template_w = 2 * (template_w - 1)  # Convert from Fourier to real space width

    # Create rotation matrices from Euler angles
    # euler_angles (N, 3) -> rotation_matrices (N, 3, 3)
    rotation_matrices = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, euler_angles, degrees=True, device=primary_device
    )

    # Calculate CTF filters for all particles at once
    # Output shape: (N, 1, h, w//2+1)
    ctf_filters = calculate_ctf_filter_stack_full_args(
        defocus_u=defocus_u,  # in Angstrom
        defocus_v=defocus_v,  # in Angstrom
        astigmatism_angle=defocus_angle,  # in degrees
        defocus_offsets=torch.tensor([0.0]),  # no offset for subtraction
        pixel_size_offsets=torch.tensor([0.0]),  # no offset for subtraction
        **ctf_kwargs,
    )

    num_particles = len(positions_x)

    # Loop through each particle and subtract its projection
    pbar = tqdm.tqdm(
        range(num_particles),
        total=num_particles,
        desc="Subtracting particle projections",
    )

    for i in pbar:
        # Get particle position (top-left corner of the template in the image)
        pos_x, pos_y = int(positions_x[i].item()), int(positions_y[i].item())

        # Get current particle's rotation matrix
        # (3, 3) -> (1, 3, 3) - Add batch dim for extract_central_slices_rfft_3d
        rot_matrix = rotation_matrices[i].unsqueeze(0)

        # Get the CTF filter for this particle - shape: (1, 1, h, w//2+1)
        ctf_filter = ctf_filters[i]

        # Get the projective filter for this particle
        # (h, w//2+1) -> (1, h, w//2+1) - Add batch dim to match fourier_slice
        proj_filter = projective_filters[i].unsqueeze(0)

        # Extract and process the projection - Output: (1, h, w)
        projection = extract_and_process_projection(
            template_dft=template_dft,
            rot_matrix=rot_matrix,
            ctf_filter=ctf_filter,
            proj_filter=proj_filter,
        )

        # Calculate valid regions for boundary-safe subtraction
        img_y_slice, img_x_slice, proj_y_slice, proj_x_slice = calculate_valid_regions(
            position_y=pos_y,
            position_x=pos_x,
            image_shape=image.shape,  # (H, W)
            projection_shape=projection.shape[-2:],  # (h, w)
        )

        # Subtract the projection from the image
        # projection[0]: (h, w) - access first (and only) batch element
        subtracted_image[img_y_slice, img_x_slice] -= projection[
            0, proj_y_slice, proj_x_slice
        ]

    return subtracted_image
