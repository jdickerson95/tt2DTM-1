"""
Optimize the pixel sizes of the template and image.
"""

from typing import Optional, Union

import torch
from ttsim3d.models import Simulator, SimulatorConfig

from tt2dtm.pydantic_models import MatchTemplateManager
from tt2dtm.utils.data_io import load_mrc_image

YAML_CONFIG_PATH = "match_template_manager_example.yaml"
PDB_PATH = "parsed_6Q8Y_whole_LSU_match3.pdb"
ORIENTATION_BATCH_SIZE = 64
CROP_FACTOR = 4
GPU_IDS = None


def crop_center(image: torch.Tensor, division_factor: int) -> torch.Tensor:
    """
    Crop the center portion of an image based on a division factor.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor to crop
    division_factor : int
        Factor by which to divide the image dimensions

    Returns
    -------
    torch.Tensor
        Cropped center portion of the input image
    """
    h, w = image.shape
    new_h = h // division_factor
    new_w = w // division_factor
    start_h = h // 2 - new_h // 2
    start_w = w // 2 - new_w // 2
    return image[start_h : start_h + new_h, start_w : start_w + new_w]


def run_ttsim3d(
    px_value: float, gpu_ids: Optional[Union[int, list[int]]] = None
) -> torch.Tensor:
    """
    Run ttsim3d simulation with given pixel size.

    Parameters
    ----------
    px_value : float
        Pixel size value in Angstroms
    gpu_ids : Optional[Union[int, list[int]]], optional
        GPU device IDs to use for simulation, by default None

    Returns
    -------
    torch.Tensor
        Simulated template volume
    """
    sim_conf = SimulatorConfig(
        voltage=300.0,
        apply_dose_weighting=True,
        dose_start=0.0,
        dose_end=50.0,
        dose_filter_modify_signal="rel_diff",
        upsampling=-1,
    )

    sim = Simulator(
        pdb_filepath=PDB_PATH,
        pixel_spacing=px_value,
        volume_shape=(512, 512, 512),
        b_factor_scaling=0.5,
        additional_b_factor=0,
        simulator_config=sim_conf,
    )
    return sim.run(gpu_ids=gpu_ids)


def evaluate_peaks(mtm: MatchTemplateManager, method: str = "match") -> float:
    """
    Evaluate peaks using either match_template or refine_template.

    Parameters
    ----------
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    method : str, optional
        Method to use for evaluation, either "match" or "refine", by default "match"

    Returns
    -------
    float
        Mean SNR value from peak detection
    """
    if method == "match":
        mtm.run_match_template(ORIENTATION_BATCH_SIZE, do_result_export=False)
    elif method == "refine":
        # TODO: Implement refine_template
        raise NotImplementedError("refine_template not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}")

    peaks = mtm.results_to_dataframe()
    return peaks["scaled_mip"].mean()


def evaluate_template_px(
    px_value: float, mtm: MatchTemplateManager, method: str = "match"
) -> float:
    """
    Evaluate a single template pixel size and return the mean SNR.

    Parameters
    ----------
    px_value : float
        Template pixel size to evaluate
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    method : str, optional
        Method to use for evaluation, by default "match"

    Returns
    -------
    float
        Mean SNR value for the given pixel size
    """
    template_volume = run_ttsim3d(px_value=px_value, gpu_ids=GPU_IDS)
    mtm.template_volume = template_volume
    return evaluate_peaks(mtm, method)


def evaluate_micrograph_px(
    px_value: float, mtm: MatchTemplateManager, method: str = "match"
) -> float:
    """
    Evaluate SNR for a given micrograph pixel size, keeping template constant.

    Parameters
    ----------
    px_value : float
        Micrograph pixel size to evaluate
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    method : str, optional
        Method to use for evaluation, by default "match"

    Returns
    -------
    float
        Mean SNR value for the given pixel size
    """
    original_px = mtm.optics_group.pixel_size
    mtm.optics_group.pixel_size = px_value
    snr = evaluate_peaks(mtm, method)
    mtm.optics_group.pixel_size = original_px
    return snr


def optimize_pixel_size_grid(
    mtm: MatchTemplateManager,
    initial_px: float,
    method: str = "match",
    coarse_range: float = 0.05,
    coarse_step: float = 0.01,
    fine_range: float = 0.005,
    fine_step: float = 0.001,
) -> float:
    """
    Two-stage template pixel size optimization using grid search.

    Parameters
    ----------
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    initial_px : float
        Initial pixel size guess
    method : str, optional
        Method to use for evaluation, by default "match"
    coarse_range : float, optional
        Range for coarse search, by default 0.05
    coarse_step : float, optional
        Step size for coarse search, by default 0.01
    fine_range : float, optional
        Range for fine search, by default 0.005
    fine_step : float, optional
        Step size for fine search, by default 0.001

    Returns
    -------
    float
        Optimal pixel size found
    """
    coarse_px_values = torch.arange(
        initial_px - coarse_range, initial_px + coarse_range + 1e-10, coarse_step
    )

    best_snr = float("-inf")
    best_px = initial_px

    print("Starting coarse search...")
    for px in coarse_px_values:
        snr = evaluate_template_px(px.item(), mtm, method)
        print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")
        if snr > best_snr:
            best_snr = snr
            best_px = px.item()

    fine_px_values = torch.arange(
        best_px - fine_range, best_px + fine_range + 1e-10, fine_step
    )

    print("\nStarting fine search...")
    for px in fine_px_values:
        snr = evaluate_template_px(px.item(), mtm, method)
        print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")
        if snr > best_snr:
            best_snr = snr
            best_px = px.item()

    print(f"\nOptimal pixel size: {best_px:.3f} Å with SNR: {best_snr:.3f}")
    return best_px


def optimize_micrograph_px_grid(
    mtm: MatchTemplateManager,
    initial_px: float,
    method: str = "match",
    coarse_range: float = 0.05,
    coarse_step: float = 0.01,
    fine_range: float = 0.005,
    fine_step: float = 0.001,
) -> float:
    """
    Two-stage micrograph pixel size optimization using grid search.

    Parameters
    ----------
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    initial_px : float
        Initial pixel size guess
    method : str, optional
        Method to use for evaluation, by default "match"
    coarse_range : float, optional
        Range for coarse search, by default 0.05
    coarse_step : float, optional
        Step size for coarse search, by default 0.01
    fine_range : float, optional
        Range for fine search, by default 0.005
    fine_step : float, optional
        Step size for fine search, by default 0.001

    Returns
    -------
    float
        Optimal pixel size found
    """
    coarse_px_values = torch.arange(
        initial_px - coarse_range, initial_px + coarse_range + 1e-10, coarse_step
    )

    best_snr = float("-inf")
    best_px = initial_px

    print("Starting coarse micrograph pixel size search...")
    for px in coarse_px_values:
        snr = evaluate_micrograph_px(px.item(), mtm, method)
        print(f"Micrograph pixel size: {px:.3f}, SNR: {snr:.3f}")
        if snr > best_snr:
            best_snr = snr
            best_px = px.item()

    fine_px_values = torch.arange(
        best_px - fine_range, best_px + fine_range + 1e-10, fine_step
    )

    print("\nStarting fine micrograph pixel size search...")
    for px in fine_px_values:
        snr = evaluate_micrograph_px(px.item(), mtm, method)
        print(f"Micrograph pixel size: {px:.3f}, SNR: {snr:.3f}")
        if snr > best_snr:
            best_snr = snr
            best_px = px.item()

    print(f"\nOptimal micrograph pixel size: {best_px:.3f} Å with SNR: {best_snr:.3f}")
    return best_px


def optimize_pixel_size_gradient(
    mtm: MatchTemplateManager,
    initial_px: float,
    method: str = "match",
    learning_rate: float = 0.001,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
    px_bounds: tuple = (None, None),
) -> float:
    """
    Optimize template pixel size using PyTorch optimizer.

    Parameters
    ----------
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    initial_px : float
        Initial pixel size guess
    method : str, optional
        Method to use for evaluation, by default "match"
    learning_rate : float, optional
        Learning rate for optimizer, by default 0.001
    max_iterations : int, optional
        Maximum number of iterations, by default 20
    tolerance : float, optional
        Convergence tolerance, by default 1e-4
    px_bounds : tuple, optional
        (min, max) bounds for pixel size, by default (None, None)

    Returns
    -------
    float
        Optimal pixel size found
    """
    px = torch.nn.Parameter(torch.tensor(initial_px))
    optimizer = torch.optim.Adam([px], lr=learning_rate)

    best_px = initial_px
    best_snr = evaluate_template_px(initial_px, mtm, method)

    print(f"Initial pixel size: {initial_px:.3f} Å, SNR: {best_snr:.3f}")

    prev_px = initial_px
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        snr = -evaluate_template_px(px.item(), mtm, method)
        snr.backward()
        optimizer.step()

        with torch.no_grad():
            if px_bounds[0] is not None:
                px.clamp_(min=px_bounds[0])
            if px_bounds[1] is not None:
                px.clamp_(max=px_bounds[1])

        current_snr = -snr.item()
        print(
            f"Iteration {iteration + 1}: px = {px.item():.3f} Å, SNR = {current_snr:.3f}"
        )

        if current_snr > best_snr:
            best_snr = current_snr
            best_px = px.item()

        if abs(px.item() - prev_px) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break

        prev_px = px.item()

    print(f"\nOptimal pixel size: {best_px:.3f} Å with SNR: {best_snr:.3f}")
    return best_px


def optimize_micrograph_px_gradient(
    mtm: MatchTemplateManager,
    initial_px: float,
    method: str = "match",
    learning_rate: float = 0.001,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
    px_bounds: tuple = (None, None),
) -> float:
    """
    Optimize micrograph pixel size using PyTorch optimizer.

    Parameters
    ----------
    mtm : MatchTemplateManager
        Manager object containing template matching parameters and methods
    initial_px : float
        Initial pixel size guess
    method : str, optional
        Method to use for evaluation, by default "match"
    learning_rate : float, optional
        Learning rate for optimizer, by default 0.001
    max_iterations : int, optional
        Maximum number of iterations, by default 20
    tolerance : float, optional
        Convergence tolerance, by default 1e-4
    px_bounds : tuple, optional
        (min, max) bounds for pixel size, by default (None, None)

    Returns
    -------
    float
        Optimal pixel size found
    """
    px = torch.nn.Parameter(torch.tensor(initial_px))
    optimizer = torch.optim.Adam([px], lr=learning_rate)

    best_px = initial_px
    best_snr = evaluate_micrograph_px(initial_px, mtm, method)

    print(f"Initial micrograph pixel size: {initial_px:.3f} Å, SNR: {best_snr:.3f}")

    prev_px = initial_px
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        snr = -evaluate_micrograph_px(px.item(), mtm, method)
        snr.backward()
        optimizer.step()

        with torch.no_grad():
            if px_bounds[0] is not None:
                px.clamp_(min=px_bounds[0])
            if px_bounds[1] is not None:
                px.clamp_(max=px_bounds[1])

        current_snr = -snr.item()
        print(
            f"Iteration {iteration + 1}: px = {px.item():.3f} Å, SNR = {current_snr:.3f}"
        )

        if current_snr > best_snr:
            best_snr = current_snr
            best_px = px.item()

        if abs(px.item() - prev_px) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break

        prev_px = px.item()

    print(f"\nOptimal micrograph pixel size: {best_px:.3f} Å with SNR: {best_snr:.3f}")
    return best_px


def main_grid() -> None:
    """
    Main function using grid search optimization.

    Loads micrograph data, optimizes template and micrograph pixel sizes using
    grid search method.
    """
    mtm = MatchTemplateManager(YAML_CONFIG_PATH)

    # Load and crop the micrograph
    micrograph_data = load_mrc_image(mtm.micrograph_path)
    cropped_micrograph_data = crop_center(micrograph_data, CROP_FACTOR)
    mtm.micrograph = cropped_micrograph_data

    # Run initial match_template to get peak positions
    mtm.run_match_template(ORIENTATION_BATCH_SIZE, do_result_export=False)
    initial_peaks = mtm.results_to_dataframe()

    # Optimize template pixel size
    initial_template_px = mtm.optics_group.pixel_size
    optimal_template_px = optimize_pixel_size_grid(
        mtm,
        initial_template_px,
        method="refine",  # Will use refinement once implemented
    )

    # Simulate template with optimal pixel size
    template = run_ttsim3d(px_value=optimal_template_px, gpu_ids=None)
    mtm.template_volume = template

    # Optimize micrograph pixel size
    initial_micrograph_px = mtm.optics_group.pixel_size
    optimal_micrograph_px = optimize_micrograph_px_grid(
        mtm,
        initial_micrograph_px,
        method="refine",  # Will use refinement once implemented
    )

    # Set final optimal pixel size
    mtm.optics_group.pixel_size = optimal_micrograph_px


def main_gradient() -> None:
    """
    Main function using gradient-based optimization.

    Loads micrograph data, optimizes template and micrograph pixel sizes using
    gradient-based optimization method.
    """
    mtm = MatchTemplateManager(YAML_CONFIG_PATH)

    # Load and crop the micrograph
    micrograph_data = load_mrc_image(mtm.micrograph_path)
    cropped_micrograph_data = crop_center(micrograph_data, CROP_FACTOR)
    mtm.micrograph = cropped_micrograph_data

    # Run initial match_template to get peak positions
    mtm.run_match_template(ORIENTATION_BATCH_SIZE, do_result_export=False)
    initial_peaks = mtm.results_to_dataframe()

    # Optimize template pixel size
    initial_template_px = mtm.optics_group.pixel_size
    px_bounds = (initial_template_px - 0.05, initial_template_px + 0.05)
    optimal_template_px = optimize_pixel_size_gradient(
        mtm,
        initial_template_px,
        method="refine",  # Will use refinement once implemented
        px_bounds=px_bounds,
    )

    # Simulate template with optimal pixel size
    template = run_ttsim3d(px_value=optimal_template_px, gpu_ids=None)
    mtm.template_volume = template

    # Optimize micrograph pixel size
    initial_micrograph_px = mtm.optics_group.pixel_size
    px_bounds = (initial_micrograph_px - 0.05, initial_micrograph_px + 0.05)
    optimal_micrograph_px = optimize_micrograph_px_gradient(
        mtm,
        initial_micrograph_px,
        method="refine",  # Will use refinement once implemented
        px_bounds=px_bounds,
    )

    # Set final optimal pixel size
    mtm.optics_group.pixel_size = optimal_micrograph_px


if __name__ == "__main__":
    # Choose which optimization method to use
    # main_gradient()  # or main_grid()
    main_grid()
