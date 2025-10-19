#!/usr/bin/env python3
"""
Helper script to calculate the number of cross correlations and pixels for 2DTM programs.

This script takes a YAML configuration file for match_template, refine_template, or inspect_peaks 
programs and calculates:
1. Number of cross correlations that will be computed (angles × defocus × pixel_size)
2. Number of pixels in the search space

Usage:
    python calculate_correlations.py config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from leopard_em.pydantic_models.managers import (
    MatchTemplateManager,
    RefineTemplateManager, 
    InspectPeaksManager
)
from leopard_em.utils.data_io import load_mrc_image


def detect_program_type(yaml_path: str) -> str:
    """Detect which program type the YAML configuration is for.
    
    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    str
        Program type: 'match_template', 'refine_template', or 'inspect_peaks'
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for match_template specific fields
    if 'micrograph_path' in config and 'orientation_search_config' in config:
        return 'match_template'
    
    # Check for refine/inspect template specific fields (both have particle_stack)
    elif 'particle_stack' in config:
        # Distinguish between refine and inspect by looking for refinement configs
        if ('defocus_refinement_config' in config or 
            'orientation_refinement_config' in config or
            'pixel_size_refinement_config' in config):
            return 'refine_template'
        else:
            return 'inspect_peaks'
    
    else:
        raise ValueError(f"Could not determine program type from configuration file: {yaml_path}")


def calculate_correlations_from_config(yaml_path: str) -> dict:
    """Calculate correlation statistics from 2DTM configuration.
    
    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    dict
        Dictionary containing correlation and pixel statistics
    """
    program_type = detect_program_type(yaml_path)
    
    # Load the appropriate manager
    try:
        if program_type == 'match_template':
            manager = MatchTemplateManager.from_yaml(yaml_path)
        elif program_type == 'refine_template':
            manager = RefineTemplateManager.from_yaml(yaml_path)
        elif program_type == 'inspect_peaks':
            manager = InspectPeaksManager.from_yaml(yaml_path)
        else:
            raise ValueError(f"Unknown program type: {program_type}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {yaml_path}: {e}")
    
    if program_type == 'match_template':
        return _calculate_match_template_stats(manager, yaml_path, program_type)
    elif program_type == 'refine_template':
        return _calculate_refine_template_stats(manager, yaml_path, program_type)
    elif program_type == 'inspect_peaks':
        return _calculate_inspect_peaks_stats(manager, yaml_path, program_type)


def _calculate_match_template_stats(manager, yaml_path: str, program_type: str) -> dict:
    """Calculate statistics for match template program."""
    # Get orientation search parameters
    orientation_config = manager.orientation_search_config
    euler_angles = orientation_config.euler_angles
    num_angles = euler_angles.shape[0]
    
    # Get defocus search parameters
    defocus_config = manager.defocus_search_config
    defocus_values = defocus_config.defocus_values
    num_defocus = defocus_values.shape[0]
    
    # Pixel size search is not typically used in match template, but check anyway
    num_pixel_sizes = 1  # Match template typically doesn't search pixel size
    
    # Total cross correlations
    total_cross_correlations = num_angles * num_defocus * num_pixel_sizes
    
    # Get micrograph information for pixel count
    micrograph_path = manager.micrograph_path
    try:
        micrograph = load_mrc_image(micrograph_path)
        micrograph_height, micrograph_width = micrograph.shape
        total_pixels = micrograph_height * micrograph_width
    except Exception as e:
        print(f"Warning: Could not load micrograph {micrograph_path}: {e}")
        micrograph_height = micrograph_width = total_pixels = None
    
    return {
        'configuration_file': yaml_path,
        'program_type': program_type,
        'num_angles': num_angles,
        'num_defocus_values': num_defocus,
        'num_pixel_sizes': num_pixel_sizes,
        'total_cross_correlations': total_cross_correlations,
        'micrograph_path': micrograph_path,
        'micrograph_dimensions': (micrograph_height, micrograph_width) if micrograph_height else None,
        'total_pixels': total_pixels,
        'search_space_description': f"Full micrograph: {micrograph_height} × {micrograph_width} pixels"
    }


def _calculate_refine_template_stats(manager, yaml_path: str, program_type: str) -> dict:
    """Calculate statistics for refine template program."""
    # Get the particle stack information
    particle_stack = manager.particle_stack
    num_particles = particle_stack.num_particles
    
    # Get orientation refinement parameters
    orientation_config = manager.orientation_refinement_config
    euler_angle_offsets = orientation_config.euler_angles_offsets
    num_angles = euler_angle_offsets.shape[0]
    
    # Get defocus refinement parameters
    defocus_config = manager.defocus_refinement_config
    defocus_offsets = defocus_config.defocus_values
    num_defocus = defocus_offsets.shape[0]
    
    # Get pixel size refinement parameters
    pixel_size_config = manager.pixel_size_refinement_config
    pixel_size_offsets = pixel_size_config.pixel_size_values
    num_pixel_sizes = pixel_size_offsets.shape[0]
    
    # Cross correlations per particle
    cross_correlations_per_particle = num_angles * num_defocus * num_pixel_sizes
    
    # Total cross correlations for all particles
    total_cross_correlations = num_particles * cross_correlations_per_particle
    
    # Get template and particle box information for pixel calculation
    template_height, template_width = particle_stack.original_template_size
    box_height, box_width = particle_stack.extracted_box_size
    
    # For refine template, the search space is the "valid" correlation region
    # This is (extracted_box_size - original_template_size + 1)^2
    search_height = box_height - template_height + 1
    search_width = box_width - template_width + 1
    pixels_per_particle = search_height * search_width
    total_pixels = num_particles * pixels_per_particle
    
    return {
        'configuration_file': yaml_path,
        'program_type': program_type,
        'num_particles': num_particles,
        'num_angles': num_angles,
        'num_defocus_values': num_defocus,
        'num_pixel_sizes': num_pixel_sizes,
        'cross_correlations_per_particle': cross_correlations_per_particle,
        'total_cross_correlations': total_cross_correlations,
        'template_size': (template_height, template_width),
        'extracted_box_size': (box_height, box_width),
        'search_region_size': (search_height, search_width),
        'pixels_per_particle': pixels_per_particle,
        'total_pixels': total_pixels,
        'unique_micrographs': particle_stack._df['micrograph_path'].nunique(),
        'search_space_description': f"Valid correlation regions: {search_height} × {search_width} pixels per particle"
    }


def _calculate_inspect_peaks_stats(manager, yaml_path: str, program_type: str) -> dict:
    """Calculate statistics for inspect peaks program."""
    # Get the particle stack information
    particle_stack = manager.particle_stack
    num_particles = particle_stack.num_particles
    
    # For inspect peaks, no search is performed - just analysis at fixed positions
    num_angles = 1
    num_defocus = 1
    num_pixel_sizes = 1
    
    # Cross correlations per particle
    cross_correlations_per_particle = num_angles * num_defocus * num_pixel_sizes
    
    # Total cross correlations for all particles
    total_cross_correlations = num_particles * cross_correlations_per_particle
    
    # Get template and particle box information
    template_height, template_width = particle_stack.original_template_size
    box_height, box_width = particle_stack.extracted_box_size
    
    # For inspect peaks, like refine template, the pixel calculation is the valid correlation region
    # This is (extracted_box_size - original_template_size + 1)^2
    search_height = box_height - template_height + 1
    search_width = box_width - template_width + 1
    pixels_per_particle = search_height * search_width
    total_pixels = num_particles * pixels_per_particle
    
    return {
        'configuration_file': yaml_path,
        'program_type': program_type,
        'num_particles': num_particles,
        'num_angles': num_angles,
        'num_defocus_values': num_defocus,
        'num_pixel_sizes': num_pixel_sizes,
        'cross_correlations_per_particle': cross_correlations_per_particle,
        'total_cross_correlations': total_cross_correlations,
        'template_size': (template_height, template_width),
        'extracted_box_size': (box_height, box_width),
        'search_region_size': (search_height, search_width),
        'pixels_per_particle': pixels_per_particle,
        'total_pixels': total_pixels,
        'unique_micrographs': particle_stack._df['micrograph_path'].nunique(),
        'search_space_description': f"Valid correlation regions: {search_height} × {search_width} pixels per particle"
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Calculate cross correlations and pixel counts for 2DTM programs"
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' does not exist.")
        sys.exit(1)
    
    try:
        # Calculate statistics
        stats = calculate_correlations_from_config(str(config_path))
        program_type = stats['program_type'].upper().replace('_', ' ')
        
        # Print results
        print("="*60)
        print(f"{program_type} CORRELATION CALCULATION")
        print("="*60)
        print(f"Configuration file: {stats['configuration_file']}")
        print(f"Program type: {stats['program_type']}")
        print()
        
        # Program-specific information
        if stats['program_type'] == 'match_template':
            print("MICROGRAPH INFORMATION:")
            print(f"  Micrograph path: {stats['micrograph_path']}")
            if stats['micrograph_dimensions']:
                print(f"  Micrograph dimensions: {stats['micrograph_dimensions'][0]} × {stats['micrograph_dimensions'][1]} pixels")
            print()
            
        elif stats['program_type'] in ['refine_template', 'inspect_peaks']:
            print("PARTICLE INFORMATION:")
            print(f"  Number of particles: {stats['num_particles']:,}")
            print(f"  Unique micrographs: {stats['unique_micrographs']}")
            print(f"  Template size: {stats['template_size'][0]} × {stats['template_size'][1]} pixels")
            print(f"  Extracted box size: {stats['extracted_box_size'][0]} × {stats['extracted_box_size'][1]} pixels")
            if 'search_region_size' in stats:
                print(f"  Valid search region: {stats['search_region_size'][0]} × {stats['search_region_size'][1]} pixels")
            print()
        
        print("SEARCH SPACE:")
        print(f"  Number of angles: {stats['num_angles']:,}")
        print(f"  Number of defocus values: {stats['num_defocus_values']:,}")
        print(f"  Number of pixel sizes: {stats['num_pixel_sizes']:,}")
        print()
        
        print("CROSS CORRELATIONS:")
        if stats['program_type'] == 'match_template':
            print(f"  Total cross correlations: {stats['total_cross_correlations']:,}")
        else:
            print(f"  Cross correlations per particle: {stats.get('cross_correlations_per_particle', 'N/A')}")
            print(f"  Total cross correlations: {stats['total_cross_correlations']:,}")
        print()
        
        print("PIXEL COUNTS:")
        if stats['program_type'] == 'match_template':
            print(f"  Total pixels in search space: {stats['total_pixels']:,}")
        else:
            print(f"  Pixels per particle: {stats['pixels_per_particle']:,}")
            print(f"  Total pixels in search space: {stats['total_pixels']:,}")
        print(f"  Search space: {stats['search_space_description']}")
        print()
        
        # Calculate and display total computational operations
        total_operations = stats['total_cross_correlations'] * stats['total_pixels']
        print("COMPUTATIONAL COMPLEXITY:")
        print(f"  Total correlations × total pixels: {total_operations:,}")
        print(f"  (This represents the total number of correlation operations)")
        
        if args.verbose:
            print()
            print("DETAILED INFORMATION:")
            if stats['program_type'] == 'match_template':
                print("  Match template searches over the entire micrograph for each")
                print("  combination of orientation and defocus values.")
                print(f"  Formula: angles × defocus × pixel_sizes")
                print(f"         = {stats['num_angles']:,} × {stats['num_defocus_values']:,} × {stats['num_pixel_sizes']:,}")
                print(f"         = {stats['total_cross_correlations']:,} total correlations")
                
            elif stats['program_type'] == 'refine_template':
                print("  Refine template searches around known particle positions using")
                print("  local orientation and defocus offsets. Search space per particle")
                print("  is the valid correlation region (extracted_box - template + 1)².")
                print(f"  Formula: particles × angles × defocus × pixel_sizes")
                print(f"         = {stats['num_particles']:,} × {stats['num_angles']:,} × {stats['num_defocus_values']:,} × {stats['num_pixel_sizes']:,}")
                print(f"         = {stats['total_cross_correlations']:,} total correlations")
                
            elif stats['program_type'] == 'inspect_peaks':
                print("  Inspect peaks analyzes correlation statistics at fixed particle")
                print("  positions without performing searches. Each particle is analyzed")
                print("  once at its known orientation and defocus.")
                print(f"  Formula: particles × angles × defocus × pixel_sizes")
                print(f"         = {stats['num_particles']:,} × {stats['num_angles']:,} × {stats['num_defocus_values']:,} × {stats['num_pixel_sizes']:,}")
                print(f"         = {stats['total_cross_correlations']:,} total correlations")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
