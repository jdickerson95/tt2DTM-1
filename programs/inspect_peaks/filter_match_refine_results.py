#!/usr/bin/env python3
"""
Filter match template and refine template results based on correlation statistics and false positive rates.

This script takes match and refine template configuration files and their corresponding results,
filters them based on correlation statistics, and outputs filtered results with combined analysis.

Usage:
    python filter_match_refine_results.py match_config.yaml refine_config.yaml match_results.csv refine_results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfcinv

# Import our correlation calculation functions
from calculate_correlations import calculate_correlations_from_config


def gaussian_noise_zscore_cutoff(num_ccg: int, false_positives: float = 1.0) -> float:
    """Determines the z-score cutoff based on Gaussian noise model and number of correlations.

    NOTE: This procedure assumes that the z-scores (normalized maximum intensity
    projections) are distributed according to a standard normal distribution. Here,
    this model is used to find the cutoff value such that there is at most
    'false_positives' number of false positives in all of the correlations.

    Parameters
    ----------
    num_ccg : int
        Total number of cross-correlograms calculated during template matching. Product
        of the number of pixels, number of defocus values, and number of orientations.
    false_positives : float, optional
        Number of false positives to allow. Default is 1.0.

    Returns
    -------
    float
        Z-score cutoff.
    """
    tmp = erfcinv(2.0 * false_positives / num_ccg)
    tmp *= np.sqrt(2.0)
    return float(tmp)


def filter_match_results(match_config: str, match_results: str, match_fp_per_micrograph: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Filter match template results based on correlation statistics.
    
    Parameters
    ----------
    match_config : str
        Path to match template configuration YAML
    match_results : str
        Path to match template results CSV
    fp_per_micrograph : float
        Desired false positives per micrograph
        
    Returns
    -------
    tuple[pd.DataFrame, dict]
        Filtered results DataFrame and statistics dictionary
    """
    # Calculate correlation statistics from config
    match_stats = calculate_correlations_from_config(match_config)
    
    if match_stats['program_type'] != 'match_template':
        raise ValueError(f"Expected match_template config, got {match_stats['program_type']}")
    
    # Load match results
    results_df = pd.read_csv(match_results)
    
    # Calculate total correlations (correlations × pixels)
    total_correlations = match_stats['total_cross_correlations']
    total_pixels = match_stats['total_pixels']
    total_operations = total_correlations * total_pixels
    
    # Calculate threshold based on false positive rate
    threshold = gaussian_noise_zscore_cutoff(total_operations, match_fp_per_micrograph)
    
    # Filter results
    score_column = 'scaled_mip' if 'scaled_mip' in results_df.columns else 'mip'
    filtered_df = results_df[results_df[score_column] > threshold].copy()
    
    stats = {
        'config_file': match_config,
        'results_file': match_results,
        'program_type': 'match_template',
        'total_correlations': total_correlations,
        'total_pixels': total_pixels,
        'total_operations': total_operations,
        'match_fp_per_micrograph': match_fp_per_micrograph,
        'threshold': threshold,
        'score_column': score_column,
        'original_count': len(results_df),
        'filtered_count': len(filtered_df),
        'filter_ratio': len(filtered_df) / len(results_df) if len(results_df) > 0 else 0
    }
    
    return filtered_df, stats


def filter_refine_results(refine_config: str, refine_results: str, match_stats: dict, refine_fp_per_micrograph: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Filter refine template results based on correlation statistics.
    
    Parameters
    ----------
    refine_config : str
        Path to refine template configuration YAML
    refine_results : str
        Path to refine template results CSV
    match_stats : dict
        Statistics from match template filtering
    fp_per_micrograph : float
        Desired false positives per micrograph
        
    Returns
    -------
    tuple[pd.DataFrame, dict]
        Filtered results DataFrame and statistics dictionary
    """
    # Calculate correlation statistics from config
    refine_stats = calculate_correlations_from_config(refine_config)
    
    if refine_stats['program_type'] != 'refine_template':
        raise ValueError(f"Expected refine_template config, got {refine_stats['program_type']}")
    
    # Load refine results
    results_df = pd.read_csv(refine_results)
    
    # For refine template, we need to calculate the combined correlation count
    # This includes correlations from match (corr × pixels) plus refine (pixels per particle × correlations per particle)
    match_operations = match_stats['total_operations']
    
    # Refine operations per particle
    refine_correlations_per_particle = refine_stats['cross_correlations_per_particle']
    refine_pixels_per_particle = refine_stats['pixels_per_particle']
    refine_operations_per_particle = refine_correlations_per_particle * refine_pixels_per_particle
    
    # Total refine operations
    num_particles = len(results_df)
    
    # Combined operations
    total_combined_operations = match_operations + refine_operations_per_particle
    
    # False positive rate per particle (since we're filtering per particle)
    fp_per_particle = refine_fp_per_micrograph / num_particles if num_particles > 0 else refine_fp_per_micrograph
    
    # Calculate threshold based on combined operations and per-particle FP rate
    threshold = gaussian_noise_zscore_cutoff(total_combined_operations, fp_per_particle)
    
    # Filter results
    score_column = 'refined_scaled_mip' if 'refined_scaled_mip' in results_df.columns else 'scaled_mip'
    if score_column not in results_df.columns:
        score_column = 'mip'
    
    filtered_df = results_df[results_df[score_column] > threshold].copy()
    
    stats = {
        'config_file': refine_config,
        'results_file': refine_results,
        'program_type': 'refine_template',
        'refine_correlations_per_particle': refine_correlations_per_particle,
        'refine_pixels_per_particle': refine_pixels_per_particle,
        'refine_operations_per_particle': refine_operations_per_particle,
        'num_particles': num_particles,
        'match_operations': match_operations,
        'total_combined_operations': total_combined_operations,
        'match_fp_per_micrograph': match_stats.get('match_fp_per_micrograph', 'N/A'),
        'refine_fp_per_micrograph': refine_fp_per_micrograph,
        'fp_per_particle': fp_per_particle,
        'threshold': threshold,
        'score_column': score_column,
        'original_count': len(results_df),
        'filtered_count': len(filtered_df),
        'filter_ratio': len(filtered_df) / len(results_df) if len(results_df) > 0 else 0
    }
    
    return filtered_df, stats


def match_peaks_between_results(match_df: pd.DataFrame, refine_df: pd.DataFrame) -> pd.DataFrame:
    """Match peaks between match and refine results based on particle indices.
    
    Parameters
    ----------
    match_df : pd.DataFrame
        Filtered match template results
    refine_df : pd.DataFrame
        Filtered refine template results
        
    Returns
    -------
    pd.DataFrame
        Combined results with particles from either match or refine (or both)
    """
    # Add source columns to track origin
    match_df = match_df.copy()
    refine_df = refine_df.copy()
    
    match_df['source'] = 'match'
    refine_df['source'] = 'refine'
    
    # If both have particle_index, we can do a proper merge
    if 'particle_index' in match_df.columns and 'particle_index' in refine_df.columns:
        # Merge on particle_index, keeping all particles from both
        combined_df = pd.merge(
            match_df, refine_df, 
            on='particle_index', 
            how='outer', 
            suffixes=('_match', '_refine')
        )
        
        # Create a combined source column
        combined_df['combined_source'] = 'both'
        combined_df.loc[combined_df['source_match'].isna(), 'combined_source'] = 'refine_only'
        combined_df.loc[combined_df['source_refine'].isna(), 'combined_source'] = 'match_only'
        
    else:
        # If no particle_index, just concatenate with source labels
        print("Warning: No particle_index column found, concatenating results without matching")
        combined_df = pd.concat([match_df, refine_df], ignore_index=True)
        combined_df['combined_source'] = combined_df['source']
    
    return combined_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Filter match and refine template results based on correlation statistics"
    )
    parser.add_argument(
        "match_config",
        help="Path to match template configuration YAML file"
    )
    parser.add_argument(
        "refine_config", 
        help="Path to refine template configuration YAML file"
    )
    parser.add_argument(
        "match_results",
        help="Path to match template results CSV file"
    )
    parser.add_argument(
        "refine_results",
        help="Path to refine template results CSV file"
    )
    parser.add_argument(
        "--match-fp", "-m",
        type=float,
        default=0.5,
        help="Desired false positives per micrograph for match stage (default: 0.5)"
    )
    parser.add_argument(
        "--refine-fp", "-r", 
        type=float,
        default=0.5,
        help="Desired false positives per micrograph for refine stage (default: 0.5)"
    )
    parser.add_argument(
        "--total-fp", "-f",
        type=float,
        help="Total desired false positives per micrograph (will split equally between match and refine if specified)"
    )
    parser.add_argument(
        "--output-prefix", "-o",
        default="filtered",
        help="Output file prefix (default: filtered)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    # Handle FP rate arguments
    if args.total_fp is not None:
        # If total FP is specified, split it equally between match and refine
        match_fp = args.total_fp / 2.0
        refine_fp = args.total_fp / 2.0
        print(f"Using total FP rate of {args.total_fp}, splitting equally: match={match_fp}, refine={refine_fp}")
    else:
        # Use individual rates
        match_fp = args.match_fp
        refine_fp = args.refine_fp
    
    # Check if files exist
    for file_path in [args.match_config, args.refine_config, args.match_results, args.refine_results]:
        if not Path(file_path).exists():
            print(f"Error: File '{file_path}' does not exist.")
            sys.exit(1)
    
    try:
        print("="*60)
        print("MATCH AND REFINE TEMPLATE RESULTS FILTERING")
        print("="*60)
        print(f"Match config: {args.match_config}")
        print(f"Refine config: {args.refine_config}")
        print(f"Match results: {args.match_results}")
        print(f"Refine results: {args.refine_results}")
        print(f"Match FP per micrograph: {match_fp}")
        print(f"Refine FP per micrograph: {refine_fp}")
        print(f"Total FP per micrograph: {match_fp + refine_fp}")
        print()
        
        # Step 1: Filter match results
        print("STEP 1: Filtering match template results...")
        match_filtered_df, match_stats = filter_match_results(
            args.match_config, args.match_results, match_fp
        )
        
        print(f"  Total correlations: {match_stats['total_correlations']:,}")
        print(f"  Total pixels: {match_stats['total_pixels']:,}")
        print(f"  Total operations: {match_stats['total_operations']:,}")
        print(f"  Threshold: {match_stats['threshold']:.4f}")
        print(f"  Filtered: {match_stats['filtered_count']:,} / {match_stats['original_count']:,} "
              f"({match_stats['filter_ratio']:.1%})")
        print()
        
        # Save filtered match results
        match_output = f"{args.output_prefix}_match_results.csv"
        match_filtered_df.to_csv(match_output, index=False)
        print(f"  Saved filtered match results to: {match_output}")
        print()
        
        # Step 2: Filter refine results
        print("STEP 2: Filtering refine template results...")
        refine_filtered_df, refine_stats = filter_refine_results(
            args.refine_config, args.refine_results, match_stats, refine_fp
        )
        
        print(f"  Refine correlations per particle: {refine_stats['refine_correlations_per_particle']:,}")
        print(f"  Refine pixels per particle: {refine_stats['refine_pixels_per_particle']:,}")
        print(f"  Refine operations per particle: {refine_stats['refine_operations_per_particle']:,}")
        print(f"  Combined operations (match + refine): {refine_stats['total_combined_operations']:,}")
        print(f"  False positives per particle: {refine_stats['fp_per_particle']:.6f}")
        print(f"  Threshold: {refine_stats['threshold']:.4f}")
        print(f"  Filtered: {refine_stats['filtered_count']:,} / {refine_stats['original_count']:,} "
              f"({refine_stats['filter_ratio']:.1%})")
        print()
        
        # Save filtered refine results
        refine_output = f"{args.output_prefix}_refine_results.csv"
        refine_filtered_df.to_csv(refine_output, index=False)
        print(f"  Saved filtered refine results to: {refine_output}")
        print()
        
        # Step 3: Match peaks and create combined results
        print("STEP 3: Matching peaks and creating combined results...")
        combined_df = match_peaks_between_results(match_filtered_df, refine_filtered_df)
        
        # Count by source
        if 'combined_source' in combined_df.columns:
            source_counts = combined_df['combined_source'].value_counts()
            print(f"  Combined results:")
            for source, count in source_counts.items():
                print(f"    {source}: {count:,}")
        else:
            print(f"  Combined results: {len(combined_df):,} total particles")
        
        # Save combined results
        combined_output = f"{args.output_prefix}_combined_results.csv"
        combined_df.to_csv(combined_output, index=False)
        print(f"  Saved combined results to: {combined_output}")
        print()
        
        # Save summary statistics
        summary_stats = {
            'match_fp_per_micrograph': match_fp,
            'refine_fp_per_micrograph': refine_fp,
            'total_fp_per_micrograph': match_fp + refine_fp,
            'match_threshold': match_stats['threshold'],
            'match_filtered_count': match_stats['filtered_count'],
            'match_original_count': match_stats['original_count'],
            'refine_threshold': refine_stats['threshold'],
            'refine_filtered_count': refine_stats['filtered_count'], 
            'refine_original_count': refine_stats['original_count'],
            'combined_count': len(combined_df),
            'match_total_operations': match_stats['total_operations'],
            'combined_total_operations': refine_stats['total_combined_operations']
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_output = f"{args.output_prefix}_summary.csv"
        summary_df.to_csv(summary_output, index=False)
        print(f"  Saved summary statistics to: {summary_output}")
        
        if args.verbose:
            print()
            print("DETAILED STATISTICS:")
            print("Match Template:")
            print(f"  Configuration: {match_stats['config_file']}")
            print(f"  Score column used: {match_stats['score_column']}")
            print(f"  Total correlations: {match_stats['total_correlations']:,}")
            print(f"  Total pixels: {match_stats['total_pixels']:,}")
            print(f"  Total operations: {match_stats['total_operations']:,}")
            print()
            print("Refine Template:")
            print(f"  Configuration: {refine_stats['config_file']}")
            print(f"  Score column used: {refine_stats['score_column']}")
            print(f"  Number of particles: {refine_stats['num_particles']:,}")
            print(f"  Operations per particle: {refine_stats['refine_operations_per_particle']:,}")
            print(f"  Combined with match operations: {refine_stats['total_combined_operations']:,}")
        
        print("="*60)
        print("FILTERING COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
