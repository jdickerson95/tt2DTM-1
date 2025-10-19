#!/usr/bin/env python3
"""
Filter CSV results using multiple refine template configurations.

This script takes multiple refine template configuration files, sums their correlation
statistics, and filters a CSV file based on the combined correlation counts and 
false positive rates.

Usage:
    python filter_multi_refine_results.py results.csv refine_config1.yaml refine_config2.yaml [refine_config3.yaml ...]
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


def calculate_combined_refine_stats(refine_configs: list[str]) -> dict:
    """Calculate combined correlation statistics from multiple refine template configs.
    
    Parameters
    ----------
    refine_configs : list[str]
        List of paths to refine template configuration YAML files
        
    Returns
    -------
    dict
        Combined statistics dictionary
    """
    combined_stats = {
        'config_files': refine_configs,
        'individual_stats': [],
        'total_correlations_per_particle': 0,
        'total_pixels_per_particle': 0,
        'total_operations_per_particle': 0,
        'num_particles': None,
        'program_types': []
    }
    
    print(f"Calculating correlation statistics from {len(refine_configs)} refine configs...")
    
    for i, config_path in enumerate(refine_configs):
        print(f"  Processing config {i+1}/{len(refine_configs)}: {Path(config_path).name}")
        
        # Calculate statistics for this config
        stats = calculate_correlations_from_config(config_path)
        
        # Verify it's a refine template config
        if stats['program_type'] != 'refine_template':
            raise ValueError(f"Expected refine_template config, got {stats['program_type']} for {config_path}")
        
        combined_stats['individual_stats'].append(stats)
        combined_stats['program_types'].append(stats['program_type'])
        
        # Sum the per-particle statistics
        combined_stats['total_correlations_per_particle'] += stats['cross_correlations_per_particle']
        combined_stats['total_pixels_per_particle'] += stats['pixels_per_particle']
        
        # Check that all configs have the same number of particles
        if combined_stats['num_particles'] is None:
            combined_stats['num_particles'] = stats['num_particles']
        elif combined_stats['num_particles'] != stats['num_particles']:
            raise ValueError(f"Mismatch in number of particles: {combined_stats['num_particles']} vs {stats['num_particles']} in {config_path}")
        
        print(f"    Correlations per particle: {stats['cross_correlations_per_particle']:,}")
        print(f"    Pixels per particle: {stats['pixels_per_particle']:,}")
        print(f"    Number of particles: {stats['num_particles']:,}")
    
    # Calculate combined operations per particle
    combined_stats['total_operations_per_particle'] = (
        combined_stats['total_correlations_per_particle'] * 
        combined_stats['total_pixels_per_particle']
    )
    
    # Total operations across all particles
    combined_stats['total_operations'] = (
        combined_stats['total_operations_per_particle'] * 
        combined_stats['num_particles']
    )
    
    print(f"\nCombined statistics:")
    print(f"  Total correlations per particle: {combined_stats['total_correlations_per_particle']:,}")
    print(f"  Total pixels per particle: {combined_stats['total_pixels_per_particle']:,}")
    print(f"  Total operations per particle: {combined_stats['total_operations_per_particle']:,}")
    print(f"  Number of particles: {combined_stats['num_particles']:,}")
    print(f"  Total operations (all particles): {combined_stats['total_operations']:,}")
    
    return combined_stats


def filter_csv_results(csv_file: str, combined_stats: dict, fp_per_micrograph: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Filter CSV results based on combined correlation statistics.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file to filter
    combined_stats : dict
        Combined correlation statistics from multiple refine configs
    fp_per_micrograph : float
        Desired false positives per micrograph
        
    Returns
    -------
    tuple[pd.DataFrame, dict]
        Filtered results DataFrame and filtering statistics dictionary
    """
    # Load CSV results
    results_df = pd.read_csv(csv_file)
    num_particles = len(results_df)
    
    print(f"\nLoaded CSV file: {csv_file}")
    print(f"  Number of rows: {num_particles:,}")
    
    # Verify particle count matches (if applicable)
    if combined_stats['num_particles'] != num_particles:
        print(f"Warning: Config particle count ({combined_stats['num_particles']:,}) "
              f"doesn't match CSV row count ({num_particles:,})")
        print("Using CSV row count for calculations...")
    
    # Calculate false positive rate per particle
    fp_per_particle = fp_per_micrograph / num_particles if num_particles > 0 else fp_per_micrograph
    
    # Calculate threshold based on combined operations per particle and per-particle FP rate
    threshold = gaussian_noise_zscore_cutoff(combined_stats['total_operations_per_particle'], fp_per_particle)
    
    # Determine which score column to use
    score_columns = ['refined_scaled_mip', 'scaled_mip', 'mip']
    score_column = None
    
    for col in score_columns:
        if col in results_df.columns:
            score_column = col
            break
    
    if score_column is None:
        available_cols = list(results_df.columns)
        raise ValueError(f"No suitable score column found. Available columns: {available_cols}")
    
    print(f"  Using score column: {score_column}")
    print(f"  Score range: {results_df[score_column].min():.4f} to {results_df[score_column].max():.4f}")
    
    # Filter results
    filtered_df = results_df[results_df[score_column] > threshold].copy()
    
    filter_stats = {
        'csv_file': csv_file,
        'num_particles': num_particles,
        'fp_per_micrograph': fp_per_micrograph,
        'fp_per_particle': fp_per_particle,
        'threshold': threshold,
        'score_column': score_column,
        'original_count': len(results_df),
        'filtered_count': len(filtered_df),
        'filter_ratio': len(filtered_df) / len(results_df) if len(results_df) > 0 else 0,
        'combined_operations_per_particle': combined_stats['total_operations_per_particle'],
        'combined_correlations_per_particle': combined_stats['total_correlations_per_particle'],
        'combined_pixels_per_particle': combined_stats['total_pixels_per_particle']
    }
    
    print(f"  False positives per micrograph: {fp_per_micrograph}")
    print(f"  False positives per particle: {fp_per_particle:.6f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Filtered: {len(filtered_df):,} / {len(results_df):,} ({filter_stats['filter_ratio']:.1%})")
    
    return filtered_df, filter_stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Filter CSV results using multiple refine template configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter results using 2 refine configs
  python filter_multi_refine_results.py results.csv refine1.yaml refine2.yaml
  
  # Filter with custom false positive rate
  python filter_multi_refine_results.py results.csv refine1.yaml refine2.yaml --fp-rate 0.1
  
  # Filter with custom output prefix
  python filter_multi_refine_results.py results.csv refine1.yaml refine2.yaml -o filtered_multi
        """
    )
    
    parser.add_argument(
        "csv_file",
        help="Path to CSV file to filter"
    )
    parser.add_argument(
        "refine_configs",
        nargs='+',
        help="Paths to refine template configuration YAML files (at least 1 required)"
    )
    parser.add_argument(
        "--fp-rate", "-f",
        type=float,
        default=0.5,
        help="Desired false positives per micrograph (default: 0.5)"
    )
    parser.add_argument(
        "--output-prefix", "-o",
        default="multi_filtered",
        help="Output file prefix (default: multi_filtered)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    for config_path in args.refine_configs:
        if not Path(config_path).exists():
            print(f"Error: Configuration file '{config_path}' does not exist.")
            sys.exit(1)
    
    try:
        print("="*70)
        print("MULTI-REFINE TEMPLATE RESULTS FILTERING")
        print("="*70)
        print(f"CSV file: {args.csv_file}")
        print(f"Refine configs ({len(args.refine_configs)}):")
        for i, config in enumerate(args.refine_configs, 1):
            print(f"  {i}. {config}")
        print(f"False positives per micrograph: {args.fp_rate}")
        print()
        
        # Step 1: Calculate combined correlation statistics
        print("STEP 1: Calculating combined correlation statistics...")
        combined_stats = calculate_combined_refine_stats(args.refine_configs)
        print()
        
        # Step 2: Filter CSV results
        print("STEP 2: Filtering CSV results...")
        filtered_df, filter_stats = filter_csv_results(
            args.csv_file, combined_stats, args.fp_rate
        )
        print()
        
        # Step 3: Save results
        print("STEP 3: Saving results...")
        
        # Save filtered results
        filtered_output = f"{args.output_prefix}_results.csv"
        filtered_df.to_csv(filtered_output, index=False)
        print(f"  Saved filtered results to: {filtered_output}")
        
        # Save summary statistics
        summary_stats = {
            'csv_file': args.csv_file,
            'num_refine_configs': len(args.refine_configs),
            'refine_config_files': '; '.join(args.refine_configs),
            'fp_per_micrograph': args.fp_rate,
            'fp_per_particle': filter_stats['fp_per_particle'],
            'threshold': filter_stats['threshold'],
            'score_column': filter_stats['score_column'],
            'original_count': filter_stats['original_count'],
            'filtered_count': filter_stats['filtered_count'],
            'filter_ratio': filter_stats['filter_ratio'],
            'combined_correlations_per_particle': combined_stats['total_correlations_per_particle'],
            'combined_pixels_per_particle': combined_stats['total_pixels_per_particle'],
            'combined_operations_per_particle': combined_stats['total_operations_per_particle'],
            'total_operations': combined_stats['total_operations']
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_output = f"{args.output_prefix}_summary.csv"
        summary_df.to_csv(summary_output, index=False)
        print(f"  Saved summary statistics to: {summary_output}")
        
        if args.verbose:
            print()
            print("DETAILED STATISTICS:")
            print(f"Combined Refine Template Statistics:")
            for i, stats in enumerate(combined_stats['individual_stats'], 1):
                print(f"  Config {i}: {Path(stats['configuration_file']).name}")
                print(f"    Correlations per particle: {stats['cross_correlations_per_particle']:,}")
                print(f"    Pixels per particle: {stats['pixels_per_particle']:,}")
                print(f"    Operations per particle: {stats['cross_correlations_per_particle'] * stats['pixels_per_particle']:,}")
            print()
            print(f"Combined Totals:")
            print(f"  Total correlations per particle: {combined_stats['total_correlations_per_particle']:,}")
            print(f"  Total pixels per particle: {combined_stats['total_pixels_per_particle']:,}")
            print(f"  Total operations per particle: {combined_stats['total_operations_per_particle']:,}")
            print(f"  Total operations (all particles): {combined_stats['total_operations']:,}")
        
        print()
        print("="*70)
        print("FILTERING COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
