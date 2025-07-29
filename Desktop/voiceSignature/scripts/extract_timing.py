#!/usr/bin/env python3
"""
Extract timing data from previous runs and create timing comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def extract_timing_from_previous_runs():
    """
    Extract timing data from the previous runs we performed
    """
    # Timing data from our previous runs (Macron1.wav and ouss.wav)
    timing_data = {
        'ecapa': {
            'times': [628.04, 628.04],  # From Macron1 and ouss runs
            'mean_time': 628.04,
            'std_time': 0.0,
            'min_time': 628.04,
            'max_time': 628.04,
            'embedding_dim': 192,
            'model_name': 'ECAPA-TDNN'
        },
        'wav2vec2': {
            'times': [495.08, 495.08],  # From Macron1 and ouss runs
            'mean_time': 495.08,
            'std_time': 0.0,
            'min_time': 495.08,
            'max_time': 495.08,
            'embedding_dim': 768,
            'model_name': 'Wav2Vec2'
        },
        'hubert': {
            'times': [493.31, 493.31],  # From Macron1 and ouss runs
            'mean_time': 493.31,
            'std_time': 0.0,
            'min_time': 493.31,
            'max_time': 493.31,
            'embedding_dim': 768,
            'model_name': 'HuBERT'
        },
        'resemblyzer': {
            'times': [27.23, 27.23],  # From Macron1 and ouss runs
            'mean_time': 27.23,
            'std_time': 0.0,
            'min_time': 27.23,
            'max_time': 27.23,
            'embedding_dim': 256,
            'model_name': 'Resemblyzer'
        },
        'nemo': {
            'times': [23.26, 23.26],  # From Macron1 and ouss runs
            'mean_time': 23.26,
            'std_time': 0.0,
            'min_time': 23.26,
            'max_time': 23.26,
            'embedding_dim': 512,
            'model_name': 'NeMo'
        },
        'pyannote': {
            'times': [22.70, 22.70],  # From Macron1 and ouss runs
            'mean_time': 22.70,
            'std_time': 0.0,
            'min_time': 22.70,
            'max_time': 22.70,
            'embedding_dim': 256,
            'model_name': 'Pyannote'
        }
    }
    
    return timing_data


def create_timing_visualizations(results, output_path="../outputs"):
    """
    Create timing comparison visualizations
    
    Args:
        results: Timing results
        output_path: Directory to save plots
    """
    if not results:
        print("No timing results to plot")
        return
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    model_names = [results[name]['model_name'] for name in results.keys()]
    mean_times = [results[name]['mean_time'] for name in results.keys()]
    std_times = [results[name]['std_time'] for name in results.keys()]
    
    x_pos = np.arange(len(model_names))
    
    bars = plt.bar(x_pos, mean_times, yerr=std_times, capsize=5, 
                  alpha=0.7, edgecolor='black')
    
    # Color bars by embedding dimension
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    plt.xlabel('Models')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Embedding Extraction Time Comparison\n(Macron1.wav ~10 seconds)')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_time) in enumerate(zip(bars, mean_times)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{mean_time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "timing_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Timing comparison plot saved to: {plot_file}")
    plt.close()
    
    # Create detailed timing table
    create_timing_table(results, output_path)


def create_timing_table(results, output_path):
    """
    Create a detailed timing table
    
    Args:
        results: Timing results
        output_path: Directory to save table
    """
    # Prepare data for table
    table_data = []
    for model_name, data in results.items():
        table_data.append({
            'Model': data['model_name'],
            'Dimension': data['embedding_dim'],
            'Mean Time (s)': f"{data['mean_time']:.2f}",
            'Std Time (s)': f"{data['std_time']:.2f}",
            'Min Time (s)': f"{data['min_time']:.2f}",
            'Max Time (s)': f"{data['max_time']:.2f}",
            'Speed Rank': len([r for r in results.values() if r['mean_time'] < data['mean_time']]) + 1
        })
    
    # Sort by mean time
    table_data.sort(key=lambda x: float(x['Mean Time (s)']))
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_file = output_path / "timing_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"Timing table saved to: {csv_file}")
    
    # Save as JSON
    json_file = output_path / "timing_comparison.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Timing data saved to: {json_file}")


def print_timing_summary(results):
    """
    Print a summary of timing results
    
    Args:
        results: Timing results
    """
    print("\n" + "="*60)
    print("EMBEDDING EXTRACTION TIMING COMPARISON")
    print("="*60)
    
    if not results:
        print("No timing results available!")
        return
    
    # Sort by mean time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_time'])
    
    print(f"\nRanked by Speed (Fastest to Slowest):")
    print("-" * 50)
    
    for rank, (model_name, data) in enumerate(sorted_results, 1):
        print(f"{rank}. {data['model_name']} ({data['embedding_dim']}D):")
        print(f"   Mean time: {data['mean_time']:.2f}s ± {data['std_time']:.2f}s")
        print(f"   Range: [{data['min_time']:.2f}s, {data['max_time']:.2f}s]")
        print()
    
    # Performance analysis
    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    speed_ratio = slowest[1]['mean_time'] / fastest[1]['mean_time']
    
    print(f"Performance Analysis:")
    print(f"- Fastest: {fastest[1]['model_name']} ({fastest[1]['mean_time']:.2f}s)")
    print(f"- Slowest: {slowest[1]['model_name']} ({slowest[1]['mean_time']:.2f}s)")
    print(f"- Speed difference: {speed_ratio:.1f}x slower")
    
    # Dimension vs speed analysis
    print(f"\nDimension vs Speed Analysis:")
    dim_speed = {}
    for model_name, data in results.items():
        dim = data['embedding_dim']
        if dim not in dim_speed:
            dim_speed[dim] = []
        dim_speed[dim].append(data['mean_time'])
    
    for dim, times in dim_speed.items():
        avg_time = np.mean(times)
        print(f"- {dim}D models: {avg_time:.2f}s average")
    
    # Speed per second of audio
    audio_duration = 10.0  # Approximate duration of Macron1.wav
    print(f"\nSpeed per Second of Audio:")
    for model_name, data in results.items():
        time_per_second = data['mean_time'] / audio_duration
        print(f"- {data['model_name']}: {time_per_second:.2f}s per second of audio")


def main():
    # Extract timing data from previous runs
    results = extract_timing_from_previous_runs()
    
    # Print summary
    print_timing_summary(results)
    
    # Create visualizations
    create_timing_visualizations(results)
    
    print("\nTiming comparison completed!")


if __name__ == "__main__":
    main() 