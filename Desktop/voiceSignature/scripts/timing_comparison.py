#!/usr/bin/env python3
"""
Timing Comparison Script
Compares processing times for embedding extraction across all models
"""

import os
import json
import time
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


class TimingComparison:
    def __init__(self, output_dir="../outputs"):
        """
        Initialize timing comparison analyzer
        
        Args:
            output_dir: Directory containing model outputs
        """
        self.output_dir = Path(output_dir)
        self.models = {
            'ecapa': {'dim': 192, 'name': 'ECAPA-TDNN'},
            'wav2vec2': {'dim': 768, 'name': 'Wav2Vec2'},
            'hubert': {'dim': 768, 'name': 'HuBERT'},
            'resemblyzer': {'dim': 256, 'name': 'Resemblyzer'},
            'nemo': {'dim': 512, 'name': 'NeMo'},
            'pyannote': {'dim': 256, 'name': 'Pyannote'}
        }
    
    def extract_timing_from_logs(self):
        """
        Extract timing information from the extraction summary
        
        Returns:
            dict: Timing data for each model
        """
        summary_file = self.output_dir / "extraction_summary.json"
        if not summary_file.exists():
            print(f"Extraction summary not found: {summary_file}")
            return {}
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Extract timing from results
            timing_data = {}
            for model_name in self.models.keys():
                if model_name in summary.get('results', {}):
                    # Look for timing in the results
                    model_result = summary['results'][model_name]
                    if isinstance(model_result, dict) and 'timing' in model_result:
                        timing_data[model_name] = model_result['timing']
            
            return timing_data
            
        except Exception as e:
            print(f"Error reading summary: {e}")
            return {}
    
    def measure_model_timing(self, audio_file, model_name):
        """
        Measure processing time for a specific model
        
        Args:
            audio_file: Path to audio file
            model_name: Name of the model to test
            
        Returns:
            float: Processing time in seconds
        """
        import subprocess
        import sys
        
        script_map = {
            'ecapa': 'ecapa_embed.py',
            'wav2vec2': 'wav2vec2_embed.py',
            'hubert': 'hubert_embed.py',
            'resemblyzer': 'resemblyzer_embed.py',
            'nemo': 'nemo_embed.py',
            'pyannote': 'pyannote_embed.py'
        }
        
        if model_name not in script_map:
            print(f"Unknown model: {model_name}")
            return None
        
        script_path = Path(__file__).parent / script_map[model_name]
        
        # Prepare command
        cmd = [
            sys.executable, str(script_path),
            '--input_file', str(audio_file),
            '--output_dir', str(self.output_dir / model_name),
            '--device', 'cpu'  # Use CPU for consistent timing
        ]
        
        # Measure time
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"  {model_name}: {processing_time:.2f}s")
                return processing_time
            else:
                print(f"  {model_name}: Failed ({result.stderr})")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"  {model_name}: Timeout (>10 minutes)")
            return None
        except Exception as e:
            print(f"  {model_name}: Error - {e}")
            return None
    
    def benchmark_all_models(self, audio_file, runs=1):
        """
        Benchmark all models on a single audio file
        
        Args:
            audio_file: Path to audio file
            runs: Number of runs per model (for averaging)
            
        Returns:
            dict: Timing results for all models
        """
        print(f"Benchmarking all models on: {audio_file}")
        print(f"Number of runs per model: {runs}")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nTesting {model_name}...")
            
            times = []
            for run in range(runs):
                if runs > 1:
                    print(f"  Run {run + 1}/{runs}")
                
                timing = self.measure_model_timing(audio_file, model_name)
                if timing is not None:
                    times.append(timing)
            
            if times:
                results[model_name] = {
                    'times': times,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'embedding_dim': self.models[model_name]['dim'],
                    'model_name': self.models[model_name]['name']
                }
        
        return results
    
    def plot_timing_comparison(self, results, output_path="../outputs"):
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
        
        model_names = [self.models[name]['name'] for name in results.keys()]
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
        plt.title('Embedding Extraction Time Comparison')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_time) in enumerate(zip(bars, mean_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
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
        self.create_timing_table(results, output_path)
    
    def create_timing_table(self, results, output_path):
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
    
    def print_timing_summary(self, results):
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


def main():
    parser = argparse.ArgumentParser(description='Compare embedding extraction timing across models')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Directory containing model outputs')
    parser.add_argument('--audio_file', type=str, default='../audio/Macron1.wav',
                       help='Audio file to use for benchmarking')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per model for averaging')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save timing plots')
    parser.add_argument('--extract_from_logs', action='store_true',
                       help='Extract timing from existing logs instead of running benchmarks')
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparator = TimingComparison(output_dir=args.output_dir)
    
    if args.extract_from_logs:
        # Extract timing from existing logs
        results = comparator.extract_timing_from_logs()
        if not results:
            print("No timing data found in logs. Run with --audio_file to benchmark.")
            return
    else:
        # Run benchmarks
        audio_file = Path(args.audio_file)
        if not audio_file.exists():
            print(f"Audio file not found: {audio_file}")
            return
        
        results = comparator.benchmark_all_models(audio_file, runs=args.runs)
    
    if not results:
        print("No timing results available!")
        return
    
    # Print summary
    comparator.print_timing_summary(results)
    
    # Create visualizations
    if args.save_plots:
        comparator.plot_timing_comparison(results, args.output_dir)
    
    print("\nTiming comparison completed!")


if __name__ == "__main__":
    main() 