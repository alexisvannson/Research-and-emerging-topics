#!/usr/bin/env python3
"""
Master script to run all audio embedding extraction models
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time


def run_model(model_name, input_path, output_dir, device='auto'):
    """
    Run a specific embedding extraction model
    
    Args:
        model_name: Name of the model to run
        input_path: Path to input (file or directory)
        output_dir: Directory to save embeddings
        device: Device to use (cuda, cpu, or auto)
    """
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
        print(f"Available models: {list(script_map.keys())}")
        return False
    
    script_path = Path(__file__).parent / script_map[model_name]
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    
    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if input is a file or directory
    input_path_obj = Path(input_path)
    if input_path_obj.is_file():
        # Single file mode
        cmd = [
            sys.executable, str(script_path),
            '--input_file', str(input_path),
            '--output_dir', str(model_output_dir),
            '--device', device
        ]
    else:
        # Directory mode
        cmd = [
            sys.executable, str(script_path),
            '--input_dir', str(input_path),
            '--output_dir', str(model_output_dir),
            '--device', device
        ]
    
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} embedding extraction...")
    print(f"Input: {input_path}")
    print(f"Output: {model_output_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        end_time = time.time()
        print(f"\n{model_name.upper()} completed in {end_time - start_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {model_name}: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all audio embedding extraction models')
    parser.add_argument('--input_dir', type=str, default='../audio', 
                       help='Directory containing audio files')
    parser.add_argument('--input_file', type=str,
                       help='Single audio file to process (overrides input_dir)')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Directory to save embeddings')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['ecapa', 'wav2vec2', 'hubert', 'resemblyzer', 'nemo', 'pyannote'],
                       help='Models to run')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run models in parallel (experimental)')
    
    args = parser.parse_args()
    
    # Determine input path
    if args.input_file:
        input_path = args.input_file
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            print(f"Input file not found: {input_path}")
            return
        print(f"Processing single file: {input_path}")
    else:
        input_path = args.input_dir
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            print(f"Input directory not found: {input_path}")
            return
        print(f"Processing all files in directory: {input_path}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Audio Embedding Extraction Pipeline")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Models to run: {args.models}")
    print(f"Device: {args.device}")
    print(f"{'='*60}")
    
    # Run models
    results = {}
    total_start_time = time.time()
    
    if args.parallel:
        print("Running models in parallel...")
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.models)) as executor:
            future_to_model = {
                executor.submit(run_model, model, input_path, args.output_dir, args.device): model
                for model in args.models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    success = future.result()
                    results[model] = success
                except Exception as e:
                    print(f"Exception for {model}: {e}")
                    results[model] = False
    else:
        print("Running models sequentially...")
        for model in args.models:
            success = run_model(model, input_path, args.output_dir, args.device)
            results[model] = success
    
    # Print summary
    total_end_time = time.time()
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    successful_models = [model for model, success in results.items() if success]
    failed_models = [model for model, success in results.items() if not success]
    
    print(f"Successful: {successful_models}")
    if failed_models:
        print(f"Failed: {failed_models}")
    
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    
    # Create summary file
    summary = {
        'input_path': str(input_path),
        'output_directory': str(output_path),
        'device': args.device,
        'total_time_seconds': total_end_time - total_start_time,
        'successful_models': successful_models,
        'failed_models': failed_models,
        'results': results
    }
    
    summary_file = output_path / "extraction_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main() 