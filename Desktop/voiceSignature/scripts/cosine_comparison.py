#!/usr/bin/env python3
"""
Cosine Similarity Comparison Script
Compares embeddings from the same model across different audio files
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd


class CosineComparison:
    def __init__(self, output_dir="../outputs"):
        """
        Initialize cosine comparison analyzer
        
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
        
    def load_embeddings(self, model_name, file_pattern="*_embeddings.json"):
        """
        Load embeddings from a specific model
        
        Args:
            model_name: Name of the model
            file_pattern: Pattern to match embedding files
            
        Returns:
            dict: Dictionary of embeddings by filename
        """
        model_dir = self.output_dir / model_name
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            return {}
        
        embeddings = {}
        for embedding_file in model_dir.glob(file_pattern):
            try:
                with open(embedding_file, 'r') as f:
                    data = json.load(f)
                
                filename = Path(data['file_path']).stem
                embeddings[filename] = {
                    'full_embedding': np.array(data['full_embedding']),
                    'segment_embeddings': [np.array(emb) for emb in data['segment_embeddings']],
                    'embedding_dim': data['embedding_dim'],
                    'model_name': data.get('model_name', model_name),
                    'duration': data['duration']
                }
                
            except Exception as e:
                print(f"Error loading {embedding_file}: {e}")
        
        return embeddings
    
    def compute_cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        # Normalize embeddings
        emb1_norm = normalize(emb1.reshape(1, -1), norm='l2')
        emb2_norm = normalize(emb2.reshape(1, -1), norm='l2')
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1_norm, emb2_norm)[0][0]
        return similarity
    
    def compare_audio_files(self, filenames=None):
        """
        Compare embeddings across different audio files for each model
        
        Args:
            filenames: List of filenames to compare (if None, uses all available)
            
        Returns:
            dict: Comparison results
        """
        print("Loading embeddings from all models...")
        
        # Load embeddings from all models
        all_embeddings = {}
        for model_name in self.models.keys():
            embeddings = self.load_embeddings(model_name)
            if embeddings:
                all_embeddings[model_name] = embeddings
                print(f"Loaded {len(embeddings)} files from {model_name}")
        
        if not all_embeddings:
            print("No embeddings found!")
            return {}
        
        # Determine files to compare
        if filenames is None:
            # Find all files across all models
            all_files = set()
            for model_embeddings in all_embeddings.values():
                all_files.update(model_embeddings.keys())
            filenames = list(all_files)
        
        print(f"Comparing {len(filenames)} files across {len(all_embeddings)} models")
        
        # Compare embeddings for each model
        results = {
            'model_comparisons': {},
            'file_similarities': {},
            'summary_stats': {}
        }
        
        # For each model, compare embeddings across files
        for model_name, model_embeddings in all_embeddings.items():
            print(f"\nAnalyzing {model_name}...")
            
            model_files = [f for f in filenames if f in model_embeddings]
            if len(model_files) < 2:
                print(f"  Not enough files for {model_name} (need at least 2)")
                continue
            
            # Compute pairwise similarities for this model
            similarities = {}
            similarity_matrix = np.zeros((len(model_files), len(model_files)))
            
            for i, file1 in enumerate(model_files):
                for j, file2 in enumerate(model_files):
                    if i == j:
                        similarity_matrix[i, j] = 1.0  # Self-similarity
                        continue
                    
                    emb1 = model_embeddings[file1]['full_embedding']
                    emb2 = model_embeddings[file2]['full_embedding']
                    
                    similarity = self.compute_cosine_similarity(emb1, emb2)
                    similarities[f"{file1}_vs_{file2}"] = similarity
                    similarity_matrix[i, j] = similarity
            
            # Store results for this model
            results['model_comparisons'][model_name] = {
                'files': model_files,
                'similarities': similarities,
                'similarity_matrix': similarity_matrix.tolist(),
                'embedding_dim': self.models[model_name]['dim'],
                'model_name': self.models[model_name]['name'],
                'mean_similarity': np.mean(list(similarities.values())),
                'std_similarity': np.std(list(similarities.values())),
                'min_similarity': np.min(list(similarities.values())),
                'max_similarity': np.max(list(similarities.values()))
            }
        
        return results
    
    def plot_similarity_matrices(self, results, output_path="../outputs"):
        """
        Create similarity matrix visualizations for each model
        
        Args:
            results: Comparison results
            output_path: Directory to save plots
        """
        if not results['model_comparisons']:
            print("No model comparisons to plot")
            return
        
        # Create subplots for each model
        n_models = len(results['model_comparisons'])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, data) in enumerate(results['model_comparisons'].items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            similarity_matrix = np.array(data['similarity_matrix'])
            files = data['files']
            
            # Create heatmap
            sns.heatmap(similarity_matrix, 
                       annot=True, 
                       fmt='.3f',
                       xticklabels=files,
                       yticklabels=files,
                       cmap='RdYlBu_r',
                       center=0.5,
                       vmin=0,
                       vmax=1,
                       ax=ax)
            
            ax.set_title(f"{data['model_name']} ({data['embedding_dim']}D)")
            ax.set_xlabel('Audio Files')
            ax.set_ylabel('Audio Files')
        
        # Hide unused subplots
        for i in range(len(results['model_comparisons']), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / "audio_file_similarity_matrices.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Similarity matrices saved to: {plot_file}")
        plt.close()
    
    def plot_similarity_distributions(self, results, output_path="../outputs"):
        """
        Create distribution plots for each model's similarities
        
        Args:
            results: Comparison results
            output_path: Directory to save plots
        """
        if not results['model_comparisons']:
            print("No model comparisons to plot")
            return
        
        # Create subplots for each model
        n_models = len(results['model_comparisons'])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (model_name, data) in enumerate(results['model_comparisons'].items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            similarities = list(data['similarities'].values())
            
            ax.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(data['mean_similarity'], color='red', linestyle='--', 
                      label=f"Mean: {data['mean_similarity']:.3f}")
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Frequency')
            ax.set_title(f"{data['model_name']} ({data['embedding_dim']}D)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(results['model_comparisons']), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / "audio_file_similarity_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Similarity distributions saved to: {plot_file}")
        plt.close()
    
    def save_results(self, results, output_path="../outputs"):
        """
        Save comparison results to JSON
        
        Args:
            results: Comparison results
            output_path: Directory to save results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'model_comparisons':
                json_results[key] = {}
                for model_key, model_data in value.items():
                    json_results[key][model_key] = {
                        'files': model_data['files'],
                        'similarities': model_data['similarities'],
                        'similarity_matrix': model_data['similarity_matrix'],
                        'embedding_dim': model_data['embedding_dim'],
                        'model_name': model_data['model_name'],
                        'mean_similarity': float(model_data['mean_similarity']),
                        'std_similarity': float(model_data['std_similarity']),
                        'min_similarity': float(model_data['min_similarity']),
                        'max_similarity': float(model_data['max_similarity'])
                    }
        
        results_file = output_path / "audio_file_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def print_summary(self, results):
        """
        Print a summary of comparison results
        
        Args:
            results: Comparison results
        """
        print("\n" + "="*60)
        print("AUDIO FILE COSINE SIMILARITY COMPARISON SUMMARY")
        print("="*60)
        
        # Print model-wise comparisons
        print("\nModel-wise Audio File Comparisons:")
        print("-" * 50)
        for model_name, data in results['model_comparisons'].items():
            print(f"\n{data['model_name']} ({data['embedding_dim']}D):")
            print(f"  Files: {data['files']}")
            print(f"  Mean similarity: {data['mean_similarity']:.3f} ± {data['std_similarity']:.3f}")
            print(f"  Range: [{data['min_similarity']:.3f}, {data['max_similarity']:.3f}]")
            
            # Print pairwise similarities
            print("  Pairwise similarities:")
            for pair, similarity in data['similarities'].items():
                file1, file2 = pair.split('_vs_')
                print(f"    {file1} vs {file2}: {similarity:.3f}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Compare embeddings using cosine similarity')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Directory containing model outputs')
    parser.add_argument('--files', type=str, nargs='+',
                       help='Specific files to compare (optional)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save similarity plots')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparator = CosineComparison(output_dir=args.output_dir)
    
    # Run comparison
    results = comparator.compare_audio_files(filenames=args.files)
    
    if not results:
        print("No results to analyze!")
        return
    
    # Print summary
    comparator.print_summary(results)
    
    # Save results
    if args.save_results:
        comparator.save_results(results, args.output_dir)
    
    # Create plots
    if args.save_plots:
        comparator.plot_similarity_matrices(results, args.output_dir)
        comparator.plot_similarity_distributions(results, args.output_dir)
    
    print("Audio file cosine similarity comparison completed!")


if __name__ == "__main__":
    main() 