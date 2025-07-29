#!/usr/bin/env python3
"""
Macron1 vs Macron2 Cosine Distance Visualization
Compares embeddings between Macron1 and Macron2 across all models
with respect to their embedding dimensions
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
from typing import Dict, List, Tuple


class MacronComparisonVisualizer:
    def __init__(self, output_dir="../outputs"):
        """
        Initialize Macron comparison visualizer
        
        Args:
            output_dir: Directory containing model outputs
        """
        self.output_dir = Path(output_dir)
        self.models = {
            'ecapa': {'dim': 192, 'name': 'ECAPA-TDNN', 'color': '#1f77b4'},
            'wav2vec2': {'dim': 768, 'name': 'Wav2Vec2', 'color': '#ff7f0e'},
            'hubert': {'dim': 768, 'name': 'HuBERT', 'color': '#2ca02c'},
            'resemblyzer': {'dim': 256, 'name': 'Resemblyzer', 'color': '#d62728'},
            'nemo': {'dim': 512, 'name': 'NeMo', 'color': '#9467bd'},
            'pyannote': {'dim': 256, 'name': 'Pyannote', 'color': '#8c564b'}
        }
        
    def load_macron_embeddings(self) -> Dict[str, Dict]:
        """
        Load Macron1 and Macron2 embeddings from all available models
        
        Returns:
            dict: Dictionary with model names as keys and embeddings as values
        """
        embeddings = {}
        
        for model_name in self.models.keys():
            model_dir = self.output_dir / model_name
            if not model_dir.exists():
                print(f"Model directory not found: {model_dir}")
                continue
            
            # Look for Macron1 and Macron2 embeddings
            macron1_file = model_dir / f"Macron1_{model_name}_embeddings.json"
            macron2_file = model_dir / f"Macron2_{model_name}_embeddings.json"
            
            model_embeddings = {}
            
            # Load Macron1 embeddings
            if macron1_file.exists():
                try:
                    with open(macron1_file, 'r') as f:
                        data = json.load(f)
                    model_embeddings['Macron1'] = {
                        'full_embedding': np.array(data['full_embedding']),
                        'embedding_dim': data['embedding_dim'],
                        'duration': data['duration']
                    }
                    print(f"Loaded Macron1 embeddings from {model_name} ({data['embedding_dim']}D)")
                except Exception as e:
                    print(f"Error loading Macron1 from {model_name}: {e}")
            
            # Load Macron2 embeddings
            if macron2_file.exists():
                try:
                    with open(macron2_file, 'r') as f:
                        data = json.load(f)
                    model_embeddings['Macron2'] = {
                        'full_embedding': np.array(data['full_embedding']),
                        'embedding_dim': data['embedding_dim'],
                        'duration': data['duration']
                    }
                    print(f"Loaded Macron2 embeddings from {model_name} ({data['embedding_dim']}D)")
                except Exception as e:
                    print(f"Error loading Macron2 from {model_name}: {e}")
            
            if len(model_embeddings) == 2:  # Both Macron1 and Macron2 available
                embeddings[model_name] = model_embeddings
        
        return embeddings
    
    def compute_cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine distance between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            float: Cosine distance (1 - cosine_similarity)
        """
        # Normalize embeddings
        emb1_norm = normalize(emb1.reshape(1, -1), norm='l2')
        emb2_norm = normalize(emb2.reshape(1, -1), norm='l2')
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1_norm, emb2_norm)[0][0]
        
        # Convert to distance (1 - similarity)
        distance = 1 - similarity
        return distance
    
    def analyze_macron_comparison(self) -> Dict:
        """
        Analyze Macron1 vs Macron2 comparison across all models
        
        Returns:
            dict: Analysis results
        """
        print("Loading Macron1 and Macron2 embeddings...")
        embeddings = self.load_macron_embeddings()
        
        if not embeddings:
            print("No Macron embeddings found!")
            return {}
        
        results = {
            'model_comparisons': {},
            'summary_stats': {}
        }
        
        # Compare Macron1 vs Macron2 for each model
        for model_name, model_embeddings in embeddings.items():
            if 'Macron1' in model_embeddings and 'Macron2' in model_embeddings:
                emb1 = model_embeddings['Macron1']['full_embedding']
                emb2 = model_embeddings['Macron2']['full_embedding']
                
                # Compute cosine distance
                distance = self.compute_cosine_distance(emb1, emb2)
                similarity = 1 - distance
                
                # Get model info
                model_info = self.models[model_name]
                
                results['model_comparisons'][model_name] = {
                    'embedding_dim': model_info['dim'],
                    'model_name': model_info['name'],
                    'color': model_info['color'],
                    'cosine_distance': distance,
                    'cosine_similarity': similarity,
                    'macron1_duration': model_embeddings['Macron1']['duration'],
                    'macron2_duration': model_embeddings['Macron2']['duration']
                }
                
                print(f"{model_info['name']} ({model_info['dim']}D): Distance = {distance:.4f}, Similarity = {similarity:.4f}")
        
        # Compute summary statistics
        if results['model_comparisons']:
            distances = [data['cosine_distance'] for data in results['model_comparisons'].values()]
            similarities = [data['cosine_similarity'] for data in results['model_comparisons'].values()]
            dimensions = [data['embedding_dim'] for data in results['model_comparisons'].values()]
            
            results['summary_stats'] = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities),
                'mean_dimension': np.mean(dimensions),
                'std_dimension': np.std(dimensions)
            }
        
        return results
    
    def plot_distance_vs_dimension(self, results: Dict, output_path: str = "../outputs"):
        """
        Create scatter plot of cosine distance vs embedding dimension
        
        Args:
            results: Analysis results
            output_path: Directory to save plots
        """
        if not results['model_comparisons']:
            print("No data to plot!")
            return
        
        # Prepare data
        models = list(results['model_comparisons'].keys())
        distances = [results['model_comparisons'][model]['cosine_distance'] for model in models]
        dimensions = [results['model_comparisons'][model]['embedding_dim'] for model in models]
        names = [results['model_comparisons'][model]['model_name'] for model in models]
        colors = [results['model_comparisons'][model]['color'] for model in models]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distance vs Dimension
        for i, (name, distance, dim, color) in enumerate(zip(names, distances, dimensions, colors)):
            ax1.scatter(dim, distance, c=color, s=100, alpha=0.7, label=name)
            ax1.annotate(name, (dim, distance), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Embedding Dimension')
        ax1.set_ylabel('Cosine Distance (1 - Similarity)')
        ax1.set_title('Macron1 vs Macron2: Distance vs Embedding Dimension')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add trend line
        z = np.polyfit(dimensions, distances, 1)
        p = np.poly1d(z)
        ax1.plot(dimensions, p(dimensions), "r--", alpha=0.8, label=f'Trend line')
        
        # Plot 2: Similarity vs Dimension
        similarities = [1 - d for d in distances]
        for i, (name, similarity, dim, color) in enumerate(zip(names, similarities, dimensions, colors)):
            ax2.scatter(dim, similarity, c=color, s=100, alpha=0.7, label=name)
            ax2.annotate(name, (dim, similarity), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Embedding Dimension')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Macron1 vs Macron2: Similarity vs Embedding Dimension')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(dimensions, similarities, 1)
        p = np.poly1d(z)
        ax2.plot(dimensions, p(dimensions), "r--", alpha=0.8, label=f'Trend line')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / "macron_distance_vs_dimension.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Distance vs dimension plot saved to: {plot_file}")
        plt.close()
    
    def plot_model_comparison_bar(self, results: Dict, output_path: str = "../outputs"):
        """
        Create bar chart comparing models
        
        Args:
            results: Analysis results
            output_path: Directory to save plots
        """
        if not results['model_comparisons']:
            print("No data to plot!")
            return
        
        # Prepare data
        models = list(results['model_comparisons'].keys())
        distances = [results['model_comparisons'][model]['cosine_distance'] for model in models]
        similarities = [results['model_comparisons'][model]['cosine_similarity'] for model in models]
        dimensions = [results['model_comparisons'][model]['embedding_dim'] for model in models]
        names = [results['model_comparisons'][model]['model_name'] for model in models]
        colors = [results['model_comparisons'][model]['color'] for model in models]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distance comparison
        bars1 = ax1.bar(range(len(names)), distances, color=colors, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Cosine Distance')
        ax1.set_title('Macron1 vs Macron2: Cosine Distance by Model')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, distance, dim in zip(bars1, distances, dimensions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{distance:.3f}\n({dim}D)', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Similarity comparison
        bars2 = ax2.bar(range(len(names)), similarities, color=colors, alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Macron1 vs Macron2: Cosine Similarity by Model')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, similarity, dim in zip(bars2, similarities, dimensions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{similarity:.3f}\n({dim}D)', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / "macron_model_comparison_bars.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Model comparison bars saved to: {plot_file}")
        plt.close()
    
    def create_summary_table(self, results: Dict, output_path: str = "../outputs"):
        """
        Create a summary table of results
        
        Args:
            results: Analysis results
            output_path: Directory to save table
        """
        if not results['model_comparisons']:
            print("No data for summary table!")
            return
        
        # Prepare data for table
        table_data = []
        for model_name, data in results['model_comparisons'].items():
            table_data.append({
                'Model': data['model_name'],
                'Dimension': data['embedding_dim'],
                'Cosine Distance': f"{data['cosine_distance']:.4f}",
                'Cosine Similarity': f"{data['cosine_similarity']:.4f}",
                'Macron1 Duration': f"{data['macron1_duration']:.2f}s",
                'Macron2 Duration': f"{data['macron2_duration']:.2f}s"
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Sort by dimension
        df = df.sort_values('Dimension')
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_path / "macron_comparison_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"Summary table saved to: {csv_file}")
        
        # Save as JSON
        json_file = output_path / "macron_comparison_summary.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
        
        # Print table
        print("\n" + "="*80)
        print("MACRON1 vs MACRON2 COMPARISON SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        if results['summary_stats']:
            stats = results['summary_stats']
            print(f"\nOverall Statistics:")
            print(f"  Mean Distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")
            print(f"  Mean Similarity: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}")
            print(f"  Mean Dimension: {stats['mean_dimension']:.1f} ± {stats['std_dimension']:.1f}")
    
    def run_analysis(self, output_path: str = "../outputs"):
        """
        Run complete analysis and create all visualizations
        
        Args:
            output_path: Directory to save outputs
        """
        print("Starting Macron1 vs Macron2 comparison analysis...")
        
        # Run analysis
        results = self.analyze_macron_comparison()
        
        if not results:
            print("No results to analyze!")
            return
        
        # Create visualizations
        self.plot_distance_vs_dimension(results, output_path)
        self.plot_model_comparison_bar(results, output_path)
        self.create_summary_table(results, output_path)
        
        print("\nAnalysis completed! Check the outputs directory for results.")


def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Macron1 vs Macron2 embeddings across models')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Directory containing model outputs')
    parser.add_argument('--save_dir', type=str, default='../outputs',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = MacronComparisonVisualizer(output_dir=args.output_dir)
    
    # Run analysis
    visualizer.run_analysis(output_path=args.save_dir)


if __name__ == "__main__":
    main() 