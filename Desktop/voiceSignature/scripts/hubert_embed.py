#!/usr/bin/env python3
"""
HuBERT Embedding Extractor
Extracts 768-dimensional advanced audio representation embeddings
"""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from transformers import HubertModel, AutoFeatureExtractor
from pathlib import Path
import argparse
import json
from tqdm import tqdm


class HuBERTEmbeddingExtractor:
    def __init__(self, model_name="facebook/hubert-base-ls960", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize HuBERT embedding extractor
        
        Args:
            model_name: Name of the HuBERT model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        print(f"Loading HuBERT model: {model_name} on {device}...")
        
        # Load the HuBERT model and feature extractor
        self.model = HubertModel.from_pretrained(model_name).to(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("HuBERT model loaded successfully!")
    
    def extract_embedding(self, audio_path, segment_length=3.0, hop_length=1.5):
        """
        Extract embeddings from audio file using sliding window approach
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of each segment in seconds
            hop_length: Hop length between segments in seconds
            
        Returns:
            dict: Dictionary containing embeddings and metadata
        """
        print(f"Processing: {audio_path}")
        
        # Load audio
        try:
            signal, sample_rate = sf.read(audio_path)
            if len(signal.shape) > 1:
                signal = signal[:, 0]  # Convert to mono if stereo
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None
        
        # Resample to 16kHz if necessary (HuBERT expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            signal = torch.tensor(signal, dtype=torch.float32)
            signal = resampler(signal)
            sample_rate = 16000
        else:
            signal = torch.tensor(signal, dtype=torch.float32)
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sample_rate)
        hop_samples = int(hop_length * sample_rate)
        
        embeddings = []
        timestamps = []
        
        # Extract embeddings using sliding window
        for start in range(0, len(signal) - segment_samples, hop_samples):
            end = start + segment_samples
            segment = signal[start:end]
            
            # Prepare input for the model
            inputs = self.feature_extractor(
                segment, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Extract embedding for this segment
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state as embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            embeddings.append(embedding)
            timestamps.append(start / sample_rate)
        
        # Also extract a single embedding for the entire file
        inputs = self.feature_extractor(
            signal, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            full_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return {
            'file_path': audio_path,
            'sample_rate': sample_rate,
            'duration': len(signal) / sample_rate,
            'embedding_dim': len(full_embedding),
            'model_name': self.model_name,
            'full_embedding': full_embedding.tolist(),
            'segment_embeddings': [emb.tolist() for emb in embeddings],
            'segment_timestamps': timestamps,
            'segment_length': segment_length,
            'hop_length': hop_length
        }
    
    def process_single_file(self, audio_path, output_dir):
        """
        Process a single audio file
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save embeddings
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result = self.extract_embedding(audio_path)
        if result:
            # Save result
            audio_file = Path(audio_path)
            output_file = output_path / f"{audio_file.stem}_hubert_embeddings.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Embeddings saved to: {output_file}")
            return result
        return None
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save embeddings
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in audio_extensions]
        
        print(f"Found {len(audio_files)} audio files to process")
        
        results = {}
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            result = self.extract_embedding(str(audio_file))
            if result:
                # Save individual file result
                output_file = output_path / f"{audio_file.stem}_hubert_embeddings.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                results[audio_file.name] = result
        
        # Save summary
        summary_file = output_path / "hubert_summary.json"
        summary = {
            'model': 'HuBERT',
            'model_name': self.model_name,
            'embedding_dim': 768,
            'files_processed': len(results),
            'files': list(results.keys())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Extract HuBERT embeddings from audio files')
    parser.add_argument('--input_dir', type=str, default='../audio', 
                       help='Directory containing audio files')
    parser.add_argument('--input_file', type=str,
                       help='Single audio file to process (overrides input_dir)')
    parser.add_argument('--output_dir', type=str, default='../outputs/hubert',
                       help='Directory to save embeddings')
    parser.add_argument('--model_name', type=str, default='facebook/hubert-base-ls960',
                       help='HuBERT model to use')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize extractor
    extractor = HuBERTEmbeddingExtractor(model_name=args.model_name, device=device)
    
    # Process files
    if args.input_file:
        # Process single file
        extractor.process_single_file(args.input_file, args.output_dir)
    else:
        # Process directory
        extractor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 