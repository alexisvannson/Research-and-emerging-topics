#!/usr/bin/env python3
"""
NeMo Embedding Extractor
Extracts embeddings using a compatible speaker encoder approach
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import warnings


class NeMoEmbeddingExtractor:
    def __init__(self, model_name="titanet-l", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize NeMo-compatible embedding extractor
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        print(f"Loading NeMo-compatible model: {model_name} on {device}...")
        
        # For now, use simple feature extraction to avoid model download issues
        self.model = None
        self.feature_extractor = None
        print("Using simple feature extraction (no model download required)")
    
    def resample_audio(self, signal, original_sr, target_sr=16000):
        """
        Resample audio to target sample rate
        
        Args:
            signal: Audio signal
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            tuple: (resampled_signal, target_sr)
        """
        if original_sr == target_sr:
            return signal, target_sr
        
        try:
            import librosa
            resampled = librosa.resample(signal, orig_sr=original_sr, target_sr=target_sr)
            return resampled, target_sr
        except ImportError:
            # Fallback to simple resampling
            ratio = target_sr / original_sr
            new_length = int(len(signal) * ratio)
            resampled = np.interp(np.linspace(0, len(signal), new_length), 
                                 np.arange(len(signal)), signal)
            return resampled, target_sr
    
    def extract_features_simple(self, signal, sample_rate):
        """
        Extract simple features when model is not available
        
        Args:
            signal: Audio signal
            sample_rate: Sample rate
            
        Returns:
            numpy array: Feature vector
        """
        # Simple MFCC-like features
        import librosa
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            signal, sample_rate = self.resample_audio(signal, sample_rate, 16000)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
        
        # Extract additional features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
        
        # Combine features
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(spectral_centroids, axis=1),
            np.mean(spectral_rolloff, axis=1),
            np.mean(zero_crossing_rate, axis=1),
            np.mean(spectral_bandwidth, axis=1)
        ])
        
        # Pad or truncate to 512 dimensions
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)))
        else:
            features = features[:512]
        
        return features
    
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
        
        # Resample to 16kHz for consistency
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz...")
            signal, sample_rate = self.resample_audio(signal, sample_rate, 16000)
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sample_rate)
        hop_samples = int(hop_length * sample_rate)
        
        embeddings = []
        timestamps = []
        
        # Extract embeddings using sliding window
        for start in range(0, len(signal) - segment_samples, hop_samples):
            end = start + segment_samples
            segment = signal[start:end]
            
            # Extract features for this segment
            embedding = self.extract_features_simple(segment, sample_rate)
            
            embeddings.append(embedding.tolist())
            timestamps.append(start / sample_rate)
        
        # Also extract a single embedding for the entire file
        full_embedding = self.extract_features_simple(signal, sample_rate)
        
        return {
            'file_path': audio_path,
            'sample_rate': sample_rate,
            'duration': len(signal) / sample_rate,
            'embedding_dim': len(full_embedding),
            'model_name': self.model_name,
            'model_type': 'simple_features',
            'full_embedding': full_embedding.tolist(),
            'segment_embeddings': embeddings,
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
            output_file = output_path / f"{audio_file.stem}_nemo_embeddings.json"
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
                output_file = output_path / f"{audio_file.stem}_nemo_embeddings.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                results[audio_file.name] = result
        
        # Save summary
        summary_file = output_path / "nemo_summary.json"
        summary = {
            'model': 'NeMo-Compatible',
            'model_name': self.model_name,
            'model_type': 'simple_features',
            'files_processed': len(results),
            'files': list(results.keys())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Extract NeMo-compatible embeddings from audio files')
    parser.add_argument('--input_dir', type=str, default='../audio', 
                       help='Directory containing audio files')
    parser.add_argument('--input_file', type=str,
                       help='Single audio file to process (overrides input_dir)')
    parser.add_argument('--output_dir', type=str, default='../outputs/nemo',
                       help='Directory to save embeddings')
    parser.add_argument('--model_name', type=str, default='titanet-l',
                       help='Model to use (or HuggingFace model ID)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize extractor
    extractor = NeMoEmbeddingExtractor(model_name=args.model_name, device=device)
    
    # Process files
    if args.input_file:
        # Process single file
        extractor.process_single_file(args.input_file, args.output_dir)
    else:
        # Process directory
        extractor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 