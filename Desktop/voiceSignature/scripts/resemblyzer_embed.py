#!/usr/bin/env python3
"""
Resemblyzer Embedding Extractor
Extracts 256-dimensional voice encoding embeddings
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
import librosa


class ResemblyzerEmbeddingExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Resemblyzer embedding extractor
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Loading Resemblyzer model on {device}...")
        
        try:
            # Import resemblyzer components
            from resemblyzer import VoiceEncoder, preprocess_wav
            from pathlib import Path
            
            self.voice_encoder = VoiceEncoder(device=device)
            self.preprocess_wav = preprocess_wav
            print("Resemblyzer model loaded successfully!")
            
        except ImportError as e:
            print(f"Error importing Resemblyzer: {e}")
            print("Please install resemblyzer: pip install resemblyzer")
            raise
    
    def normalize_audio(self, audio_path):
        """
        Normalize audio to prevent warnings and improve quality
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            tuple: (normalized_audio, sample_rate)
        """
        try:
            # Load audio with librosa for better normalization
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Normalize audio to prevent warnings
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Ensure audio is not too quiet
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                print(f"Warning: Audio {audio_path} is very quiet (RMS: {rms:.4f})")
            
            return audio, sr
        except Exception as e:
            print(f"Error normalizing audio {audio_path}: {e}")
            # Fallback to soundfile
            return sf.read(audio_path)
    
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
        
        # Load and normalize audio
        try:
            signal, sample_rate = self.normalize_audio(audio_path)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None
        
        # Preprocess audio for Resemblyzer
        try:
            # Save normalized audio temporarily
            temp_path = f"temp_normalized_{os.path.basename(audio_path)}"
            sf.write(temp_path, signal, sample_rate)
            
            # Preprocess with resemblyzer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav = self.preprocess_wav(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sample_rate)
        hop_samples = int(hop_length * sample_rate)
        
        embeddings = []
        timestamps = []
        
        # Extract embeddings using sliding window
        for start in range(0, len(signal) - segment_samples, hop_samples):
            end = start + segment_samples
            segment = signal[start:end]
            
            # Save temporary segment file
            temp_segment_path = f"temp_segment_{start}.wav"
            sf.write(temp_segment_path, segment, sample_rate)
            
            try:
                # Preprocess segment with warnings suppressed
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    segment_wav = self.preprocess_wav(temp_segment_path)
                
                # Extract embedding for this segment
                embedding = self.voice_encoder.embed_utterance(segment_wav)
                
                embeddings.append(embedding.tolist())
                timestamps.append(start / sample_rate)
                
            except Exception as e:
                print(f"Error processing segment at {start}: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_segment_path):
                    os.remove(temp_segment_path)
        
        # Also extract a single embedding for the entire file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_embedding = self.voice_encoder.embed_utterance(wav)
                full_embedding = full_embedding.tolist()
        except Exception as e:
            print(f"Error extracting full embedding: {e}")
            full_embedding = None
        
        return {
            'file_path': audio_path,
            'sample_rate': sample_rate,
            'duration': len(signal) / sample_rate,
            'embedding_dim': 256 if full_embedding else None,
            'full_embedding': full_embedding,
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
            output_file = output_path / f"{audio_file.stem}_resemblyzer_embeddings.json"
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
                output_file = output_path / f"{audio_file.stem}_resemblyzer_embeddings.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                results[audio_file.name] = result
        
        # Save summary
        summary_file = output_path / "resemblyzer_summary.json"
        summary = {
            'model': 'Resemblyzer',
            'embedding_dim': 256,
            'files_processed': len(results),
            'files': list(results.keys())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Extract Resemblyzer embeddings from audio files')
    parser.add_argument('--input_dir', type=str, default='../audio', 
                       help='Directory containing audio files')
    parser.add_argument('--input_file', type=str,
                       help='Single audio file to process (overrides input_dir)')
    parser.add_argument('--output_dir', type=str, default='../outputs/resemblyzer',
                       help='Directory to save embeddings')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize extractor
    extractor = ResemblyzerEmbeddingExtractor(device=device)
    
    # Process files
    if args.input_file:
        # Process single file
        extractor.process_single_file(args.input_file, args.output_dir)
    else:
        # Process directory
        extractor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 