#!/usr/bin/env python3
"""
Test script to verify installation and model loading
"""

import sys
import importlib
from pathlib import Path


def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False


def test_torch():
    """Test PyTorch installation"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, will use CPU")
        return True
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False


def test_models():
    """Test model loading (without downloading)"""
    print("\n🔍 Testing model imports...")
    
    # Test SpeechBrain
    try:
        from speechbrain.pretrained import EncoderClassifier
        print("✅ SpeechBrain: Can import EncoderClassifier")
    except ImportError as e:
        print(f"❌ SpeechBrain: {e}")
    
    # Test Transformers
    try:
        from transformers import Wav2Vec2Model, HubertModel
        print("✅ Transformers: Can import Wav2Vec2 and HuBERT")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
    
    # Test Resemblyzer
    try:
        from resemblyzer import VoiceEncoder
        print("✅ Resemblyzer: Can import VoiceEncoder")
    except ImportError as e:
        print(f"❌ Resemblyzer: {e}")
    
    # Test NeMo
    try:
        import nemo.collections.asr as nemo_asr
        print("✅ NeMo: Can import nemo.collections.asr")
    except ImportError as e:
        print(f"❌ NeMo: {e}")


def main():
    print("🧪 Testing Audio Embedding Extraction Installation")
    print("=" * 50)
    
    # Test basic dependencies
    print("\n📦 Testing basic dependencies...")
    basic_deps = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("transformers", "Transformers"),
        ("speechbrain", "SpeechBrain"),
        ("librosa", "Librosa"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("soundfile", "SoundFile"),
        ("tqdm", "TQDM"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn")
    ]
    
    basic_results = []
    for module, desc in basic_deps:
        result = test_import(module, desc)
        basic_results.append(result)
    
    # Test PyTorch specifically
    print("\n🔥 Testing PyTorch...")
    torch_ok = test_torch()
    
    # Test model-specific dependencies
    print("\n🤖 Testing model-specific dependencies...")
    model_deps = [
        ("resemblyzer", "Resemblyzer"),
        ("nemo", "NeMo Toolkit"),
        ("omegaconf", "OmegaConf")
    ]
    
    model_results = []
    for module, desc in model_deps:
        result = test_import(module, desc)
        model_results.append(result)
    
    # Test model loading
    test_models()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 50)
    
    basic_success = sum(basic_results)
    model_success = sum(model_results)
    total_basic = len(basic_results)
    total_model = len(model_deps)
    
    print(f"Basic dependencies: {basic_success}/{total_basic} ✅")
    print(f"Model dependencies: {model_success}/{total_model} ✅")
    print(f"PyTorch: {'✅' if torch_ok else '❌'}")
    
    if basic_success == total_basic and model_success == total_model and torch_ok:
        print("\n🎉 All tests passed! Installation is complete.")
        print("\nYou can now run the embedding extraction scripts:")
        print("  cd scripts")
        print("  python run_all_embeddings.py --input_dir ../audio --output_dir ../outputs")
    else:
        print("\n⚠️  Some dependencies are missing. Please install them:")
        print("  pip install -r requirements.txt")
        
        if not torch_ok:
            print("\nFor PyTorch installation issues, visit:")
            print("  https://pytorch.org/get-started/locally/")
        
        if not model_results[0]:  # Resemblyzer
            print("\nFor Resemblyzer installation:")
            print("  pip install resemblyzer")
        
        if not model_results[1]:  # NeMo
            print("\nFor NeMo installation:")
            print("  pip install nemo-toolkit[asr]")


if __name__ == "__main__":
    main() 