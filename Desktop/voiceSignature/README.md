# Audio Embedding Extraction Pipeline

This project implements audio embedding extraction using multiple state-of-the-art models for voice signature analysis.

## 🎯 Models Implemented

| Model | Dimensions | Purpose | Implementation |
|-------|------------|---------|----------------|
| **SpeechBrain ECAPA** | 192D | Speaker verification | `scripts/ecapa_embed.py` |
| **Wav2Vec2** | 768D | General-purpose audio embeddings | `scripts/wav2vec2_embed.py` |
| **HuBERT** | 768D | Advanced audio representation | `scripts/hubert_embed.py` |
| **Resemblyzer** | 256D | Voice encoding | `scripts/resemblyzer_embed.py` |
| **NeMo** | Variable | NVIDIA NeMo embeddings | `scripts/nemo_embed.py` |

## 📁 Project Structure

```
voiceSignature/
├── audio/                    # Input audio files
├── scripts/                  # Embedding extraction scripts
│   ├── ecapa_embed.py       # SpeechBrain ECAPA-TDNN
│   ├── wav2vec2_embed.py    # Wav2Vec2 embeddings
│   ├── hubert_embed.py      # HuBERT embeddings
│   ├── resemblyzer_embed.py # Resemblyzer voice encoding
│   ├── nemo_embed.py        # NeMo embeddings
│   ├── run_all_embeddings.py # Master script
│   └── test_installation.py # Installation test
├── outputs/                  # Generated embeddings
│   ├── ecapa/               # ECAPA embeddings
│   ├── wav2vec2/            # Wav2Vec2 embeddings
│   ├── hubert/              # HuBERT embeddings
│   ├── resemblyzer/         # Resemblyzer embeddings
│   └── nemo/                # NeMo embeddings
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd voiceSignature
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Option 1: Run All Models (Recommended)

Use the master script to run all models:

**Process all files in a directory:**
```bash
cd scripts
python run_all_embeddings.py --input_dir ../audio --output_dir ../outputs
```

**Process a single file:**
```bash
cd scripts
python run_all_embeddings.py --input_file ../audio/Macron1.wav --output_dir ../outputs
```

**Options:**
- `--input_dir`: Directory containing audio files (default: `../audio`)
- `--input_file`: Single audio file to process (overrides input_dir)
- `--output_dir`: Directory to save embeddings (default: `../outputs`)
- `--device`: Device to use (`cuda`, `cpu`, or `auto`)
- `--models`: Specific models to run (e.g., `--models ecapa wav2vec2`)
- `--parallel`: Run models in parallel (experimental)

### Option 2: Run Individual Models

#### SpeechBrain ECAPA-TDNN (192D)
```bash
cd scripts
# Process all files
python ecapa_embed.py --input_dir ../audio --output_dir ../outputs/ecapa
# Process single file
python ecapa_embed.py --input_file ../audio/Macron1.wav --output_dir ../outputs/ecapa
```

#### Wav2Vec2 (768D)
```bash
cd scripts
# Process all files
python wav2vec2_embed.py --input_dir ../audio --output_dir ../outputs/wav2vec2
# Process single file
python wav2vec2_embed.py --input_file ../audio/Macron1.wav --output_dir ../outputs/wav2vec2
```

#### HuBERT (768D)
```bash
cd scripts
# Process all files
python hubert_embed.py --input_dir ../audio --output_dir ../outputs/hubert
# Process single file
python hubert_embed.py --input_file ../audio/Macron1.wav --output_dir ../outputs/hubert
```

#### Resemblyzer (256D)
```bash
cd scripts
# Process all files
python resemblyzer_embed.py --input_dir ../audio --output_dir ../outputs/resemblyzer
# Process single file
python resemblyzer_embed.py --input_file ../audio/Macron1.wav --output_dir ../outputs/resemblyzer
```

#### NeMo (Variable dimensions)
```bash
cd scripts
# Process all files
python nemo_embed.py --input_dir ../audio --output_dir ../outputs/nemo
# Process single file
python nemo_embed.py --input_file ../audio/Macron1.wav --output_dir ../outputs/nemo
```

## 📈 Output Format

Each model generates JSON files with the following structure:

```json
{
  "file_path": "path/to/audio.wav",
  "sample_rate": 16000,
  "duration": 120.5,
  "embedding_dim": 192,
  "full_embedding": [...],
  "segment_embeddings": [
    [...],  // Embedding for segment 1
    [...],  // Embedding for segment 2
    ...
  ],
  "segment_timestamps": [0.0, 1.5, 3.0, ...],
  "segment_length": 3.0,
  "hop_length": 1.5
}
```

### Output Files

- **Individual embeddings**: `{filename}_{model}_embeddings.json`
- **Summary**: `{model}_summary.json`
- **Master summary**: `extraction_summary.json`

## 🔧 Model Details

### SpeechBrain ECAPA-TDNN
- **Dimensions**: 192D
- **Purpose**: Speaker verification
- **Model**: `speechbrain/spkrec-ecapa-voxceleb`
- **Best for**: Speaker identification, voice biometrics

### Wav2Vec2
- **Dimensions**: 768D
- **Purpose**: General-purpose audio embeddings
- **Model**: `facebook/wav2vec2-base`
- **Best for**: General audio analysis, speech recognition

### HuBERT
- **Dimensions**: 768D
- **Purpose**: Advanced audio representation
- **Model**: `facebook/hubert-base-ls960`
- **Best for**: Self-supervised learning, speech understanding

### Resemblyzer
- **Dimensions**: 256D
- **Purpose**: Voice encoding
- **Model**: VoiceEncoder from resemblyzer
- **Best for**: Voice cloning, speaker similarity

### NeMo
- **Dimensions**: Variable (model-dependent)
- **Purpose**: NVIDIA NeMo toolkit embeddings
- **Model**: `titanet-l`
- **Best for**: Production-scale audio processing

## ⚙️ Configuration

### Device Selection
- **Auto**: Automatically detects CUDA availability
- **CUDA**: Forces GPU usage (requires CUDA-compatible GPU)
- **CPU**: Forces CPU usage (slower but more compatible)

### Segment Parameters
All models use sliding window extraction with:
- **Segment length**: 3.0 seconds (default)
- **Hop length**: 1.5 seconds (default)

These can be modified in each script's `extract_embedding` method.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--device cpu` or reduce batch size
2. **Model download issues**: Check internet connection and disk space
3. **Audio format issues**: Ensure audio files are in supported formats (WAV, MP3, FLAC, etc.)

### Dependencies Issues

- **SpeechBrain**: `pip install speechbrain`
- **Transformers**: `pip install transformers`
- **Resemblyzer**: `pip install resemblyzer`
- **NeMo**: `pip install nemo-toolkit[asr]`

## 📊 Performance

Expected processing times (approximate, on GPU):
- **ECAPA**: ~2-3 seconds per minute of audio
- **Wav2Vec2**: ~1-2 seconds per minute of audio
- **HuBERT**: ~1-2 seconds per minute of audio
- **Resemblyzer**: ~3-4 seconds per minute of audio
- **NeMo**: ~2-3 seconds per minute of audio

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SpeechBrain team for ECAPA-TDNN
- Facebook AI Research for Wav2Vec2 and HuBERT
- Resemblyzer contributors
- NVIDIA for NeMo toolkit 