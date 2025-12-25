"""Preprocess datasets for long document summarization."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    # Split on double newlines
    paragraphs = re.split(r"\n\s*\n", text)
    # Remove empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    try:
        sentences = nltk.sent_tokenize(text)
        return sentences
    except LookupError:
        nltk.download("punkt")
        sentences = nltk.sent_tokenize(text)
        return sentences


def count_tokens(text: str) -> int:
    """Count approximate number of tokens (words) in text.

    Args:
        text: Input text

    Returns:
        Token count
    """
    return len(text.split())


def filter_by_length(
    dataset_path: Path,
    min_length: int = 5000,
    max_length: int = 15000,
    text_column: str = "article",
) -> List[Dict]:
    """Filter dataset by document length.

    Args:
        dataset_path: Path to dataset
        min_length: Minimum token count
        max_length: Maximum token count
        text_column: Name of text column

    Returns:
        Filtered samples
    """
    dataset = load_from_disk(str(dataset_path))

    filtered_samples = []
    for sample in tqdm(dataset, desc="Filtering by length"):
        if text_column not in sample:
            # Try to find the correct column name
            for col in ["article", "text", "document", "source", "input"]:
                if col in sample:
                    text_column = col
                    break

        text = sample.get(text_column, "")
        token_count = count_tokens(text)

        if min_length <= token_count <= max_length:
            filtered_samples.append(sample)

    return filtered_samples


def add_structure_info(sample: Dict, text_column: str = "article") -> Dict:
    """Add structural information to sample (paragraphs, sentences).

    Args:
        sample: Dataset sample
        text_column: Name of text column

    Returns:
        Sample with added structure info
    """
    text = sample.get(text_column, "")

    # Add paragraph and sentence splits
    paragraphs = split_into_paragraphs(text)
    sentences = split_into_sentences(text)

    sample["num_paragraphs"] = len(paragraphs)
    sample["num_sentences"] = len(sentences)
    sample["paragraphs"] = paragraphs
    sample["sentences"] = sentences
    sample["token_count"] = count_tokens(text)

    return sample


def create_splits(
    samples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets.

    Args:
        samples: List of samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        Tuple of (train, val, test) splits
    """
    np.random.seed(42)
    np.random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]

    return train, val, test


def preprocess_dataset(
    dataset_name: str,
    raw_dir: Path,
    processed_dir: Path,
    min_length: int = 5000,
    max_length: int = 15000,
) -> None:
    """Preprocess a single dataset.

    Args:
        dataset_name: Name of the dataset
        raw_dir: Directory with raw data
        processed_dir: Directory to save processed data
        min_length: Minimum token count
        max_length: Maximum token count
    """
    print(f"\nProcessing {dataset_name}...")

    dataset_path = raw_dir / dataset_name
    if not dataset_path.exists():
        print(f"Dataset {dataset_name} not found. Skipping...")
        return

    # Determine text column name based on dataset
    text_column_map = {
        "arxiv": "article",
        "pubmed": "article",
        "multi_news": "document",
        "booksum": "chapter",
        "billsum": "text",
        "cnn_dailymail": "article",
    }
    text_column = text_column_map.get(dataset_name, "article")

    # Filter by length
    filtered_samples = filter_by_length(
        dataset_path, min_length, max_length, text_column
    )
    print(f"Filtered to {len(filtered_samples)} samples")

    if len(filtered_samples) == 0:
        print(f"No samples after filtering for {dataset_name}")
        return

    # Add structure information
    processed_samples = []
    for sample in tqdm(filtered_samples, desc="Adding structure info"):
        processed_sample = add_structure_info(sample, text_column)
        processed_samples.append(processed_sample)

    # Create train/val/test splits
    train, val, test = create_splits(processed_samples)

    # Save processed data
    output_dir = processed_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name}: {len(split_data)} samples to {output_file}")

    # Save statistics
    stats = {
        "dataset_name": dataset_name,
        "total_samples": len(processed_samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "avg_token_count": np.mean([s["token_count"] for s in processed_samples]),
        "avg_paragraphs": np.mean([s["num_paragraphs"] for s in processed_samples]),
        "avg_sentences": np.mean([s["num_sentences"] for s in processed_samples]),
    }

    stats_file = output_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics for {dataset_name}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main() -> None:
    """Main preprocessing function."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Long Document Summarization - Data Preprocessing")

    # Download NLTK data if needed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Process each dataset
    datasets = ["arxiv", "pubmed", "multi_news", "booksum", "billsum", "cnn_dailymail"]

    for dataset_name in datasets:
        try:
            preprocess_dataset(dataset_name, raw_dir, processed_dir)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            print("Continuing with next dataset...")

    print("Preprocessing complete!")
    print(f"Processed data saved to: {processed_dir}")


if __name__ == "__main__":
    main()
