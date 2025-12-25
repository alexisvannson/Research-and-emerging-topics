"""Download and prepare datasets for long document summarization."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


def download_arxiv(output_dir: Path, num_samples: Optional[int] = None) -> None:
    """Download arXiv dataset for scientific paper summarization.

    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to download (None for all)
    """
    print("Downloading arXiv dataset...")
    dataset = load_dataset("ccdv/arxiv-summarization", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    output_path = output_dir / "arxiv"
    dataset.save_to_disk(str(output_path))
    print(f"Saved arXiv dataset to {output_path}")


def download_pubmed(output_dir: Path, num_samples: Optional[int] = None) -> None:
    """Download PubMed dataset for scientific paper summarization.

    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to download (None for all)
    """
    print("Downloading PubMed dataset...")
    dataset = load_dataset("ccdv/pubmed-summarization", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    output_path = output_dir / "pubmed"
    dataset.save_to_disk(str(output_path))
    print(f"Saved PubMed dataset to {output_path}")


def download_multi_news(output_dir: Path, num_samples: Optional[int] = None) -> None:
    """Download Multi-News dataset for multi-document summarization.

    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to download (None for all)
    """
    print("Downloading Multi-News dataset...")
    dataset = load_dataset("multi_news", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    output_path = output_dir / "multi_news"
    dataset.save_to_disk(str(output_path))
    print(f"Saved Multi-News dataset to {output_path}")


def download_booksum(output_dir: Path, num_samples: Optional[int] = None) -> None:
    """Download BookSum dataset for long book summarization.

    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to download (None for all)
    """
    print("Downloading BookSum dataset...")
    try:
        dataset = load_dataset("kmfoda/booksum", split="train")

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        output_path = output_dir / "booksum"
        dataset.save_to_disk(str(output_path))
        print(f"Saved BookSum dataset to {output_path}")
    except Exception as e:
        print(f"Warning: Could not download BookSum: {e}")
        print("Skipping BookSum dataset...")


def download_billsum(output_dir: Path, num_samples: Optional[int] = None) -> None:
    """Download BillSum dataset for US Congressional bill summarization.

    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to download (None for all)
    """
    print("Downloading BillSum dataset...")
    dataset = load_dataset("billsum", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    output_path = output_dir / "billsum"
    dataset.save_to_disk(str(output_path))
    print(f"Saved BillSum dataset to {output_path}")


def download_cnn_dailymail(output_dir: Path, num_samples: Optional[int] = None) -> None:
    """Download CNN/DailyMail dataset (for baseline comparison).

    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to download (None for all)
    """
    print("Downloading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    output_path = output_dir / "cnn_dailymail"
    dataset.save_to_disk(str(output_path))
    print(f"Saved CNN/DailyMail dataset to {output_path}")


def get_dataset_statistics(dataset_path: Path) -> Dict[str, any]:
    """Calculate statistics for a downloaded dataset.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Dictionary containing dataset statistics
    """
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))

    stats = {
        "num_samples": len(dataset),
        "columns": dataset.column_names,
    }

    # Calculate length statistics if text columns exist
    text_column = None
    for col in ["article", "text", "document", "source"]:
        if col in dataset.column_names:
            text_column = col
            break

    if text_column:
        lengths = [len(sample[text_column].split()) for sample in dataset]
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["min_length"] = min(lengths)
        stats["max_length"] = max(lengths)

    return stats



def main() -> None:
    """Main function to download all datasets."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Long Document Summarization - Dataset Download")

    # Download datasets (using smaller samples for testing)
    # For full training, set num_samples=None

    for dataset_name, download_func in DATASETS_TO_DOWNLOAD.items():
        try:
            print(f"\n{'=' * 80}")
            download_func(output_dir, num_samples=1000)  # Download 1000 samples each

            # Print statistics
            dataset_path = output_dir / dataset_name
            if dataset_path.exists():
                stats = get_dataset_statistics(dataset_path)
                print(f"\nStatistics for {dataset_name}:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            print("Continuing with next dataset...")

    print("Dataset download complete!")
    print(f"Datasets saved to: {output_dir}")


if __name__ == "__main__":
    DATASETS_TO_DOWNLOAD = {
    "arxiv": download_arxiv,
    "billsum": download_billsum,
    # "pubmed": download_pubmed,
    # "multi_news": download_multi_news,
    # "booksum": download_booksum,
    # "cnn_dailymail": download_cnn_dailymail,
    }
    main()
