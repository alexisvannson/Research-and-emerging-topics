"""Preprocessing utilities for summarization data."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SummarizationDataset(Dataset):
    """PyTorch Dataset for summarization."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 1024,
        max_target_length: int = 256,
        source_key: str = "article",
        target_key: str = "summary",
    ):
        """Initialize dataset.

        Args:
            data: List of data samples
            tokenizer: Tokenizer for encoding
            max_source_length: Max source tokens
            max_target_length: Max target tokens
            source_key: Key for source text
            target_key: Key for target text
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_key = source_key
        self.target_key = target_key

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item.

        Args:
            idx: Index

        Returns:
            Dictionary with encoded inputs
        """
        sample = self.data[idx]

        # Get source and target text
        source_text = sample.get(self.source_key, "")
        target_text = sample.get(self.target_key, "")

        # Encode source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Encode target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


def load_processed_data(dataset_name: str, split: str, data_dir: Path) -> List[Dict]:
    """Load processed dataset.

    Args:
        dataset_name: Name of dataset
        split: Split name (train/val/test)
        data_dir: Data directory

    Returns:
        List of samples
    """
    file_path = data_dir / dataset_name / f"{split}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def create_dataloaders(
    config: Dict, tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train/val/test dataloaders.

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path("data/processed")

    # Load datasets
    datasets_list = config["data"]["datasets"]
    train_data = []
    val_data = []
    test_data = []

    for dataset_name in datasets_list:
        try:
            train_data.extend(load_processed_data(dataset_name, "train", data_dir))
            val_data.extend(load_processed_data(dataset_name, "val", data_dir))
            test_data.extend(load_processed_data(dataset_name, "test", data_dir))
        except FileNotFoundError:
            print(f"Warning: Dataset {dataset_name} not found, skipping...")

    # Limit samples if specified
    if "train_samples" in config["data"]:
        train_data = train_data[: config["data"]["train_samples"]]
    if "val_samples" in config["data"]:
        val_data = val_data[: config["data"]["val_samples"]]
    if "test_samples" in config["data"]:
        test_data = test_data[: config["data"]["test_samples"]]

    # Create datasets
    train_dataset = SummarizationDataset(
        train_data,
        tokenizer,
        max_source_length=config["data"]["max_source_length"],
        max_target_length=config["data"]["max_target_length"],
    )

    val_dataset = SummarizationDataset(
        val_data,
        tokenizer,
        max_source_length=config["data"]["max_source_length"],
        max_target_length=config["data"]["max_target_length"],
    )

    test_dataset = SummarizationDataset(
        test_data,
        tokenizer,
        max_source_length=config["data"]["max_source_length"],
        max_target_length=config["data"]["max_target_length"],
    )

    return train_dataset, val_dataset, test_dataset
