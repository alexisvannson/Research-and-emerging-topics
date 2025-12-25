"""Training pipeline for summarization models."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from models.utils import AverageMeter, set_seed
from src.preprocessing import create_dataloaders


def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm

    Returns:
        Dictionary with training metrics
    """
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc="Training")

    for step, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update parameters
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update metrics
        loss_meter.update(loss.item() * gradient_accumulation_steps)
        pbar.set_postfix({"loss": loss_meter.avg})

    return {"train_loss": loss_meter.avg}


def validate(model, val_loader: DataLoader, device: str) -> Dict[str, float]:
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to use

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss_meter.update(loss.item())

    return {"val_loss": loss_meter.avg}


def train_model(config: Dict, output_dir: Path):
    """Main training function.

    Args:
        config: Training configuration
        output_dir: Directory to save checkpoints
    """
    # Set seed
    set_seed(config.get("seed", 42))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = config["abstractive"]["model_name"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Check if fine-tuning should be skipped
    skip_fine_tuning = config["training"].get("skip_fine_tuning", False)

    if skip_fine_tuning:
        print("Fine-tuning is disabled. Using pretrained model without training.")
        print(f"Saving pretrained model to {output_dir}")

        # Save the pretrained model directly
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Pretrained model saved to {output_dir}")
        return

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataset, val_dataset, test_dataset = create_dataloaders(config, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )

    num_training_steps = (
        len(train_loader) * int(config["training"]["num_epochs"])
    ) // int(config["training"].get("gradient_accumulation_steps", 1))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["training"].get("warmup_steps", 500)),
        num_training_steps=num_training_steps,
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(int(config["training"]["num_epochs"])):
        print(f"\nEpoch {epoch + 1}/{int(config['training']['num_epochs'])}")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_accumulation_steps=int(
                config["training"].get("gradient_accumulation_steps", 1)
            ),
            max_grad_norm=float(config["training"].get("max_grad_norm", 1.0)),
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Combine metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        history.append(metrics)

        # Print metrics
        print(f"Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")

        # Save checkpoint if best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint periodically
        if (epoch + 1) % config["logging"].get("save_interval", 1000) == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! History saved to {history_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train summarization model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Train model
    output_dir = Path(args.output_dir) / config["model"]["name"]
    train_model(config, output_dir)


if __name__ == "__main__":
    main()
