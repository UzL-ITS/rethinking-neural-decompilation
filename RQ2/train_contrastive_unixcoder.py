#!/usr/bin/env python3
"""
Finetune UniXcoder with contrastive learning for (Ghidra, Source) matching.

Uses InfoNCE loss to learn embeddings where:
- Ghidra pseudo-code is close to its corresponding source code
- Ghidra pseudo-code is far from other source codes

Training data: LLM4Binary/decompile-eval (excluding first 100 test samples)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unixcoder import UniXcoder


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    # Data
    data_path: str = str(Path(__file__).resolve().parent / "../../LLM4Decompile/legacy-test/decompile-eval-executable-gcc-ghidra.json")
    test_samples: int = 100  # First N samples reserved for testing
    max_length: int = 512

    # Model
    model_name: str = "microsoft/unixcoder-base"
    projection_dim: int = 256  # Project embeddings to this dim

    # Training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 0.07  # InfoNCE temperature
    weight_decay: float = 0.01

    # Output
    output_dir: str = "./checkpoints"
    save_every: int = 2


# ============================================================================
# Dataset
# ============================================================================

class ContrastiveDataset(Dataset):
    """Dataset of (ghidra, source) pairs for contrastive learning."""

    def __init__(self, data: List[Dict], max_length: int = 512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'ghidra': item['input_asm_prompt'],
            'source': item['c_func'],
            'idx': idx
        }


def collate_fn(batch, tokenizer, max_length=512):
    """Collate batch with tokenization."""
    ghidra_texts = [item['ghidra'] for item in batch]
    source_texts = [item['source'] for item in batch]
    indices = [item['idx'] for item in batch]

    # Tokenize using UniXcoder's tokenizer
    ghidra_tokens = tokenizer.tokenize(ghidra_texts, max_length=max_length, padding=True, mode="<encoder-only>")
    source_tokens = tokenizer.tokenize(source_texts, max_length=max_length, padding=True, mode="<encoder-only>")

    return {
        'ghidra_ids': torch.tensor(ghidra_tokens),
        'source_ids': torch.tensor(source_tokens),
        'indices': torch.tensor(indices)
    }


# ============================================================================
# Model
# ============================================================================

class ContrastiveUniXcoder(nn.Module):
    """UniXcoder with projection head for contrastive learning."""

    def __init__(self, model_name: str, projection_dim: int = 256):
        super().__init__()

        # Load base UniXcoder
        self.encoder = UniXcoder(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for base

        # Projection head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode input and return projected embeddings."""
        _, sentence_embeddings = self.encoder(input_ids)
        projected = self.projection(sentence_embeddings)
        # L2 normalize
        projected = F.normalize(projected, p=2, dim=-1)
        return projected

    def forward(self, ghidra_ids: torch.Tensor, source_ids: torch.Tensor):
        """Forward pass returning both embeddings."""
        ghidra_emb = self.encode(ghidra_ids)
        source_emb = self.encode(source_ids)
        return ghidra_emb, source_emb


# ============================================================================
# Loss
# ============================================================================

def info_nce_loss(ghidra_emb: torch.Tensor, source_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE loss.

    For each ghidra embedding, its positive is the corresponding source.
    All other sources in the batch are negatives.
    """
    batch_size = ghidra_emb.shape[0]
    device = ghidra_emb.device

    # Similarity matrix [B, B]
    logits = torch.matmul(ghidra_emb, source_emb.T) / temperature

    # Labels: diagonal is positive
    labels = torch.arange(batch_size, device=device)

    # Cross entropy loss (both directions)
    loss_ghidra = F.cross_entropy(logits, labels)
    loss_source = F.cross_entropy(logits.T, labels)

    return (loss_ghidra + loss_source) / 2


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        ghidra_ids = batch['ghidra_ids'].to(device)
        source_ids = batch['source_ids'].to(device)

        # Forward
        ghidra_emb, source_emb = model(ghidra_ids, source_ids)

        # Loss
        loss = info_nce_loss(ghidra_emb, source_emb, config.temperature)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    return total_loss / num_batches


def evaluate(model, dataloader, config, device):
    """Evaluate accuracy: what fraction of ghidra codes match their correct source."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ghidra_ids = batch['ghidra_ids'].to(device)
            source_ids = batch['source_ids'].to(device)

            ghidra_emb, source_emb = model(ghidra_ids, source_ids)

            # Compute similarities
            similarities = torch.matmul(ghidra_emb, source_emb.T)

            # Check if argmax matches diagonal
            predictions = similarities.argmax(dim=1)
            targets = torch.arange(len(predictions), device=device)

            correct += (predictions == targets).sum().item()
            total += len(predictions)

    return correct / total if total > 0 else 0


# ============================================================================
# Main
# ============================================================================

def main():
    config = TrainConfig()

    print("=" * 70)
    print("CONTRASTIVE UNIXCODER TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Projection dim: {config.projection_dim}")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading data from {config.data_path}")
    with open(config.data_path, 'r') as f:
        all_data = json.load(f)

    # Split: first N samples for testing, rest for training
    test_data = all_data[:config.test_samples]
    train_data = all_data[config.test_samples:]

    print(f"  Test samples (excluded): {len(test_data)}")
    print(f"  Training samples: {len(train_data)}")

    # Create datasets
    train_dataset = ContrastiveDataset(train_data, config.max_length)
    test_dataset = ContrastiveDataset(test_data, config.max_length)

    # Initialize model for tokenizer
    print("\nLoading model...")
    model = ContrastiveUniXcoder(config.model_name, config.projection_dim)
    model = model.to(device)

    # Create dataloaders with custom collate
    def make_collate(tokenizer, max_length):
        def collate(batch):
            return collate_fn(batch, tokenizer, max_length)
        return collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_collate(model.encoder, config.max_length),
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=make_collate(model.encoder, config.max_length),
        num_workers=4,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Create output directory
    Path(config.output_dir).mkdir(exist_ok=True)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_acc = 0
    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, device, epoch)
        print(f"\nEpoch {epoch+1}/{config.epochs} - Train loss: {train_loss:.4f}")

        # Evaluate on held-out test set
        test_acc = evaluate(model, test_loader, config, device)
        print(f"Test accuracy (matching): {test_acc*100:.1f}%")

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or test_acc > best_acc:
            checkpoint_path = Path(config.output_dir) / f"contrastive_unixcoder_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_acc': test_acc,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_path = Path(config.output_dir) / "contrastive_unixcoder_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'test_acc': test_acc,
                    'config': config
                }, best_path)
                print(f"New best model: {best_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best test accuracy: {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()
