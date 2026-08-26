#!/usr/bin/env python3
"""
Fine-tune LLM4Decompile-9b on obfuscation dataset.

This script:
1. Loads the obfuscation training data
2. De-duplicates by Ghidra code to avoid training bias
3. Removes any samples that overlap with the test set
4. Fine-tunes LLM4Decompile-9b for 1 epoch
5. Reports training loss
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

_SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = "LLM4Binary/llm4decompile-9b-v2"
TRAIN_DATA = _SCRIPT_DIR / "../RQ3_obfuscation_finetuning/train-test-split_v2/train_non_empty_functions.json"
TEST_DATA = _SCRIPT_DIR / "../RQ3_obfuscation_finetuning/train-test-split_v2/test_non_empty_functions.json"
OUTPUT_DIR = _SCRIPT_DIR / "checkpoints/llm4decompile-9b-obf-epoch1"
MAX_LENGTH = 4096
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
LOGGING_STEPS = 50
IGNORE_INDEX = -100


# =============================================================================
# Prompt formatting (following LLM4Decompile format)
# =============================================================================
def build_instruction_prompt(ghidra_code: str) -> str:
    """Build the instruction prompt for LLM4Decompile."""
    return f"""# This is the assembly code:
{ghidra_code}
# What is the source code?
"""


def normalize_code(code: str) -> str:
    """Normalize code for comparison (remove extra whitespace, normalize newlines)."""
    if not code:
        return ""
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in code.split('\n')]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


def hash_code(code: str) -> str:
    """Create a hash of normalized code for deduplication."""
    normalized = normalize_code(code)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def compute_total_sequence_length(ghidra_code: str, source_code: str, tokenizer) -> int:
    """Compute full training sequence length before truncation."""
    prompt = build_instruction_prompt(ghidra_code)
    target = f"{source_code}\n{tokenizer.eos_token}"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_tensors=None,
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
    )["input_ids"]

    return len(prompt_ids) + len(target_ids)


# =============================================================================
# Data loading and cleaning
# =============================================================================
def load_and_clean_data(tokenizer):
    """Load training data, deduplicate, and remove test set overlap."""
    print("=" * 70)
    print("LOADING AND CLEANING DATA")
    print("=" * 70)

    print(f"\nLoading training data from {TRAIN_DATA}...")
    # df = pd.read_parquet(TRAIN_DATA)
    df = pd.read_json(TRAIN_DATA)
    original_count = len(df)
    print(f"  Original samples: {original_count:,}")

    print(f"\nLoading test data from {TEST_DATA}...")
    with open(TEST_DATA, 'r') as f:
        test_data = json.load(f)
    print(f"  Test samples: {len(test_data):,}")

    print("\nFiltering test samples by full sequence length...")
    filtered_test_data = []
    skipped_test_long = 0
    for sample in tqdm(test_data, desc="Filtering test lengths"):
        total_len = compute_total_sequence_length(
            sample.get('input_asm_prompt', ''),
            sample.get('c_func', ''),
            tokenizer,
        )
        if total_len <= MAX_LENGTH:
            filtered_test_data.append(sample)
        else:
            skipped_test_long += 1
    test_data = filtered_test_data
    print(f"  Skipped {skipped_test_long:,} overlength test samples")
    print(f"  Test samples within limit: {len(test_data):,}")

    test_source_hashes = set()
    for sample in test_data:
        c_func = sample.get('c_func', '')
        test_source_hashes.add(hash_code(c_func))
    print(f"  Unique test source hashes: {len(test_source_hashes):,}")

    # Step 1: De-duplicate by Ghidra code (c_code_obf)
    print("\nStep 1: De-duplicating by Ghidra code...")
    df['ghidra_hash'] = df['input_asm_prompt'].apply(hash_code)
    df_dedup = df.drop_duplicates(subset=['ghidra_hash'], keep='first').copy()
    dedup_count = len(df_dedup)
    print(f"  After Ghidra dedup: {dedup_count:,} ({original_count - dedup_count:,} removed)")

    # Step 2: Remove samples that overlap with test set
    print("\nStep 2: Removing test set overlap...")
    df_dedup['source_hash'] = df_dedup['c_func'].apply(hash_code)
    df_clean = df_dedup[~df_dedup['source_hash'].isin(test_source_hashes)].copy()
    clean_count = len(df_clean)
    removed_test = dedup_count - clean_count
    print(f"  After test removal: {clean_count:,} ({removed_test:,} test overlaps removed)")

    print("\nStep 3: Filtering training samples by full sequence length...")
    valid_indices = []
    skipped_train_long = 0
    for idx in tqdm(range(len(df_clean)), desc="Filtering train lengths"):
        row = df_clean.iloc[idx]
        total_len = compute_total_sequence_length(
            row['input_asm_prompt'],
            row['c_func'],
            tokenizer,
        )
        if total_len <= MAX_LENGTH:
            valid_indices.append(idx)
        else:
            skipped_train_long += 1
    df_clean = df_clean.iloc[valid_indices].copy()
    filtered_count = len(df_clean)
    print(f"  Skipped {skipped_train_long:,} overlength train samples")
    print(f"  Final training size within limit: {filtered_count:,}")

    # Summary
    print("\n" + "=" * 70)
    print("DATA CLEANING SUMMARY")
    print("=" * 70)
    print(f"  Original samples:      {original_count:>10,}")
    print(f"  After Ghidra dedup:    {dedup_count:>10,} (-{original_count - dedup_count:,})")
    print(f"  After test removal:    {clean_count:>10,} (-{removed_test:,})")
    print(f"  After length filter:   {filtered_count:>10,} (-{skipped_train_long:,})")
    print(f"  Final training size:   {filtered_count:>10,}")

    print("\n  Obfuscation pass distribution:")
    for pass_name, count in df_clean['obf_pass'].value_counts().items():
        print(f"    {str(pass_name):20s}: {count:>8,} ({count/clean_count*100:.1f}%)")

    print("\n  Optimization level distribution:")
    for opt_level, count in df_clean['type'].value_counts().items():
        print(f"    {opt_level:20s}: {count:>8,} ({count/clean_count*100:.1f}%)")

    return df_clean

def check_sequence_lengths(df, tokenizer):
    """Report how many samples exceed MAX_LENGTH before truncation."""
    too_long = 0
    max_len_seen = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking lengths"):
        total_len = compute_total_sequence_length(
            row['input_asm_prompt'],
            row['c_func'],
            tokenizer,
        )
        max_len_seen = max(max_len_seen, total_len)

        if total_len > MAX_LENGTH:
            too_long += 1

    print(f"\nSamples longer than {MAX_LENGTH}: {too_long}")
    print(f"Maximum sequence length: {max_len_seen}")
# =============================================================================
# Dataset class
# =============================================================================
class ObfuscationDataset(Dataset):
    """Dataset for obfuscation fine-tuning."""

    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ghidra_code = row['input_asm_prompt']
        source_code = row['c_func']

        prompt = build_instruction_prompt(ghidra_code)
        target = f"{source_code}\n{self.tokenizer.eos_token}"

        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_tokens['input_ids'])

        target_tokens = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=False,
            return_tensors=None,
        )

        input_ids = prompt_tokens['input_ids'] + target_tokens['input_ids']

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Encountered overlength sample after filtering: {len(input_ids)} > {MAX_LENGTH}"
            )

        labels = input_ids.copy()
        mask_len = min(prompt_len, len(labels))
        labels[:mask_len] = [IGNORE_INDEX] * mask_len

        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


# =============================================================================
# Data collator
# =============================================================================
@dataclass
class CustomDataCollator:
    """Collate examples for supervised fine-tuning with padding."""

    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(inst['input_ids']) for inst in instances]
        labels = [torch.tensor(inst['labels']) for inst in instances]
        attention_mask = [torch.tensor(inst['attention_mask']) for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


# =============================================================================
# Main training function
# =============================================================================
def main():
    print("=" * 70)
    print("LLM4DECOMPILE OBFUSCATION FINE-TUNING")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("\n" + "=" * 70)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 70)

    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    df_clean = load_and_clean_data(tokenizer)
    check_sequence_lengths(df_clean, tokenizer)

    print(f"\nLoading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    print(f"  Model loaded with {model.num_parameters():,} parameters")
    model.gradient_checkpointing_enable()
    print("\n" + "=" * 70)
    print("CREATING DATASET")
    print("=" * 70)

    dataset = ObfuscationDataset(df_clean, tokenizer)
    print(f"\nDataset size: {len(dataset):,} samples")

    print("\nSample training example:")
    sample = dataset[0]
    print(f"  Input length: {len(sample['input_ids'])} tokens")
    print(f"  Labels (non-masked): {sum(1 for x in sample['labels'] if x != IGNORE_INDEX)} tokens")
    decoded = tokenizer.decode(sample['input_ids'][:200])
    print(f"  First 200 tokens: {decoded[:300]}...")

    data_collator = CustomDataCollator(tokenizer=tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        optim="adamw_bnb_8bit",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nTraining for 1 epoch...")
    print("(Loss will be reported every 50 steps)")
    print("-" * 70)

    trainer.train()

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    trainer.save_model()
    trainer.save_state()
    print(f"\nModel saved to {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
