#!/usr/bin/env python3
"""
Train a lightweight reranker from existing pipeline caches and export scores.

This script learns to predict candidate quality without running execution at
inference time. It uses:
- phase1_compilation.pkl
- phase3_5_execution_guided.pkl (primary regression target)
- phase4_test_results.pkl (auxiliary pass/fail target)

Outputs:
- checkpoints/execution_guided_reranker.pt
- cache/phase3_8_exec_pred.pkl  (metric key: execution_pred)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainConfig:
    # Kept for old pickle compatibility.
    data_path: str = "unused-but-required-for-unpickling"
    test_samples: int = 750
    max_length: int = 512
    model_name: str = "microsoft/unixcoder-base"
    projection_dim: int = 256
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 0.07
    weight_decay: float = 0.01
    output_dir: str = "./checkpoints"
    save_every: int = 2


@dataclass
class Candidate:
    code: str
    temperature: float
    compiles: bool = False
    passes_tests: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Sample:
    idx: int
    opt: str
    ghidra: str
    c_func: str
    c_test: str
    real_io_pairs: dict = field(default_factory=dict)
    real_iospec: str = ""
    candidates: List[Candidate] = field(default_factory=list)


class RerankerDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "target_exec": torch.tensor(row["target_exec"], dtype=torch.float32),
            "target_pass": torch.tensor(row["target_pass"], dtype=torch.float32),
            "sample_idx": row["sample_idx"],
            "cand_idx": row["cand_idx"],
        }


def collate_fn(batch: List[dict], tokenizer) -> dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        return_tensors="pt",
    )
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "target_exec": torch.stack([x["target_exec"] for x in batch]),
        "target_pass": torch.stack([x["target_pass"] for x in batch]),
        "sample_idx": [x["sample_idx"] for x in batch],
        "cand_idx": [x["cand_idx"] for x in batch],
    }


class ExecutionGuidedReranker(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.exec_head = nn.Linear(hidden_size, 1)   # predicts execution_guided score [0,1]
        self.pass_head = nn.Linear(hidden_size, 1)   # predicts pass probability

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            pooled = out.last_hidden_state[:, 0]
        else:
            pooled = out[0][:, 0]
        pooled = self.dropout(pooled)
        exec_logit = self.exec_head(pooled).squeeze(-1)
        pass_logit = self.pass_head(pooled).squeeze(-1)
        return exec_logit, pass_logit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and apply an execution-guided distillation reranker.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    common.add_argument("--num-samples", type=int, default=656)
    common.add_argument("--num-candidates", type=int, default=100)
    common.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    common.add_argument("--max-length", type=int, default=1024)
    common.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train_and_score", parents=[common])
    train.add_argument("--fit-samples", type=int, default=100)
    train.add_argument("--val-samples", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--alpha-pass", type=float, default=0.5, help="Weight for auxiliary pass/fail loss.")
    train.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints/execution_guided_reranker.pt"))
    train.add_argument("--score-cache-name", type=str, default="phase3_8_exec_pred.pkl")
    train.add_argument("--overwrite-score-cache", action="store_true")

    infer = sub.add_parser("infer_one", parents=[common])
    infer.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints/execution_guided_reranker.pt"))
    infer.add_argument("--prompt", type=str, required=True)
    infer.add_argument("--code", type=str, required=True)

    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def make_text(sample: Sample, candidate_code: str) -> str:
    return (
        "# Task context:\n"
        f"{sample.ghidra}\n\n"
        "# Candidate code:\n"
        f"{candidate_code}\n"
    )


def build_rows(
    samples: List[Sample],
    compile_map: dict,
    exec_map: dict,
    pass_map: dict,
) -> List[dict]:
    rows: List[dict] = []
    for sample in samples:
        for cand_idx, cand in enumerate(sample.candidates):
            key = (sample.idx, cand_idx)
            compiles = bool(compile_map.get(key, False))
            # Primary target: execution-guided score; non-compiling => 0
            target_exec = float(exec_map.get(key, {}).get("execution_guided", 0.0)) if compiles else 0.0
            # Auxiliary target: pass/fail
            target_pass = 1.0 if bool(pass_map.get(key, False)) else 0.0
            rows.append(
                {
                    "sample_idx": sample.idx,
                    "cand_idx": cand_idx,
                    "text": make_text(sample, cand.code),
                    "target_exec": target_exec,
                    "target_pass": target_pass,
                }
            )
    return rows


def split_rows_by_sample(rows: List[dict], fit_samples: int, val_samples: int) -> Tuple[List[dict], List[dict], List[dict]]:
    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []
    for row in rows:
        idx = int(row["sample_idx"])
        if idx < fit_samples:
            train_rows.append(row)
        elif idx < fit_samples + val_samples:
            val_rows.append(row)
        else:
            test_rows.append(row)
    return train_rows, val_rows, test_rows


def run_epoch(
    model: ExecutionGuidedReranker,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    alpha_pass: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    running = 0.0
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_exec = batch["target_exec"].to(device)
        target_pass = batch["target_pass"].to(device)

        pred_exec, pred_pass_logit = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = mse(torch.sigmoid(pred_exec), target_exec) + alpha_pass * bce(pred_pass_logit, target_pass)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        count += 1
    return running / max(count, 1)


@torch.inference_mode()
def evaluate_loss(
    model: ExecutionGuidedReranker,
    loader: DataLoader,
    device: torch.device,
    alpha_pass: float,
) -> float:
    model.eval()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    running = 0.0
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_exec = batch["target_exec"].to(device)
        target_pass = batch["target_pass"].to(device)
        pred_exec, pred_pass_logit = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = mse(torch.sigmoid(pred_exec), target_exec) + alpha_pass * bce(pred_pass_logit, target_pass)
        running += float(loss.item())
        count += 1
    return running / max(count, 1)


@torch.inference_mode()
def score_rows(
    model: ExecutionGuidedReranker,
    rows: List[dict],
    tokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[Tuple[int, int], dict]:
    ds = RerankerDataset(rows, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
    out: Dict[Tuple[int, int], dict] = {}
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pred_exec, pred_pass_logit = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_exec = torch.sigmoid(pred_exec).detach().cpu().tolist()
        pred_pass = torch.sigmoid(pred_pass_logit).detach().cpu().tolist()
        for sidx, cidx, exec_prob, pass_prob in zip(batch["sample_idx"], batch["cand_idx"], pred_exec, pred_pass):
            out[(int(sidx), int(cidx))] = {
                "execution_pred": float(exec_prob),
                "pass_pred": float(pass_prob),
            }
    return out


def train_and_score(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    candidates_file = args.cache_dir / f"candidates_{args.num_samples}_{args.num_candidates}.pkl"
    compile_file = args.cache_dir / "phase1_compilation.pkl"
    exec_file = args.cache_dir / "phase3_5_execution_guided.pkl"
    pass_file = args.cache_dir / "phase4_test_results.pkl"
    score_cache = args.cache_dir / args.score_cache_name

    samples: List[Sample] = load_pickle(candidates_file)
    compile_map = load_pickle(compile_file)
    exec_map = load_pickle(exec_file) if exec_file.exists() else {}
    pass_map = load_pickle(pass_file)

    rows = build_rows(samples, compile_map, exec_map, pass_map)
    train_rows, val_rows, test_rows = split_rows_by_sample(rows, args.fit_samples, args.val_samples)

    print(f"Rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} total={len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ExecutionGuidedReranker(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_ds = RerankerDataset(train_rows, tokenizer, args.max_length)
    val_ds = RerankerDataset(val_rows, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, optimizer, device, args.alpha_pass)
        va = evaluate_loss(model, val_loader, device, args.alpha_pass)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={tr:.6f} val_loss={va:.6f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "state_dict": best_state,
            "max_length": args.max_length,
            "fit_samples": args.fit_samples,
            "val_samples": args.val_samples,
        },
        args.checkpoint_path,
    )
    print(f"Saved checkpoint to {args.checkpoint_path}")

    existing = {}
    if score_cache.exists() and not args.overwrite_score_cache:
        existing = load_pickle(score_cache)

    predicted_all = score_rows(
        model=model,
        rows=rows,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    existing.update(predicted_all)
    save_pickle(score_cache, existing)
    print(f"Saved score cache to {score_cache} with {len(existing)} entries")

    # Quick held-out summary on sample-level top1 by execution_pred
    by_sample: Dict[int, List[Tuple[int, dict]]] = {}
    for row in test_rows:
        key = (int(row["sample_idx"]), int(row["cand_idx"]))
        score_obj = predicted_all.get(key)
        if score_obj is None:
            continue
        by_sample.setdefault(int(row["sample_idx"]), []).append((int(row["cand_idx"]), score_obj))

    correct = 0
    total = 0
    for sample_idx, cand_scores in by_sample.items():
        best_cidx, _ = max(cand_scores, key=lambda x: x[1]["execution_pred"])
        if bool(pass_map.get((sample_idx, best_cidx), False)):
            correct += 1
        total += 1
    if total > 0:
        print(f"Held-out top1 pass using execution_pred: {correct}/{total} = {100.0 * correct / total:.2f}%")


@torch.inference_mode()
def infer_one(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = int(ckpt.get("max_length", args.max_length))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ExecutionGuidedReranker(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    text = "# Task context:\n" + args.prompt + "\n\n# Candidate code:\n" + args.code + "\n"
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    pred_exec, pred_pass = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    out = {
        "execution_pred_probability": float(torch.sigmoid(pred_exec).item()),
        "pass_pred_probability": float(torch.sigmoid(pred_pass).item()),
    }
    print(json.dumps(out, indent=2))


def main() -> None:
    args = parse_args()
    if args.cmd == "train_and_score":
        train_and_score(args)
    elif args.cmd == "infer_one":
        infer_one(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
