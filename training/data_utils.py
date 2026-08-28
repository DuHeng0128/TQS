"""
Data loading utilities for CoderForge training.

Dataset: togethercomputer/CoderForge-Preview, trajectories-tokenized_qwencoder subset
- input_ids: int32 numpy array, already tokenized
- labels:    int64 numpy array, -100 for tokens outside loss mask
- reward:    float, 1.0 = success

No pre-packing: each trajectory is truncated/padded to max_seq_len at __getitem__ time.
This keeps memory usage at O(num_trajectories × avg_seq_len × 4 bytes) rather than
O(num_packed_sequences × max_seq_len × 8 bytes).
"""

import glob
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CoderForgeTokenizedDataset(Dataset):
    """
    Loads parquet files from a local CoderForge trajectories-tokenized_qwencoder directory,
    filters to reward == 1.0, and stores raw numpy arrays (not tensors) to minimize memory.

    __getitem__ returns a dict of tensors, padded/truncated to max_seq_len on the fly.
    """

    def __init__(
        self,
        local_dir: Optional[str] = None,
        hf_repo: str = "togethercomputer/CoderForge-Preview",
        hf_config: str = "trajectories-tokenized_qwencoder",
        max_seq_len: int = 131072,
    ):
        self.max_seq_len = max_seq_len
        # Store as numpy arrays (int32/int64) — much cheaper than tensors
        self.input_ids: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []

        if local_dir and os.path.isdir(local_dir):
            files = sorted(glob.glob(os.path.join(local_dir, "*.parquet")))
            logger.info(f"Loading {len(files)} parquet files from {local_dir}")
            for fpath in files:
                df = pd.read_parquet(fpath, columns=["input_ids", "labels", "reward"])
                df = df[df["reward"] == 1.0]
                for _, row in df.iterrows():
                    ids = np.asarray(row["input_ids"], dtype=np.int32)
                    lbls = np.asarray(row["labels"], dtype=np.int32)  # int32 saves vs int64
                    if len(ids) > max_seq_len:
                        ids = ids[:max_seq_len]
                        lbls = lbls[:max_seq_len]
                    self.input_ids.append(ids)
                    self.labels.append(lbls)
        else:
            from datasets import load_dataset
            logger.info(f"Loading from HuggingFace: {hf_repo} / {hf_config}")
            ds = load_dataset(hf_repo, name=hf_config, split="train")
            ds = ds.filter(lambda x: x["reward"] == 1.0)
            for row in ds:
                ids = np.asarray(row["input_ids"], dtype=np.int32)
                lbls = np.asarray(row["labels"], dtype=np.int32)
                if len(ids) > max_seq_len:
                    ids = ids[:max_seq_len]
                    lbls = lbls[:max_seq_len]
                self.input_ids.append(ids)
                self.labels.append(lbls)

        logger.info(f"Loaded {len(self.input_ids)} trajectories with reward=1.0")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx].astype(np.int64)
        lbls = self.labels[idx].astype(np.int64)
        seq_len = len(ids)
        pad_len = self.max_seq_len - seq_len

        if pad_len > 0:
            ids = np.concatenate([ids, np.zeros(pad_len, dtype=np.int64)])
            lbls = np.concatenate([lbls, np.full(pad_len, -100, dtype=np.int64)])

        attention_mask = np.ones(self.max_seq_len, dtype=np.int64)
        if pad_len > 0:
            attention_mask[seq_len:] = 0

        position_ids = np.arange(self.max_seq_len, dtype=np.int64)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbls, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }


# Keep PackedCoderForgeDataset as an alias so train_14b.py import doesn't break.
# It simply wraps CoderForgeTokenizedDataset unchanged.
class PackedCoderForgeDataset(CoderForgeTokenizedDataset):
    """Thin alias — packing is no longer done at init; each item is padded on the fly."""

    def __init__(self, raw_dataset: CoderForgeTokenizedDataset, **kwargs):
        # Re-use the already-loaded data from raw_dataset
        self.max_seq_len = raw_dataset.max_seq_len
        self.input_ids = raw_dataset.input_ids
        self.labels = raw_dataset.labels
        logger.info(f"PackedCoderForgeDataset: {len(self)} sequences (pad-on-the-fly mode)")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_valid_tokens(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())
