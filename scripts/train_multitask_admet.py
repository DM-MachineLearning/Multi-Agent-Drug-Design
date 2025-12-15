#!/usr/bin/env python3
"""
Paper-grade multi-task ADMET classification using ECFP + RDKit descriptors.

- Hard parameter sharing (shared trunk)
- Masked loss for missing labels
- Task weighting + task dropout
- CPU only
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import roc_auc_score, accuracy_score


# =========================
# CONFIG
# =========================

SEED = 42
DATA_ROOT = Path("Datasets/Auto_ML_dataset")
OUTPUT_DIR = Path("outputs_multitask")
OUTPUT_DIR.mkdir(exist_ok=True)

SMILES_COL = "SMILES"
LABEL_COL = "bioclass"

TASKS = [
    "BBBP",
    "Caco2_permeability",
    "CYP1A2_inhibition",
    "CYP2C9_inhibition",
    "CYP2C19_inhibition",
    "CYP2D6_inhibition",
    "CYP3A4_inhibition",
    "HERG_inhibition",
    "HLM_stability",
    "P-gp_substrate",
    "RLM_stability",
]

FP_RADIUS = 2
FP_BITS = 2048
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors",
    "NumHAcceptors", "NumRotatableBonds", "RingCount"
]

BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
TASK_DROPOUT_P = 0.1
DEVICE = torch.device("cpu")


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


# =========================
# FEATURIZATION
# =========================

_morgan = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

def smiles_to_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(FP_BITS + len(DESCRIPTOR_NAMES), dtype=np.float32)

    fp = np.asarray(_morgan.GetFingerprint(mol), dtype=np.float32)

    descs = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
    ], dtype=np.float32)

    return np.concatenate([fp, descs])


def featurize_dataframe(df: pd.DataFrame):
    X = np.vstack(df[SMILES_COL].apply(smiles_to_features))
    y = df[LABEL_COL].values.astype(np.float32)
    return X, y


# =========================
# DATASET (MASKED)
# =========================

class MultiTaskDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = {}
        self.mask = {}

        for task, arr in y.items():
            m = ~np.isnan(arr)
            self.mask[task] = torch.tensor(m.astype(np.float32))
            self.y[task] = torch.tensor(
                np.nan_to_num(arr, nan=0.0), dtype=torch.float32
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        labels = {t: self.y[t][idx] for t in TASKS}
        masks = {t: self.mask[t][idx] for t in TASKS}
        return self.X[idx], labels, masks


# =========================
# MODEL
# =========================

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, tasks: List[str]):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleDict({
            task: nn.Linear(256, 1) for task in tasks
        })

    def forward(self, x):
        h = self.trunk(x)
        return {t: self.heads[t](h).squeeze(1) for t in TASKS}


# =========================
# LOAD DATA (UNION LOGIC)
# =========================

def load_union_dataset(split: str):
    all_X = []
    all_y = {t: [] for t in TASKS}

    for task in TASKS:
        task_dir = DATA_ROOT / task
        prefix = task.split("_")[0]
        path = task_dir / f"{prefix}_{split}.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)
        X_task, y_task = featurize_dataframe(df)
        n = len(X_task)

        all_X.append(X_task)
        for t in TASKS:
            if t == task:
                all_y[t].append(y_task)
            else:
                all_y[t].append(np.full(n, np.nan))

        print(f"{task}: {split}={n}")

    X = np.vstack(all_X)
    y = {t: np.concatenate(all_y[t]) for t in TASKS}
    return X, y


# =========================
# TRAINING
# =========================

def main():
    print("Loading datasets...")
    X_train, y_train = load_union_dataset("train")
    X_val, y_val = load_union_dataset("val")
    X_test, y_test = load_union_dataset("test")

    train_loader = DataLoader(
        MultiTaskDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        MultiTaskDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )

    model = MultiTaskMLP(X_train.shape[1], TASKS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    task_weights = {
        t: 1.0 / math.sqrt(np.sum(~np.isnan(y_train[t])))
        for t in TASKS
    }

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, labels, masks in train_loader:
            X = X.to(DEVICE)
            preds = model(X)

            loss = 0.0
            for task in TASKS:
                if random.random() < TASK_DROPOUT_P:
                    continue

                m = masks[task].to(DEVICE)
                if m.sum() == 0:
                    continue

                y = labels[task].to(DEVICE)
                p = preds[task]

                bce = nn.functional.binary_cross_entropy_with_logits(
                    p, y, reduction="none"
                )
                loss += task_weights[task] * (bce * m).sum() / m.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f}")

    # =========================
    # EVALUATION
    # =========================

    model.eval()
    metrics = {}

    with torch.no_grad():
        for task in TASKS:
            y_true, y_score = [], []

            for X, labels, masks in test_loader:
                m = masks[task]
                if m.sum() == 0:
                    continue

                logits = model(X)[task]
                probs = torch.sigmoid(logits)

                y_true.extend(labels[task][m.bool()].numpy())
                y_score.extend(probs[m.bool()].numpy())

            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_score)
                acc = accuracy_score(y_true, np.array(y_score) >= 0.5)
            else:
                auc, acc = float("nan"), float("nan")

            metrics[task] = {"roc_auc": auc, "accuracy": acc}

    json.dump(metrics, open(OUTPUT_DIR / "test_metrics.json", "w"), indent=2)
    torch.save(model.state_dict(), OUTPUT_DIR / "multitask_model.pt")

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
