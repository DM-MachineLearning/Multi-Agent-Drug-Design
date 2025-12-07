#!/usr/bin/env python3
"""
Train an MLP regressor to predict pChEMBL Value from SMILES using:
- ECFP fingerprints (configurable radius & n_bits)
- Molecular weight (normalized)
- PyTorch MLP architecture (with BatchNorm & 3 hidden layers)
- Optional Optuna hyperparameter optimization

Targets are standardized (mean 0, std 1) during training, but metrics
are computed in the original pChEMBL space.

Outputs:
- Trained model checkpoint (.pt)
- Metrics JSON (R2, RMSE, MAE, Pearson r)
- (If Optuna used) hyperparameters + training artifacts

Example usage:
    python scripts/train_pchembl_regressor.py --data "Datasets/AKT1 CHEMBL_regression.xlsx"
    python scripts/train_pchembl_regressor.py --data data.csv --use-optuna --optuna-trials 50 --radius 3 --n-bits 4096
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import optuna
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_GRAD_NORM = 5.0  # for gradient clipping


# ============================================================================
# RDKit FEATURIZATION
# ============================================================================

def smiles_to_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    mol = smiles_to_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float32)


def smiles_to_molecular_weight(smiles: str) -> float:
    mol = smiles_to_mol(smiles)
    return float(Descriptors.MolWt(mol))


def validate_smiles(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False


# ============================================================================
# MODEL
# ============================================================================

class PChEMBLRegressor(nn.Module):
    """MLP for regression on ECFP fingerprints + molecular weight.
    
    Architecture:
        Linear(input_dim -> hidden)
        BatchNorm
        ReLU
        (Dropout)
        Linear(hidden -> hidden)
        BatchNorm
        ReLU
        (Dropout)
        Linear(hidden -> hidden)
        ReLU
        Linear(hidden -> 1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Second hidden layer
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Third hidden layer (no BN to keep it simple)
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Model predicts *normalized* pChEMBL (mean 0, std 1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_batch(self, x):
        return self(x).cpu().numpy()

    @torch.no_grad()
    def predict(self, x):
        return float(self(x.unsqueeze(0)).item())

    def save_state(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_state(cls, path, map_location="cpu", input_dim=2049, hidden_dim=512, dropout=0.0):
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location=map_location))
        return model


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_dataset(data_path: str, target_col: Optional[str] = None):
    data_path = Path(data_path)

    if data_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(data_path)
    elif data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    df.columns = df.columns.str.lower()

    # auto-detect target column
    if target_col:
        target_col = target_col.lower()
    else:
        candidates = [c for c in df.columns if "pchembl" in c]
        if not candidates:
            raise ValueError("Could not find a pChEMBL column.")
        target_col = candidates[0]

    return df, target_col


def prepare_data(df: pd.DataFrame, target_col: str, radius: int = 2, n_bits: int = 2048, seed: int = 42):
    smiles_col = None
    for col in ["smiles", "smile"]:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError("No SMILES column found.")

    valid_rows = []
    invalid_rows = []

    print("Processing molecules...")

    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        target = row[target_col]

        if pd.isna(target):
            continue

        if not validate_smiles(smiles):
            invalid_rows.append({"idx": idx, "smiles": smiles})
            continue

        try:
            fp = smiles_to_ecfp(smiles, radius, n_bits)
            mw = smiles_to_molecular_weight(smiles)
            valid_rows.append({"fp": fp, "mw": mw, "target": float(target), "smiles": smiles})
        except Exception as e:
            invalid_rows.append({"idx": idx, "smiles": smiles, "error": str(e)})

    if invalid_rows:
        Path("data").mkdir(exist_ok=True)
        pd.DataFrame(invalid_rows).to_csv("data/invalid_smiles_pchembl.csv", index=False)
        print(f"Invalid SMILES saved to data/invalid_smiles_pchembl.csv")

    print(f"Valid rows: {len(valid_rows)}")

    train_val, test = train_test_split(valid_rows, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed)

    print(f"Split → train: {len(train)}, val: {len(val)}, test: {len(test)}")

    return train, val, test


def normalize_mw(rows, mw_mean: float, mw_std: float):
    for r in rows:
        r["mw"] = (r["mw"] - mw_mean) / mw_std


# ============================================================================
# TRAINING + EVALUATION
# ============================================================================

def evaluate_model(model, rows, batch_size, device, y_mean: float, y_std: float):
    """
    Evaluate model on given rows. The model outputs normalized predictions,
    which are then denormalized back to original pChEMBL space for metrics.
    """
    preds, trues = [], []

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        fps = [r["fp"] for r in batch]
        mws = [r["mw"] for r in batch]
        ys = [r["target"] for r in batch]

        fp_tensor = torch.stack(fps)
        mw_tensor = torch.tensor(mws, dtype=torch.float32).unsqueeze(1)

        feats = torch.cat([fp_tensor, mw_tensor], dim=1).to(device)

        with torch.no_grad():
            batch_preds_norm = model.predict_batch(feats)  # normalized
        batch_preds = batch_preds_norm * y_std + y_mean    # back to original scale

        preds.extend(batch_preds)
        trues.extend(ys)

    preds = np.array(preds)
    trues = np.array(trues)

    r2 = r2_score(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    pearson_r, pearson_p = pearsonr(trues, preds)

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
    }


def train_one_model(
    train_rows,
    val_rows,
    input_dim,
    hidden_dim,
    lr,
    dropout,
    epochs,
    batch_size,
    weight_decay,
    device,
    y_mean: float,
    y_std: float,
    max_grad_norm: Optional[float] = None,
):

    model = PChEMBLRegressor(input_dim, hidden_dim, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_rows)

        total_loss = 0.0
        for i in range(0, len(train_rows), batch_size):
            batch = train_rows[i:i+batch_size]
            fps = [r["fp"] for r in batch]
            mws = [r["mw"] for r in batch]
            ys = [r["target"] for r in batch]

            fp_tensor = torch.stack(fps)
            mw_tensor = torch.tensor(mws, dtype=torch.float32).unsqueeze(1)
            feats = torch.cat([fp_tensor, mw_tensor], 1).to(device)
            y_true = torch.tensor(ys, dtype=torch.float32, device=device)

            # Normalize targets
            y_true_norm = (y_true - y_mean) / y_std

            opt.zero_grad()
            y_pred_norm = model(feats)
            loss = loss_fn(y_pred_norm, y_true_norm)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            opt.step()

            total_loss += loss.item()

        val_m = evaluate_model(model, val_rows, batch_size, device, y_mean, y_std)
        val_loss = val_m["rmse"] ** 2  # MSE in original space

        print(
            f"Epoch {epoch}: train_loss={total_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_r2={val_m['r2']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = PChEMBLRegressor(input_dim, hidden_dim, dropout).to(device)
            best_model.load_state_dict(model.state_dict())

    return best_model, best_val_loss


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial, train_rows, val_rows, input_dim, device, y_mean: float, y_std: float):
    hidden_dim = trial.suggest_int("hidden_dim", 256, 2048, step=256)
    lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

    # Fewer epochs for speed during search; then retrain best with more epochs
    model, val_loss = train_one_model(
        train_rows,
        val_rows,
        input_dim,
        hidden_dim,
        lr,
        dropout,
        epochs=20,
        batch_size=64,
        weight_decay=weight_decay,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        max_grad_norm=MAX_GRAD_NORM,
    )

    return val_loss


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=2, help="ECFP radius (e.g., 2 or 3)")
    parser.add_argument("--n-bits", type=int, default=2048, help="ECFP bit length (e.g., 2048 or 4096)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df, target_col = load_dataset(args.data)
    train_rows, val_rows, test_rows = prepare_data(df, target_col, radius=args.radius, n_bits=args.n_bits, seed=args.seed)

    # Normalize molecular weight based on *train* statistics
    train_mws = np.array([r["mw"] for r in train_rows], dtype=np.float32)
    mw_mean = float(train_mws.mean())
    mw_std = float(train_mws.std() + 1e-8)

    print(f"MW normalization: mean={mw_mean:.4f}, std={mw_std:.4f}")

    normalize_mw(train_rows, mw_mean, mw_std)
    normalize_mw(val_rows, mw_mean, mw_std)
    normalize_mw(test_rows, mw_mean, mw_std)

    # Target normalization stats (used during training; metrics in original space)
    train_targets = np.array([r["target"] for r in train_rows], dtype=np.float32)
    y_mean = float(train_targets.mean())
    y_std = float(train_targets.std() + 1e-8)

    print(f"Target normalization: mean={y_mean:.4f}, std={y_std:.4f}")

    input_dim = args.n_bits + 1  # ECFP + normalized MW

    if args.use_optuna:
        print("\n=== Running Optuna Search ===")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: objective(t, train_rows, val_rows, input_dim, device, y_mean, y_std),
            n_trials=args.optuna_trials,
        )

        best = study.best_params
        print("\nBest hyperparameters:", best)

        # Retrain best model on train + val with more epochs
        best_model, _ = train_one_model(
            train_rows + val_rows,
            val_rows,  # still monitor on original val set
            input_dim,
            best["hidden_dim"],
            best["lr"],
            best["dropout"],
            args.epochs,
            args.batch_size,
            best["weight_decay"],
            device,
            y_mean,
            y_std,
            max_grad_norm=MAX_GRAD_NORM,
        )

        Path("models").mkdir(exist_ok=True)
        best_model.save_state("models/pchembl_regressor_optuna.pt")
        json.dump(best, open("hparams_pchembl_optuna.json", "w"), indent=2)

        metrics = evaluate_model(best_model, test_rows, args.batch_size, device, y_mean, y_std)
        json.dump(metrics, open("pchembl_test_metrics_optuna.json", "w"), indent=2)

        print("\nTest metrics:", metrics)
        return

    # No optuna → simple training
    print("\nTraining without Optuna...")
    model, _ = train_one_model(
        train_rows,
        val_rows,
        input_dim,
        hidden_dim=512,
        lr=1e-3,
        dropout=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=1e-5,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        max_grad_norm=MAX_GRAD_NORM,
    )

    Path("models").mkdir(exist_ok=True)
    model.save_state("models/pchembl_regressor.pt")

    metrics = evaluate_model(model, test_rows, args.batch_size, device, y_mean, y_std)
    json.dump(metrics, open("pchembl_test_metrics.json", "w"), indent=2)

    print("\nTest metrics:", metrics)


if __name__ == "__main__":
    main()
