#!/usr/bin/env python3
""" 
Train AutoML-based ADMET classifiers using ECFP fingerprints.

Dataset assumptions:
- One folder per ADMET task
- Files named:
    <PREFIX>_train.csv
    <PREFIX>_val.csv   (optional)
    <PREFIX>_test.csv

Outputs (written to repo root):
- <TASK>_metrics.json
- <TASK>_leaderboard.json
- <TASK>_config.json
- admet_summary.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, accuracy_score


# ============================================================================
# CONFIG
# ============================================================================

DATA_ROOT = Path("Datasets/Auto_ML_dataset")
OUTPUT_ROOT = Path(".")

SMILES_COL = "SMILES"
LABEL_COL = "bioclass"

FP_RADIUS = 2
FP_BITS = 2048
EVAL_METRIC = "roc_auc"

ADMET_TASKS = [
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


# ============================================================================
# FEATURIZATION (modern RDKit API)
# ============================================================================

_morgan_gen = GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS
)


def smiles_to_ecfp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.int8)

    fp = _morgan_gen.GetFingerprint(mol)
    return np.asarray(fp, dtype=np.int8)


def featurize_df(df: pd.DataFrame) -> pd.DataFrame:
    fps = np.vstack(df[SMILES_COL].apply(smiles_to_ecfp))
    fp_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(FP_BITS)])
    fp_df[LABEL_COL] = df[LABEL_COL].values
    return fp_df


# ============================================================================
# TRAINING PER TASK
# ============================================================================

def train_task(task: str) -> dict:
    print(f"\n=== {task} ===")

    task_dir = DATA_ROOT / task
    prefix = task.split("_")[0]

    train_path = task_dir / f"{prefix}_train.csv"
    val_path   = task_dir / f"{prefix}_val.csv"
    test_path  = task_dir / f"{prefix}_test.csv"

    print(f"Train path: {train_path}")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train or test file for {task}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    val_df   = pd.read_csv(val_path) if val_path.exists() else None

    train_feat = featurize_df(train_df)
    test_feat  = featurize_df(test_df)
    val_feat   = featurize_df(val_df) if val_df is not None else None

    # Combine train + val for AutoGluon (bagging handles validation internally)
    train_data = (
        pd.concat([train_feat, val_feat], ignore_index=True)
        if val_feat is not None
        else train_feat
    )

    predictor = TabularPredictor(
        label=LABEL_COL,
        eval_metric=EVAL_METRIC,
        path=f"autogluon_{task}"
    )

    predictor.fit(
        train_data=train_data,
        presets="best_quality",
        dynamic_stacking=False,
        num_stack_levels=1,
    )

    # --------------------
    # Test evaluation
    # --------------------
    X_test = test_feat.drop(columns=[LABEL_COL])
    y_true = test_feat[LABEL_COL].values

    y_prob = predictor.predict_proba(X_test)[1].values
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    print(f"Test AUC: {auc:.4f} | Test Acc: {acc:.4f}")

    # --------------------
    # Save outputs
    # --------------------
    metrics = {
        "task": task,
        "auc": float(auc),
        "accuracy": float(acc),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)) if val_df is not None else 0,
        "n_test": int(len(test_df)),
    }

    leaderboard = predictor.leaderboard(silent=True).to_dict(orient="records")

    config = {
        "fingerprint": {
            "type": "ECFP (Morgan)",
            "radius": FP_RADIUS,
            "n_bits": FP_BITS,
        },
        "automl": "AutoGluon Tabular",
        "eval_metric": EVAL_METRIC,
        "prefix_used": prefix,
        "has_validation": val_df is not None,
    }

    json.dump(metrics, open(OUTPUT_ROOT / f"{task}_metrics.json", "w"), indent=2)
    json.dump(leaderboard, open(OUTPUT_ROOT / f"{task}_leaderboard.json", "w"), indent=2)
    json.dump(config, open(OUTPUT_ROOT / f"{task}_config.json", "w"), indent=2)

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    results = []

    for task in ADMET_TASKS:
        results.append(train_task(task))

    summary = {
        "model_type": "AutoML (AutoGluon)",
        "fingerprint": "ECFP4 (radius=2, 2048 bits)",
        "tasks": results,
    }

    json.dump(summary, open("admet_summary.json", "w"), indent=2)

    print("\n=== ALL TASKS COMPLETE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
