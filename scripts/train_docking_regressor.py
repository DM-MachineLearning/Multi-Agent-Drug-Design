"""Training script for the docking regressor with full CLI, data loading, and evaluation."""
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib

try:
    import optuna
except ImportError:  # Optuna is optional; only required when --use-optuna is set
    optuna = None

# Use non-interactive backend for plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from madm.properties import DockingRegressor
from madm.data.featurization import smiles_to_ecfp
from rdkit import Chem


class DockingDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        fp = row["fp"]
        y = torch.tensor(row["target"], dtype=torch.float32)
        return fp, y


def validate_smiles(smiles: str) -> bool:
    """Check if a string is a valid SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def load_dataset(data_path: str, target_col: str = None) -> Tuple[pd.DataFrame, str]:
    """Load dataset from Excel or CSV file.
    
    Returns:
        DataFrame and detected target column name
    """
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.xlsx':
        df = pd.read_excel(data_path)
    elif data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Detect target column
    if target_col:
        target_col = target_col.lower()
    else:
        # Try to find dockscore or dock column
        if 'dockscore' in df.columns:
            target_col = 'dockscore'
        elif 'dock' in df.columns:
            target_col = 'dock'
        else:
            raise ValueError("Could not find 'dockscore' or 'dock' column in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    return df, target_col


def prepare_data(df: pd.DataFrame, target_col: str, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Prepare and validate data, return train/val/test splits.
    
    Returns:
        train_rows, val_rows, test_rows (each is list of dicts with 'fp' and 'target')
    """
    # Find SMILES column
    smiles_col = None
    for col in ['smiles', 'smile']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        # Check if 'ligand' column exists and contains valid SMILES
        if 'ligand' in df.columns:
            # Validate a sample
            sample_ligands = df['ligand'].dropna().head(10)
            valid_count = sum(validate_smiles(str(s)) for s in sample_ligands)
            
            if valid_count == 0:
                # No valid SMILES found - create invalid_ligands.csv
                invalid_dir = Path("data")
                invalid_dir.mkdir(exist_ok=True)
                invalid_path = invalid_dir / "invalid_ligands.csv"
                
                df_invalid = df[['ligand', target_col]].copy()
                df_invalid.to_csv(invalid_path, index=False)
                
                print(f"\nERROR: 'ligand' column does not contain valid SMILES strings.")
                print(f"Invalid ligands saved to: {invalid_path}")
                print("\nPlease provide a SMILES column or mapping file.")
                print("Expected format: column named 'smiles' or 'SMILES' with valid SMILES strings.")
                raise ValueError("No valid SMILES column found")
            else:
                smiles_col = 'ligand'
        else:
            raise ValueError("No 'smiles' or 'ligand' column found in dataset")
    
    # Filter rows with valid SMILES and non-null targets
    valid_rows = []
    invalid_rows = []
    
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        target = row[target_col]
        
        if pd.isna(target):
            continue
        
        if validate_smiles(smiles):
            try:
                fp = smiles_to_ecfp(smiles)
                valid_rows.append({
                    'fp': fp,
                    'target': float(target),
                    'smiles': smiles
                })
            except Exception as e:
                invalid_rows.append({'idx': idx, 'smiles': smiles, 'error': str(e)})
        else:
            invalid_rows.append({'idx': idx, 'smiles': smiles, 'error': 'Invalid SMILES'})
    
    if invalid_rows:
        invalid_dir = Path("data")
        invalid_dir.mkdir(exist_ok=True)
        invalid_path = invalid_dir / "invalid_ligands.csv"
        pd.DataFrame(invalid_rows).to_csv(invalid_path, index=False)
        print(f"\nWarning: {len(invalid_rows)} rows with invalid SMILES saved to {invalid_path}")
    
    if not valid_rows:
        raise ValueError("No valid rows found after SMILES validation")
    
    print(f"Loaded {len(valid_rows)} valid rows")
    
    # Split: 80% train+val, 20% test
    train_val_rows, test_rows = train_test_split(
        valid_rows, test_size=0.2, random_state=seed
    )
    
    # Split train+val: 80% train, 20% val
    train_rows, val_rows = train_test_split(
        train_val_rows, test_size=0.2, random_state=seed
    )
    
    print(f"Split: {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test")
    
    return train_rows, val_rows, test_rows


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p)
    }


def train_model(
    train_rows: List[Dict],
    val_rows: List[Dict],
    input_dim: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    epochs: int,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    patience: int = 5,
    model_dir: str = "models",
    trial: Optional["optuna.trial.Trial"] = None,
) -> Tuple[DockingRegressor, Dict]:
    """Train the docking regressor model."""
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = DockingDataset(train_rows)
    val_dataset = DockingDataset(val_rows)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DockingRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    metrics_history = []
    
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for fps, y_true in train_loader:
            fps = fps.to(device=device, dtype=torch.float32)
            y_true = y_true.to(device=device, dtype=torch.float32)
            y_pred = model(fps)
            loss = loss_fn(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(fps)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fps, y_true in val_loader:
                fps = fps.to(device=device, dtype=torch.float32)
                y_true = y_true.to(device=device, dtype=torch.float32)
                y_pred = model(fps)
                loss = loss_fn(y_pred, y_true)
                val_loss += float(loss.item()) * len(fps)
        val_loss /= len(val_dataset)

        # Report to Optuna and support pruning if a trial is provided
        if trial is not None and optuna is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                print(
                    f"  -> Trial {trial.number} pruned at epoch {epoch} "
                    f"(val_loss={val_loss:.4f})"
                )
                raise optuna.exceptions.TrialPruned()

        scheduler.step(val_loss)
        
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            model_path = model_dir_path / "docking_regressor.pt"
            model.save_state(str(model_path))
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break
    
    # Load best model
    model_path = model_dir_path / "docking_regressor.pt"
    model = DockingRegressor.load_state(
        str(model_path),
        map_location=str(device),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    
    print(f"\nBest model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    
    return model, {'history': metrics_history, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss}


def evaluate_model(
    model: DockingRegressor,
    test_rows: List[Dict],
    batch_size: int = 64,
    return_preds: bool = False,
) -> Dict[str, float]:
    """Evaluate model on test set.

    If return_preds is True, also returns y_true and y_pred arrays.
    """
    test_dataset = DockingDataset(test_rows)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Infer device from model parameters
    device = next(model.parameters()).device

    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for fps, y_true in test_loader:
            fps = fps.to(device=device, dtype=torch.float32)
            y_true = y_true.to(device=device, dtype=torch.float32)
            y_pred = model.predict_batch(fps)
            all_preds.extend(y_pred)
            all_targets.extend(y_true.cpu().numpy())
    
    all_preds = np.array(all_preds, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    
    metrics = compute_metrics(all_targets, all_preds)
    if return_preds:
        return metrics, all_targets, all_preds
    return metrics


def run_baselines(train_rows: List[Dict], val_rows: List[Dict], test_rows: List[Dict]) -> Dict:
    """Run sklearn baseline models."""
    print("\nRunning baseline models...")
    
    # Prepare data
    def rows_to_arrays(rows):
        fps = torch.stack([r['fp'] for r in rows])
        targets = np.array([r['target'] for r in rows])
        return fps.numpy(), targets
    
    X_train, y_train = rows_to_arrays(train_rows)
    X_val, y_val = rows_to_arrays(val_rows)
    X_test, y_test = rows_to_arrays(test_rows)
    
    # Combine train and val for baseline training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    results = {}
    
    # Ridge Regression
    print("  Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_full, y_train_full)
    y_pred_ridge = ridge.predict(X_test)
    results['ridge'] = compute_metrics(y_test, y_pred_ridge)
    
    # Random Forest
    print("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_full, y_train_full)
    y_pred_rf = rf.predict(X_test)
    results['random_forest'] = compute_metrics(y_test, y_pred_rf)
    
    return results


def hyperparameter_search(
    train_rows: List[Dict],
    val_rows: List[Dict],
    input_dim: int,
    max_trials: int = 10,
) -> Dict:
    """Randomized grid search for hyperparameters.

    Selection metric: highest validation R², with lowest validation MSE as tie-breaker.
    """
    print("\nRunning hyperparameter search...")

    # Full grid as requested
    grid_hidden_dim = [256, 512, 768]
    grid_lr = [1e-2, 1e-3, 5e-4, 1e-4]
    grid_dropout = [0.0, 0.1, 0.2]
    grid_weight_decay = [0.0, 1e-5, 1e-4]

    param_grid = []
    for h in grid_hidden_dim:
        for lr in grid_lr:
            for d in grid_dropout:
                for wd in grid_weight_decay:
                    param_grid.append(
                        {
                            "hidden_dim": h,
                            "lr": lr,
                            "dropout": d,
                            "weight_decay": wd,
                        }
                    )

    # Randomly sample up to max_trials combinations for runtime control
    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(param_grid))
    rng.shuffle(indices)
    indices = indices[: max_trials]

    best_params = None
    best_r2 = -np.inf
    best_mse = np.inf
    results = []
    temp_dir = Path("models/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for trial_idx, grid_idx in enumerate(indices, start=1):
            params = param_grid[grid_idx]
            hidden_dim = params["hidden_dim"]
            lr = params["lr"]
            dropout = params["dropout"]
            weight_decay = params["weight_decay"]

            print(
                f"\nTrial {trial_idx}/{len(indices)}: "
                f"hidden_dim={hidden_dim}, lr={lr}, dropout={dropout}, weight_decay={weight_decay}"
            )

            # Quick training (fewer epochs for search)
            model, _ = train_model(
                train_rows,
                val_rows,
                input_dim,
                hidden_dim,
                batch_size=64,
                lr=lr,
                epochs=15,
                weight_decay=weight_decay,
                dropout=dropout,
                patience=5,
                model_dir=str(temp_dir),
            )

            val_metrics = evaluate_model(model, val_rows, batch_size=64)
            r2 = val_metrics["r2"]
            mse = val_metrics["rmse"] ** 2

            results.append(
                {
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "val_r2": r2,
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                }
            )

            # Higher R² is better; break ties with lower MSE
            if (r2 > best_r2) or (np.isclose(r2, best_r2) and mse < best_mse):
                best_r2 = r2
                best_mse = mse
                best_params = params

        if best_params is not None:
            print(
                f"\nBest hyperparameters: {best_params} "
                f"(val_r2={best_r2:.4f}, val_mse={best_mse:.4f})"
            )
        else:
            print("\nNo successful hyperparameter trials completed.")
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    return {"best_params": best_params, "all_results": results}


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate docking score regressor")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset (.xlsx or .csv)')
    parser.add_argument('--out', type=str, default='models/docking_regressor.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only mode')
    parser.add_argument('--model-path', type=str, help='Path to model for evaluation')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline models')
    parser.add_argument('--no-hyperopt', action='store_true', help='Skip manual hyperparameter search (v1)')
    parser.add_argument('--use-optuna', action='store_true', help='Use Optuna-based hyperparameter optimization instead of manual search')
    parser.add_argument('--optuna-trials', type=int, default=50, help='Number of Optuna trials to run')
    parser.add_argument('--optuna-storage', type=str, default='', help='Optuna storage URL (e.g. "sqlite:///optuna_docking.db"); empty for in-memory')
    
    args = parser.parse_args()

    # Avoid accidentally overwriting the original v1 model when using Optuna
    default_out_path = 'models/docking_regressor.pt'
    if args.use_optuna and args.out == default_out_path:
        args.out = 'models/docking_regressor_optuna.pt'
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.data}...")
    df, target_col = load_dataset(args.data)
    
    if args.eval_only:
        # Evaluation only mode
        if not args.model_path:
            raise ValueError("--model-path required for --eval-only mode")
        
        print("Evaluation only mode")
        train_rows, val_rows, test_rows = prepare_data(df, target_col, seed=args.seed)
        
        # Detect input_dim from first fingerprint
        input_dim = len(train_rows[0]['fp'])
        
        # Load model
        model = DockingRegressor.load_state(
            args.model_path, map_location='cpu',
            input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
        )
        
        # Evaluate
        test_metrics = evaluate_model(model, test_rows, batch_size=args.batch_size)
        
        print("\nTest Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save metrics
        metrics_path = Path("docking_test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
        
    else:
        # Training mode
        print("Training mode")
        train_rows, val_rows, test_rows = prepare_data(df, target_col, seed=args.seed)

        # Detect input_dim from first fingerprint
        input_dim = len(train_rows[0]['fp'])
        print(f"Detected input_dim: {input_dim}")

        # Directory setup for reports
        reports_dir = Path("reports") / "docking"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if args.use_optuna:
            # --- Optuna-based hyperparameter optimization path (v2) ---
            if optuna is None:
                raise SystemExit(
                    "Optuna is not installed but --use-optuna was specified.\n"
                    "Please install Optuna with: pip install optuna"
                )

            print("\nUsing Optuna for hyperparameter optimization")

            # Temporary directory for trial models
            temp_dir = Path("models") / "temp_optuna"
            temp_dir.mkdir(parents=True, exist_ok=True)

            def objective(trial: "optuna.trial.Trial") -> float:
                # Define search space
                hidden_dim = trial.suggest_int("hidden_dim", 256, 1024, step=256)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                dropout = trial.suggest_float("dropout", 0.0, 0.3)
                weight_decay = trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                )

                print(
                    f"\n[Optuna] Trial {trial.number}: "
                    f"hidden_dim={hidden_dim}, lr={lr:.5f}, "
                    f"dropout={dropout:.3f}, weight_decay={weight_decay:.6f}"
                )

                # Limit epochs per trial to keep search efficient
                trial_epochs = min(args.epochs, 50)

                try:
                    model, train_info = train_model(
                        train_rows,
                        val_rows,
                        input_dim,
                        hidden_dim,
                        batch_size=args.batch_size,
                        lr=lr,
                        epochs=trial_epochs,
                        weight_decay=weight_decay,
                        dropout=dropout,
                        patience=5,
                        model_dir=str(temp_dir),
                        trial=trial,
                    )
                except optuna.exceptions.TrialPruned:
                    # Let Optuna handle the pruning logic
                    raise

                # Use the best validation loss observed during this trial
                best_val_loss = float(train_info.get("best_val_loss", np.inf))

                # Optional: also log validation R2 for debugging
                val_metrics = evaluate_model(
                    model, val_rows, batch_size=args.batch_size
                )
                print(
                    f"[Optuna] Trial {trial.number} finished: "
                    f"best_val_loss={best_val_loss:.4f}, "
                    f"val_r2={val_metrics['r2']:.4f}"
                )

                return best_val_loss

            # Create Optuna study with TPE sampler and pruning
            sampler = optuna.samplers.TPESampler(seed=args.seed)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

            if args.optuna_storage:
                study = optuna.create_study(
                    study_name="docking_regressor_optuna",
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                    storage=args.optuna_storage,
                    load_if_exists=True,
                )
            else:
                study = optuna.create_study(
                    study_name="docking_regressor_optuna",
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                )

            print(
                f"\nStarting Optuna optimization with {args.optuna_trials} trials "
                f"(storage={'in-memory' if not args.optuna_storage else args.optuna_storage})"
            )

            try:
                study.optimize(objective, n_trials=args.optuna_trials)
            finally:
                # Cleanup temporary models
                if temp_dir.exists():
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)

            print("\nOptuna optimization finished.")
            print(f"  Best objective (val_loss): {study.best_value:.6f}")
            print(f"  Best hyperparameters: {study.best_params}")

            # Save best parameters and all trials
            hparams_optuna = {
                "input_dim": input_dim,
                "best_params": study.best_params,
                "best_value": float(study.best_value),
                "direction": study.direction.name,
                "n_trials": len(study.trials),
                "sampler": "TPESampler",
                "pruner": "MedianPruner(n_warmup_steps=5)",
                "seed": args.seed,
            }
            hparams_optuna_path = Path("hparams_docking_optuna.json")
            with open(hparams_optuna_path, "w") as f:
                json.dump(hparams_optuna, f, indent=2)
            print(f"Optuna best hyperparameters saved to {hparams_optuna_path}")

            trials_summary = []
            for t in study.trials:
                trials_summary.append(
                    {
                        "number": t.number,
                        "state": str(t.state),
                        "value": float(t.value) if t.value is not None else None,
                        "params": t.params,
                    }
                )
            trials_path = Path("optuna_trials_docking.json")
            with open(trials_path, "w") as f:
                json.dump(trials_summary, f, indent=2)
            print(f"Optuna trials summary saved to {trials_path}")

            # Retrain on combined train+val with best hyperparameters
            best_params = study.best_params
            best_hidden_dim = int(best_params["hidden_dim"])
            best_lr = float(best_params["lr"])
            best_dropout = float(best_params["dropout"])
            best_weight_decay = float(best_params["weight_decay"])

            train_full_rows = train_rows + val_rows
            print(
                f"\nRetraining on combined train+val set (n={len(train_full_rows)}) "
                "with Optuna-selected hyperparameters..."
            )
            model, train_metrics = train_model(
                train_full_rows,
                val_rows,  # keep a small validation set for early stopping
                input_dim,
                best_hidden_dim,
                batch_size=args.batch_size,
                lr=best_lr,
                epochs=args.epochs,
                weight_decay=best_weight_decay,
                dropout=best_dropout,
                model_dir=Path(args.out).parent,
            )

            # Save final model and training metrics for Optuna run
            model.save_state(args.out)
            print(f"\n[Optuna] Final model saved to {args.out}")

            metrics_path = Path("docking_metrics_optuna.json")
            with open(metrics_path, "w") as f:
                json.dump(train_metrics, f, indent=2)
            print(f"[Optuna] Training metrics saved to {metrics_path}")

            # Save combined hyperparameters used for final training
            final_hparams = {
                "input_dim": input_dim,
                "hidden_dim": best_hidden_dim,
                "lr": best_lr,
                "dropout": best_dropout,
                "weight_decay": best_weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "optuna": {
                    "storage": args.optuna_storage or "in-memory",
                    "n_trials": args.optuna_trials,
                    "sampler": "TPESampler",
                    "pruner": "MedianPruner(n_warmup_steps=5)",
                },
            }
            final_hparams_path = Path("hparams_docking_optuna.json")
            with open(final_hparams_path, "w") as f:
                json.dump(final_hparams, f, indent=2)
            print(f"[Optuna] Final training hyperparameters saved to {final_hparams_path}")

            # Evaluate on test set (also get predictions for diagnostics)
            test_metrics, y_true_test, y_pred_test = evaluate_model(
                model, test_rows, batch_size=args.batch_size, return_preds=True
            )

            print("\n[Optuna] Test Metrics:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save test metrics
            test_metrics_path = Path("docking_test_metrics_optuna.json")
            with open(test_metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"[Optuna] Test metrics saved to {test_metrics_path}")

            # Diagnostics: loss vs epoch plot
            history = train_metrics.get("history", [])
            if history:
                epochs_hist = [h["epoch"] for h in history]
                train_loss_hist = [h["train_loss"] for h in history]
                val_loss_hist = [h["val_loss"] for h in history]

                plt.figure(figsize=(6, 4))
                plt.plot(epochs_hist, train_loss_hist, label="Train Loss")
                plt.plot(epochs_hist, val_loss_hist, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.title("Docking Regressor Training Curve (Optuna)")
                plt.legend()
                plt.tight_layout()
                loss_plot_path = reports_dir / "loss_curve_optuna.png"
                plt.savefig(loss_plot_path, dpi=200)
                plt.close()
                print(f"[Optuna] Loss curve saved to {loss_plot_path}")

            # Diagnostics: predicted vs actual scatter plot
            plt.figure(figsize=(5, 5))
            plt.scatter(y_true_test, y_pred_test, alpha=0.6, edgecolor="k")
            min_val = float(min(y_true_test.min(), y_pred_test.min()))
            max_val = float(max(y_true_test.max(), y_pred_test.max()))
            plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
            plt.xlabel("Actual DockScore")
            plt.ylabel("Predicted DockScore")
            plt.title("Docking Regressor (Optuna): Predicted vs Actual (Test)")
            plt.legend()
            plt.tight_layout()
            scatter_plot_path = reports_dir / "pred_vs_actual_test_optuna.png"
            plt.savefig(scatter_plot_path, dpi=200)
            plt.close()
            print(
                f"[Optuna] Predicted vs actual scatter plot saved to {scatter_plot_path}"
            )

            # Generate concise markdown training report for Optuna run
            report_path = reports_dir / "training_report_optuna.md"
            with open(report_path, "w") as f:
                f.write("# Docking Regressor Training Report (Optuna)\n\n")
                f.write("## Optuna Settings\n\n")
                f.write(f"- **n_trials**: {args.optuna_trials}\n")
                f.write(
                    f"- **storage**: "
                    f"{args.optuna_storage or 'in-memory (no persistent storage)'}\n"
                )
                f.write("- **sampler**: TPESampler\n")
                f.write("- **pruner**: MedianPruner(n_warmup_steps=5)\n")

                f.write("\n## Best Hyperparameters (from Optuna)\n\n")
                for k, v in best_params.items():
                    f.write(f"- **{k}**: {v}\n")

                f.write("\n## Final Training Hyperparameters\n\n")
                for k, v in final_hparams.items():
                    if k == "optuna":
                        continue
                    f.write(f"- **{k}**: {v}\n")

                f.write("\n## Test Set Metrics\n\n")
                for k, v in test_metrics.items():
                    f.write(f"- **{k}**: {v:.4f}\n")

                f.write("\n## Artifacts\n\n")
                f.write(f"- **Model**: `{args.out}`\n")
                f.write(f"- **Training metrics**: `{metrics_path}`\n")
                f.write(f"- **Test metrics**: `{test_metrics_path}`\n")
                f.write(f"- **Optuna best params**: `{hparams_optuna_path}`\n")
                f.write(f"- **Optuna trials**: `{trials_path}`\n")
                f.write(f"- **Final hyperparameters**: `{final_hparams_path}`\n")
                f.write(f"- **Loss curve plot**: `{loss_plot_path}`\n")
                f.write(
                    f"- **Pred vs actual plot**: `{scatter_plot_path}`\n"
                )

                f.write("\n## Recommendations\n\n")
                f.write(
                    "- **Data**: Consider expanding the dataset and checking for label noise in `DockScore`.\n"
                )
                f.write(
                    "- **Modeling**: Try deeper or residual MLPs, or compare against gradient-boosted trees on ECFP.\n"
                )
                f.write(
                    "- **Features**: Explore 3D or protein–ligand features (e.g., graph neural networks) if performance plateaus.\n"
                )

            print(f"[Optuna] Training report written to {report_path}")

            # Final clean summary for console
            print("\n=== Final Summary (Optuna) ===")
            print("Best hyperparameters (from Optuna):")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            print("\nTest metrics:")
            print(
                f"  R2={test_metrics['r2']:.4f}, "
                f"RMSE={test_metrics['rmse']:.4f}, "
                f"MAE={test_metrics['mae']:.4f}, "
                f"Pearson r={test_metrics['pearson_r']:.4f}"
            )
            print(f"\nModel path: {args.out}")
            print(
                "For further improvements, see recommendations section "
                "of the Optuna training report."
            )
        else:
            # --- Original v1 manual hyperparameter search path (backwards compatible) ---
            best_hp = None
            hp_results = None
            if not args.no_hyperopt:
                hp_results = hyperparameter_search(
                    train_rows, val_rows, input_dim, max_trials=10
                )
                best_hp = hp_results.get("best_params")
                # Use best hyperparameters
                if best_hp:
                    args.hidden_dim = best_hp["hidden_dim"]
                    args.lr = best_hp["lr"]
                    args.weight_decay = best_hp["weight_decay"]
                    args.dropout = best_hp["dropout"]
                    print(
                        "\nUsing best hyperparameters: "
                        f"hidden_dim={args.hidden_dim}, lr={args.lr}, "
                        f"dropout={args.dropout}, weight_decay={args.weight_decay}"
                    )

            # Retrain on combined train+val with chosen hyperparameters
            train_full_rows = train_rows + val_rows
            print(
                f"\nRetraining on combined train+val set (n={len(train_full_rows)}) "
                "with best hyperparameters..."
            )
            model, train_metrics = train_model(
                train_full_rows,
                val_rows,  # still keep a small validation set for early stopping
                input_dim,
                args.hidden_dim,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                model_dir=Path(args.out).parent,
            )

            # Save final model to specified path
            model.save_state(args.out)
            print(f"\nModel saved to {args.out}")

            # Save training metrics
            metrics_path = Path("docking_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(train_metrics, f, indent=2)
            print(f"Training metrics saved to {metrics_path}")

            # Save hyperparameters
            hparams = {
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "lr": args.lr,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "best_search_results": hp_results,
            }
            hparams_path = Path("hparams_docking.json")
            with open(hparams_path, "w") as f:
                json.dump(hparams, f, indent=2)
            print(f"Hyperparameters saved to {hparams_path}")

            # Evaluate on test set (also get predictions for diagnostics)
            test_metrics, y_true_test, y_pred_test = evaluate_model(
                model, test_rows, batch_size=args.batch_size, return_preds=True
            )

            print("\nTest Metrics:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save test metrics
            test_metrics_path = Path("docking_test_metrics.json")
            with open(test_metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test metrics saved to {test_metrics_path}")

            # Diagnostics: loss vs epoch plot
            history = train_metrics.get("history", [])
            if history:
                epochs_hist = [h["epoch"] for h in history]
                train_loss_hist = [h["train_loss"] for h in history]
                val_loss_hist = [h["val_loss"] for h in history]

                plt.figure(figsize=(6, 4))
                plt.plot(epochs_hist, train_loss_hist, label="Train Loss")
                plt.plot(epochs_hist, val_loss_hist, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.title("Docking Regressor Training Curve")
                plt.legend()
                plt.tight_layout()
                loss_plot_path = reports_dir / "loss_curve.png"
                plt.savefig(loss_plot_path, dpi=200)
                plt.close()
                print(f"Loss curve saved to {loss_plot_path}")

            # Diagnostics: predicted vs actual scatter plot
            plt.figure(figsize=(5, 5))
            plt.scatter(y_true_test, y_pred_test, alpha=0.6, edgecolor="k")
            min_val = float(min(y_true_test.min(), y_pred_test.min()))
            max_val = float(max(y_true_test.max(), y_pred_test.max()))
            plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
            plt.xlabel("Actual DockScore")
            plt.ylabel("Predicted DockScore")
            plt.title("Docking Regressor: Predicted vs Actual (Test)")
            plt.legend()
            plt.tight_layout()
            scatter_plot_path = reports_dir / "pred_vs_actual_test.png"
            plt.savefig(scatter_plot_path, dpi=200)
            plt.close()
            print(f"Predicted vs actual scatter plot saved to {scatter_plot_path}")

            # Generate concise markdown training report
            report_path = reports_dir / "training_report.md"
            with open(report_path, "w") as f:
                f.write("# Docking Regressor Training Report\n\n")
                f.write("## Hyperparameters\n\n")
                for k, v in hparams.items():
                    f.write(f"- **{k}**: {v}\n")

                f.write("\n## Test Set Metrics\n\n")
                for k, v in test_metrics.items():
                    f.write(f"- **{k}**: {v:.4f}\n")

                f.write("\n## Artifacts\n\n")
                f.write(f"- **Model**: `{args.out}`\n")
                f.write(f"- **Training metrics**: `{metrics_path}`\n")
                f.write(f"- **Test metrics**: `{test_metrics_path}`\n")
                f.write(f"- **Hyperparameters**: `{hparams_path}`\n")
                f.write(f"- **Loss curve plot**: `{loss_plot_path}`\n")
                f.write(f"- **Pred vs actual plot**: `{scatter_plot_path}`\n")

                f.write("\n## Recommendations\n\n")
                f.write(
                    "- **Data**: Consider expanding the dataset and checking for label noise in `DockScore`.\n"
                )
                f.write(
                    "- **Modeling**: Try deeper or residual MLPs, or compare against gradient-boosted trees on ECFP.\n"
                )
                f.write(
                    "- **Features**: Explore 3D or protein–ligand features (e.g., graph neural networks) if performance plateaus.\n"
                )

            print(f"Training report written to {report_path}")

            # Final clean summary for console
            print("\n=== Final Summary ===")
            print("Best hyperparameters:")
            print(
                f"  hidden_dim={args.hidden_dim}, lr={args.lr}, "
                f"dropout={args.dropout}, weight_decay={args.weight_decay}"
            )
            print("\nTest metrics:")
            print(
                f"  R2={test_metrics['r2']:.4f}, "
                f"RMSE={test_metrics['rmse']:.4f}, "
                f"MAE={test_metrics['mae']:.4f}, "
                f"Pearson r={test_metrics['pearson_r']:.4f}"
            )
            print(f"\nModel path: {args.out}")
            print(
                "For further improvements, see recommendations section "
                "of the training report."
            )

            # Run baselines (optional)
            if not args.no_baselines:
                baseline_results = run_baselines(train_rows, val_rows, test_rows)

                print("\nBaseline Results:")
                for model_name, metrics in baseline_results.items():
                    print(f"\n{model_name}:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value:.4f}")

                baseline_path = Path("baselines_docking.json")
                with open(baseline_path, "w") as f:
                    json.dump(baseline_results, f, indent=2)
                print(f"\nBaseline metrics saved to {baseline_path}")


if __name__ == "__main__":
    main()
