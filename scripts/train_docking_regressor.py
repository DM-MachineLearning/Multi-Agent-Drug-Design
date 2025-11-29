"""Training script for the docking regressor with full CLI, data loading, and evaluation."""
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

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
    model_dir: str = "models"
) -> Tuple[DockingRegressor, Dict]:
    """Train the docking regressor model."""
    # Create datasets
    train_dataset = DockingDataset(train_rows)
    val_dataset = DockingDataset(val_rows)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DockingRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
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
                y_pred = model(fps)
                loss = loss_fn(y_pred, y_true)
                val_loss += float(loss.item()) * len(fps)
        val_loss /= len(val_dataset)
        
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
        str(model_path), map_location='cpu', input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout
    )
    
    print(f"\nBest model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    
    return model, {'history': metrics_history, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss}


def evaluate_model(model: DockingRegressor, test_rows: List[Dict], batch_size: int = 64) -> Dict[str, float]:
    """Evaluate model on test set."""
    test_dataset = DockingDataset(test_rows)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for fps, y_true in test_loader:
            y_pred = model.predict_batch(fps)
            all_preds.extend(y_pred)
            all_targets.extend(y_true.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    return compute_metrics(all_targets, all_preds)


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
    max_trials: int = 8
) -> Dict:
    """Simple grid search for hyperparameters."""
    print("\nRunning hyperparameter search...")
    
    param_grid = {
        'hidden_dim': [256, 512],
        'lr': [1e-3, 1e-4],
        'weight_decay': [0, 1e-4]
    }
    
    best_score = float('inf')
    best_params = None
    results = []
    
    trials = 0
    for hidden_dim in param_grid['hidden_dim']:
        for lr in param_grid['lr']:
            for weight_decay in param_grid['weight_decay']:
                if trials >= max_trials:
                    break
                
                print(f"\nTrial {trials + 1}/{max_trials}: hidden_dim={hidden_dim}, lr={lr}, weight_decay={weight_decay}")
                
                # Quick training (fewer epochs for search)
                model, train_metrics = train_model(
                    train_rows, val_rows, input_dim, hidden_dim,
                    batch_size=64, lr=lr, epochs=10, weight_decay=weight_decay,
                    patience=3, model_dir="models/temp"
                )
                
                val_metrics = evaluate_model(model, val_rows)
                val_loss = val_metrics['rmse']  # Use RMSE as selection metric
                
                results.append({
                    'hidden_dim': hidden_dim,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'val_rmse': val_loss
                })
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = {'hidden_dim': hidden_dim, 'lr': lr, 'weight_decay': weight_decay}
                
                trials += 1
                if trials >= max_trials:
                    break
            if trials >= max_trials:
                break
        if trials >= max_trials:
            break
    
    print(f"\nBest hyperparameters: {best_params} (val_rmse={best_score:.4f})")
    
    return {'best_params': best_params, 'all_results': results}


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
    parser.add_argument('--no-hyperopt', action='store_true', help='Skip hyperparameter search')
    
    args = parser.parse_args()
    
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
        
        # Hyperparameter search (optional)
        if not args.no_hyperopt:
            hp_results = hyperparameter_search(train_rows, val_rows, input_dim, max_trials=8)
            # Use best hyperparameters
            if hp_results['best_params']:
                args.hidden_dim = hp_results['best_params']['hidden_dim']
                args.lr = hp_results['best_params']['lr']
                args.weight_decay = hp_results['best_params']['weight_decay']
                print(f"\nUsing best hyperparameters: hidden_dim={args.hidden_dim}, lr={args.lr}, weight_decay={args.weight_decay}")
        
        # Train model
        model, train_metrics = train_model(
            train_rows, val_rows, input_dim, args.hidden_dim,
            batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
            weight_decay=args.weight_decay, dropout=args.dropout,
            model_dir=Path(args.out).parent
        )
        
        # Save final model to specified path
        model.save_state(args.out)
        print(f"\nModel saved to {args.out}")
        
        # Save training metrics
        metrics_path = Path("docking_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        print(f"Training metrics saved to {metrics_path}")
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_rows, batch_size=args.batch_size)
        
        print("\nTest Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save test metrics
        test_metrics_path = Path("docking_test_metrics.json")
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test metrics saved to {test_metrics_path}")
        
        # Run baselines (optional)
        if not args.no_baselines:
            baseline_results = run_baselines(train_rows, val_rows, test_rows)
            
            print("\nBaseline Results:")
            for model_name, metrics in baseline_results.items():
                print(f"\n{model_name}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
            
            baseline_path = Path("baselines_docking.json")
            with open(baseline_path, 'w') as f:
                json.dump(baseline_results, f, indent=2)
            print(f"\nBaseline metrics saved to {baseline_path}")


if __name__ == "__main__":
    main()
