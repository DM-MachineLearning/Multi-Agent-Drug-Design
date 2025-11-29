"""Inference helper for docking score prediction."""
import torch
from typing import List
from pathlib import Path

from madm.properties.docking_regressor import DockingRegressor
from madm.data.featurization import smiles_to_ecfp


def load_model(model_path: str, input_dim: int = 2048, hidden_dim: int = 512, dropout: float = 0.0, map_location: str = 'cpu') -> DockingRegressor:
    """Load a trained docking regressor model.
    
    Args:
        model_path: Path to saved model state dict
        input_dim: Input dimension (must match training)
        hidden_dim: Hidden dimension (must match training)
        dropout: Dropout rate (must match training)
        map_location: Device to load on ('cpu' or 'cuda')
        
    Returns:
        Loaded DockingRegressor model
    """
    model = DockingRegressor.load_state(
        model_path, map_location=map_location,
        input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout
    )
    return model


def featurize_smiles_list(smiles_list: List[str]) -> torch.Tensor:
    """Convert a list of SMILES strings to ECFP fingerprints.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tensor of shape (batch_size, fingerprint_dim)
    """
    fps = [smiles_to_ecfp(smiles) for smiles in smiles_list]
    return torch.stack(fps)


def predict_smiles_list(smiles_list: List[str], model_path: str, input_dim: int = 2048, hidden_dim: int = 512, dropout: float = 0.0) -> List[float]:
    """Predict docking scores for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        model_path: Path to saved model state dict
        input_dim: Input dimension (must match training)
        hidden_dim: Hidden dimension (must match training)
        dropout: Dropout rate (must match training)
        
    Returns:
        List of predicted docking scores
    """
    # Load model
    model = load_model(model_path, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    
    # Featurize
    fp_batch = featurize_smiles_list(smiles_list)
    
    # Predict
    predictions = model.predict_batch(fp_batch)
    
    return predictions.tolist()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m madm.properties.infer_docking <model_path> <smiles1> [smiles2] ...")
        sys.exit(1)
    
    model_path = sys.argv[1]
    smiles_list = sys.argv[2:]
    
    predictions = predict_smiles_list(smiles_list, model_path)
    
    print("Predictions:")
    for smiles, pred in zip(smiles_list, predictions):
        print(f"  {smiles}: {pred:.4f}")

