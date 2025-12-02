"""Tests for docking regressor model."""
import os
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from madm.properties.docking_regressor import DockingRegressor
from madm.data.featurization import smiles_to_ecfp


def test_model_forward():
    """Test that forward pass works correctly."""
    model = DockingRegressor(input_dim=2048, hidden_dim=512)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 2048)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    print("✓ Forward pass test passed")


def test_model_predict():
    """Test single-sample prediction."""
    model = DockingRegressor(input_dim=2048, hidden_dim=512)
    model.eval()
    
    # Create dummy fingerprint
    fp = torch.randn(2048)
    
    # Predict
    prediction = model.predict(fp)
    
    assert isinstance(prediction, float), "Prediction should be a float"
    assert np.isfinite(prediction), "Prediction should be finite"
    
    print("✓ Single prediction test passed")


def test_model_predict_batch():
    """Test batch prediction."""
    model = DockingRegressor(input_dim=2048, hidden_dim=512)
    model.eval()
    
    # Create dummy batch
    batch_size = 8
    fp_batch = torch.randn(batch_size, 2048)
    
    # Predict
    predictions = model.predict_batch(fp_batch)
    
    assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
    assert predictions.shape == (batch_size,), f"Expected shape ({batch_size},), got {predictions.shape}"
    assert np.all(np.isfinite(predictions)), "Predictions should be finite"
    
    print("✓ Batch prediction test passed")


def test_model_save_load():
    """Test saving and loading model state."""
    model = DockingRegressor(input_dim=2048, hidden_dim=512, dropout=0.1)
    
    # Create dummy input
    x = torch.randn(2, 2048)
    output_before = model(x)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        model.save_state(temp_path)
        assert os.path.exists(temp_path), "Model file should exist"
        
        # Load
        model_loaded = DockingRegressor.load_state(
            temp_path, map_location='cpu', input_dim=2048, hidden_dim=512, dropout=0.1
        )
        
        # Check output matches
        output_after = model_loaded(x)
        assert torch.allclose(output_before, output_after, atol=1e-5), "Outputs should match after save/load"
        
        print("✓ Save/load test passed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_training_loop():
    """Test training on synthetic dataset."""
    # Create synthetic dataset
    synthetic_smiles = [
        "CCO",  # Ethanol
        "CCN",  # Ethylamine
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCc1ccccc1",  # Ethylbenzene
        "CC(C)C",  # Isobutane
        "C1CCC(CC1)O",  # Cyclohexanol
        "CCCCCCCC",  # Octane
    ]
    
    # Generate fingerprints
    fps = [smiles_to_ecfp(smiles) for smiles in synthetic_smiles]
    fp_tensor = torch.stack(fps)
    input_dim = len(fps[0])
    
    # Create random targets
    targets = torch.randn(len(synthetic_smiles)) * 2 - 5  # Random values around -5
    
    # Initialize model
    model = DockingRegressor(input_dim=input_dim, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    # Train for a few epochs
    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        predictions = model(fp_tensor)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    
    # Check predictions
    model.eval()
    with torch.no_grad():
        final_predictions = model(fp_tensor)
        assert final_predictions.shape == targets.shape, "Predictions shape should match targets"
        assert torch.all(torch.isfinite(final_predictions)), "Predictions should be finite"
    
    print("✓ Training loop test passed")


def test_dropout():
    """Test that dropout is applied when dropout > 0."""
    model_no_dropout = DockingRegressor(input_dim=2048, hidden_dim=512, dropout=0.0)
    model_with_dropout = DockingRegressor(input_dim=2048, hidden_dim=512, dropout=0.5)
    
    # Count dropout layers
    dropout_layers_no = sum(1 for m in model_no_dropout.modules() if isinstance(m, torch.nn.Dropout))
    dropout_layers_with = sum(1 for m in model_with_dropout.modules() if isinstance(m, torch.nn.Dropout))
    
    assert dropout_layers_no == 0, "No dropout layers should exist when dropout=0"
    assert dropout_layers_with > 0, "Dropout layers should exist when dropout>0"
    
    print("✓ Dropout test passed")


def run_all_tests():
    """Run all tests."""
    print("Running docking regressor tests...\n")
    
    try:
        test_model_forward()
        test_model_predict()
        test_model_predict_batch()
        test_model_save_load()
        test_training_loop()
        test_dropout()
        
        print("\n✓ All tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

