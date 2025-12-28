import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np

# --- CONFIG ---
DATA_PATH = "Models/ActivityClassifier/latent_dataset.pt"
MODEL_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
BATCH_SIZE = 32

# --- RE-DEFINE MODEL (Must match training) ---
class LatentPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super(LatentPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        return self.net(z)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # 1. Load Data
    data = torch.load(DATA_PATH)
    X_all = data['z']
    y_all = data['y']
    
    # 2. Split Data (Create a temporary Test Set)
    # We'll use 20% of the data for testing
    total_size = len(X_all)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    
    full_dataset = TensorDataset(X_all, y_all)
    # Using a fixed seed for reproducibility so we test on the same 'random' chunk
    generator = torch.Generator().manual_seed(42) 
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Testing on {len(test_dataset)} unseen samples...")

    # 3. Load Model
    model = LatentPredictor().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model file not found!")
        return

    # 4. Inference Loop
    y_true = []
    y_probs = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # Get Probabilities (0.0 to 1.0)
            probs = model(X_batch)
            
            # Convert to Class Predictions (0 or 1)
            preds = (probs > 0.6).float()
            
            # Store results
            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.numpy())

    # 5. Calculate Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # ROC-AUC (Area Under Curve)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0 # Handle case with only one class
        
    print("\n" + "="*30)
    print("   MODEL PERFORMANCE REPORT   ")
    print("="*30)
    print(f"Accuracy:   {acc:.2%}")
    print(f"ROC-AUC:    {auc:.4f} (Closer to 1.0 is better)")
    print("-" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 30)
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=['Inactive', 'Active']))

if __name__ == "__main__":
    evaluate()