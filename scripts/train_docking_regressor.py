"""Example training script for the docking regressor.

Fill in dataset loading with your own CSV / smiles + docking scores.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from madm.properties import DockingRegressor
from madm.data.featurization import smiles_to_ecfp

class DockingDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        fp = smiles_to_ecfp(row["smiles"])
        y = torch.tensor(row["dock"], dtype=torch.float32)
        return fp, y

def main():
    # TODO: load your docking dataset here as list of dicts
    rows = []  # [{"smiles": "...", "dock": -8.3}, ...]
    dataset = DockingDataset(rows)
    if not len(dataset):
        print("No data loaded - please implement dataset loading.")
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = DockingRegressor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, 51):
        total_loss = 0.0
        for fps, y_true in loader:
            y_pred = model(fps)
            loss = loss_fn(y_pred, y_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(fps)
        print(f"Epoch {epoch}: loss={total_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), "docking_regressor.pt")

if __name__ == "__main__":
    main()
