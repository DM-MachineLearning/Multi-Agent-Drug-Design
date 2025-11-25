"""Skeleton script for training the activity classifier.

You must plug in real data loading.
"""
import torch
from torch.utils.data import DataLoader
from madm.classifier import ActivityClassifier
from madm.data.datasets import MoleculePropertyDataset

def main():
    # TODO: load data as list of dicts
    rows = []  # e.g. from a CSV
    dataset = MoleculePropertyDataset(rows)
    if not len(dataset):
        print("No data loaded - please implement dataset loading.")
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ActivityClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 51):
        total_loss = 0.0
        correct = 0
        total = 0
        for fp, props, labels in loader:
            logits = model(fp, props)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(labels)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum())
            total += len(labels)
        print(f"Epoch {epoch}: loss={total_loss/len(dataset):.4f}, acc={correct/total:.3f}")

    torch.save(model.state_dict(), "activity_classifier.pt")

if __name__ == "__main__":
    main()
