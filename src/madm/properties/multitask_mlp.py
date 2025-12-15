# src/madm/models/multitask_mlp.py

import torch
import torch.nn as nn
from typing import List, Dict


class SharedTrunk(nn.Module):
    """
    Shared feature extractor for all ADMET tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiTaskADMETModel(nn.Module):
    """
    Shared-trunk, task-specific-head ADMET classifier.

    One binary head per task.
    """

    def __init__(
        self,
        input_dim: int,
        shared_hidden_dims: List[int],
        task_names: List[str],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.task_names = task_names

        # Shared trunk
        self.trunk = SharedTrunk(
            input_dim=input_dim,
            hidden_dims=shared_hidden_dims,
            dropout=dropout,
        )

        trunk_output_dim = shared_hidden_dims[-1]

        # Task-specific heads
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(trunk_output_dim, 1)
                for task in task_names
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dict: {task_name: logits tensor of shape (N,)}
        """
        shared_repr = self.trunk(x)

        outputs = {}
        for task, head in self.heads.items():
            logits = head(shared_repr).squeeze(-1)
            outputs[task] = logits

        return outputs
