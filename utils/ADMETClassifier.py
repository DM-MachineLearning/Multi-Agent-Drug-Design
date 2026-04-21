import logging
import torch
import torch.nn as nn
import os

# --- MMoE Architecture (Matches admet_predictor_balanced.pt) ---
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, output_dim)),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(output_dim, output_dim)),
            nn.SiLU()
        )
    def forward(self, x): return self.net(x)

class MMoEADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11, num_experts=8, expert_dim=512):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(latent_dim, expert_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, num_experts), nn.Softmax(dim=-1)) for _ in range(num_tasks)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 256), nn.SiLU(), nn.Linear(256, 1)) for _ in range(num_tasks)
        ])
        
    def forward(self, z):
        expert_outputs = torch.stack([expert(z) for expert in self.experts])
        final_outputs = []
        for i in range(self.num_tasks):
            gate_weights = self.gates[i](z).unsqueeze(2)
            combined_experts = torch.sum(expert_outputs.permute(1, 0, 2) * gate_weights, dim=1)
            final_outputs.append(self.heads[i](combined_experts))
        return torch.cat(final_outputs, dim=1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADMETClassifier:
    """
    Updated MMoE Classifier for predicting ADMET properties directly from 
    the VAE Latent Space (z). Uses Distillation + Balanced Sampling for high performance.
    """
    def __init__(self, model_path: str = "admet_predictor_balanced.pt", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.task_names = [
            'BBBP', 'CYP1A2_inhibition', 'CYP2C19_inhibition', 'CYP2C9_inhibition', 
            'CYP2D6_inhibition', 'CYP3A4_inhibition', 'Caco2_permeability', 
            'HLM_stability', 'P-gp_substrate', 'RLM_stability', 'hERG_inhibition'
        ]
        
        self.input_dim = 128
        # Ensure parameters match the "Final Stand" training
        self.model = MMoEADMET(latent_dim=self.input_dim, num_tasks=len(self.task_names), num_experts=8)

        logger.info(f"🚀 Loading MMoE ADMET model from {model_path}...")
        if not os.path.exists(model_path):
             model_path = os.path.join(os.getcwd(), model_path)
             
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device).eval()
        logger.info(f"✅ MMoE ADMET Classifier Ready (Mean AUC ~0.89)")

    def classify_admet(self, z: torch.Tensor) -> dict:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device)

        with torch.no_grad():
            logits = self.model(z)
            temperature = 2.0  
            probs = torch.sigmoid(logits / temperature).cpu().squeeze(0)
        
        return {task: probs[i].item() for i, task in enumerate(self.task_names)}
    
    def get_task_probability(self, z, task_name):
        z = z.to(self.device)
        logits = self.model(z)
        
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in ADMET model.")
            
        task_idx = self.task_names.index(task_name)
        target_logit = logits[:, task_idx]
        return torch.sigmoid(target_logit)
