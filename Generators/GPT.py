import time
import yaml
import torch
from pathlib import Path
from typing import Optional, List, Dict
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, BitsAndBytesConfig

from ..Datasets.SMILES import SMILESDataset

# --- Configuration Loading ---
_config_path = Path(__file__).resolve().parent / "config.yaml"
LLM_Model_Path: Optional[str] = None

if _config_path.is_file():
    try:
        with _config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            LLM_Model_Path = cfg.get("LLM_Model_Path")
    except Exception:
        LLM_Model_Path = None

class LLM:
    """
    A class representing a LLM model for drug generation. 

    Attributes:
        model_path (str): The file path to the pre-trained LLM model.
    """

    def __init__(self, model_path: Optional[str] = LLM_Model_Path):
        """
        Initializes the LLM model with the specified model path.
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = False

    def load_model(self, use_lora: bool = False):
        """
        Loads model. If use_lora=True, loads in 4-bit quantization for Llama/Mistral efficiency.
        """
        self.use_lora = use_lora
        
        if not self.model_path:
            raise ValueError("No model_path set.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            
            if self.use_lora:
                print(f"Loading {self.model_path} in 4-bit precision for LoRA...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.model = prepare_model_for_kbit_training(self.model)
                
                # Attach LoRA adapters
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    r=8, 
                    lora_alpha=32, 
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj"]
                )
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
                
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
                self.model.to(self.device)

            self.model.eval()
            return self.model
            
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    def _prepare_model_for_training(self):
        """
        Internal helper: Ensures a model is loaded (custom or base GPT2) 
        and the tokenizer is configured correctly for padding.
        """
        # 1. Try to load configured model if not loaded
        if self.model is None and self.model_path:
            try:
                self.load_model()
            except Exception:
                pass # Fallback below

        # 2. Fallback to GPT-2 if still no model
        if self.model is None:
            base = "gpt2"
            try:
                print(f"Loading base model {base} for fine-tuning...")
                self.tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
                self.model = AutoModelForCausalLM.from_pretrained(base)
                self.model.to(self.device)
            except Exception as exc:
                raise RuntimeError(f"Failed to load base model '{base}': {exc}") from exc

        # 3. Fix Padding Token (Crucial for batch training)
        if self.tokenizer.pad_token is None:
            # Use EOS as PAD if available, or add a new token
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            if self.tokenizer.pad_token == "[PAD]":
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
            # Resize model embeddings to fit new pad token
            self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_molecule(self, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generates a drug-like molecule using the LLM model.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/Tokenizer not loaded. Call load_model() first.")

        # Determine start token
        sos_id = getattr(self.tokenizer, "sos_token_id", None)
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        
        # Fallback logic for start token
        start_id = sos_id if sos_id is not None else (bos_id if bos_id is not None else pad_id)
        
        if start_id is None:
            # As a last resort, use the first token in vocabulary (usually !)
            start_id = 0 

        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        inputs = {
            "input_ids": torch.tensor([[start_id]], device=self.device),
            "attention_mask": torch.tensor([[1]], device=self.device),
        }

        gen_kwargs = dict(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
        )

        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def fine_tune(self, dataset_path: str, epochs: int = 1, batch_size: int = 8, lr: float = 5e-5):
        """
        Fine-tunes the LLM model using a SMILES dataset.
        
        Args:
            dataset_path (str): Path to the .smi or .txt dataset.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            lr (float): Learning rate.
        """
        ds_path = Path(dataset_path)

        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {ds_path}")
        
        # Handle Directory vs File input
        target_file = ds_path
        if ds_path.is_dir():
            files = list(ds_path.glob("**/*.smi")) + list(ds_path.glob("**/*.txt"))
            if not files:
                raise FileNotFoundError("No .smi or .txt files found in directory.")
            target_file = files[0]

        # Ensure Model Resources
        self._prepare_model_for_training()

        # Prepare Data
        print(f"Loading data from {target_file}...")
        dataset = SMILESDataset(target_file, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # NOTE: You might want to force use_lora=True here if you detect a large model name
        if self.model is None:
            # Simple heuristic: if 'llama' or 'mistral' in name, use LoRA
            is_large = self.model_path and any(x in self.model_path.lower() for x in ['llama', 'mistral', 'falcon'])
            self.load_model(use_lora=is_large)

        # Setup Optimizer
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumerate(loader):
                # Move batch to device
                b_input_ids = batch["input_ids"].to(self.device)
                b_masks = batch["attention_mask"].to(self.device)
                b_labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=b_input_ids, 
                    attention_mask=b_masks, 
                    labels=b_labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        timestamp = int(time.time())
        save_dir = Path.cwd() / f"finetuned_chembl_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            print(f"Model saved to: {save_dir}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save fine-tuned model: {exc}") from exc

        return str(save_dir)