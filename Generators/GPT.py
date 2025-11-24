from pathlib import Path
import yaml
from typing import Optional

from transformers import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_config_path = Path(__file__).resolve().parent / "config.yaml"

GPT_Model_Path: Optional[str] = None
if _config_path.is_file():
    try:
        with _config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            GPT_Model_Path = cfg.get("GPT_Model_Path")
    except Exception:
        GPT_Model_Path = None

class GPT:
    """
    A class representing a GPT model for drug generation. 

    Attributes:
        model_path (str): The file path to the pre-trained GPT model.

    Methods:
        load_model(): Loads the GPT model from the specified path.
        generate_molecule(prompt: str) -> str: Generates a drug-like molecule.
        fine_tune(dataset_path: str): Fine-tunes the GPT model using a dataset located at the specified path.

    Example:
        gpt = GPT(model_path="path/to/model")
        gpt.load_model()
        molecule = gpt.generate_molecule(prompt="Generate a molecule with anti-cancer properties.")
        gpt.fine_tune(dataset_path="path/to/dataset")
    """

    def __init__(self, model_path: Optional[str] = GPT_Model_Path):
        """
        Initializes the GPT model with the specified model path.

        Args:
            model_path (str, optional): The file path to the pre-trained GPT model. If not provided, defaults to the path specified in the configuration file.
        
        Raises:
            ValueError: If the model_path is not provided and not found in the configuration.
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        if self.model_path is None:
            raise ValueError("Model path must be provided either through the constructor or the configuration file.")
    
    def load_model(self):
        """
        Loads the GPT model and tokenizer from self.model_path, moves the model to an available device,
        and puts it in evaluation mode.

        After calling this, self.model, self.tokenizer and self.device will be set.
        """

        if not self.model_path:
            raise ValueError("No model_path set for loading the model.")

        try:
            # Load tokenizer and model (supports model identifier or local path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            # Allow remote code if model requires custom classes; remove if you want to avoid executing model code.
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)

            # Move model to device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Put model in eval mode (typical for generation/inference)
            self.model.eval()

            return self.model
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from '{self.model_path}': {exc}") from exc
        
    def generate_molecule(self, prompt: Optional[str] = None) -> str:
        """
        Generates a drug-like molecule using the GPT model.
        If prompt is None, generation starts from a BOS/pad/eos token (if available).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating molecules. Call load_model() first.")

        # Prepare inputs: either tokenize the prompt or synthesize a single start token
        if prompt:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        else:
            # try tokenizer/model config for a valid start token id
            bos_id = getattr(self.tokenizer, "bos_token_id", None) or getattr(self.model.config, "bos_token_id", None)
            pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.model.config, "pad_token_id", None)
            eos_id = getattr(self.tokenizer, "eos_token_id", None) or getattr(self.model.config, "eos_token_id", None)

            start_id = bos_id or pad_id or eos_id
            if start_id is None:
                # fallback: tokenize empty string (may produce valid ids)
                inputs = self.tokenizer("", return_tensors="pt", add_special_tokens=False).to(self.device)
                if inputs["input_ids"].numel() == 0:
                    raise RuntimeError("Unable to construct a valid start token for generation.")
            else:
                inputs = {
                    "input_ids": torch.tensor([[start_id]], device=self.device),
                    "attention_mask": torch.tensor([[1]], device=self.device)
                }

        # Generate output tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )

        # Decode generated tokens to string
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt if one was provided
        if prompt:
            return generated_text[len(prompt):].strip()
        return generated_text.strip()