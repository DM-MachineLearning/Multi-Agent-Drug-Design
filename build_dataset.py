import torch
import pandas as pd
from tokenizers import Tokenizer

def create_dataset():
    DATA_PATH = "data/chembl_33_chemreps.txt"
    VOCAB_PATH = "vocab.json"
    OUTPUT_PATH = "data/chembl_train.pt"
    MAX_LEN = 128
    
    print("1. Loading Tokenizer...")
    # Load directly using the native tokenizers library
    tokenizer = Tokenizer.from_file(VOCAB_PATH)
    
    # Configure padding and truncation natively
    tokenizer.enable_padding(length=MAX_LEN, pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    print(f"2. Loading raw data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep='\t', usecols=['canonical_smiles'])
    smiles_list = df['canonical_smiles'].dropna().tolist()
    print(f"   Found {len(smiles_list)} valid SMILES.")
    
    print("3. Prepending/Appending special tokens...")
    formatted_smiles = ["<s>" + smi + "</s>" for smi in smiles_list]
    
    print("4. Tokenizing and padding (backed by Rust, this takes ~10 seconds)...")
    encoded = tokenizer.encode_batch(formatted_smiles)
    
    print("5. Converting to PyTorch tensor...")
    input_ids = torch.tensor([e.ids for e in encoded], dtype=torch.long)
    
    print(f"   Final tensor shape: {input_ids.shape}")
    
    print(f"6. Saving to {OUTPUT_PATH}...")
    torch.save(input_ids, OUTPUT_PATH)
    print("✅ Dataset successfully generated and ready for training!")

if __name__ == "__main__":
    create_dataset()
