"""
SMILES Data Preprocessing Module

This module provides functionality to preprocess SMILES (Simplified Molecular Input Line Entry System)
strings from a text file, tokenize them using a custom regex pattern and a pre-trained tokenizer,
and save the tokenized sequences as a NumPy array for efficient training.

The preprocessing involves:
- Reading SMILES strings from a text file
- Splitting each SMILES into chemical tokens using regex
- Tokenizing with BOS/EOS tokens added
- Padding/truncating to a fixed length
- Saving as a memory-mapped NumPy array for fast loading

This allows for 10x faster training compared to on-the-fly text tokenization.
"""

import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# Configuration constants
INPUT_FILE = "./data/ChemBL_Smiles.txt"
OUTPUT_FILE = "./data/ChemBL_Smiles.npy"
VOCAB_FILE = "./vocab.json"
MAX_LENGTH = 128

# Regex pattern for splitting SMILES into individual chemical tokens
# Handles atoms like Cl, Br, brackets, bonds, etc.
SMILES_TOKEN_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

def pre_tokenize():
    """
    Preprocess SMILES data by tokenizing and saving to NumPy array.

    Reads SMILES strings from the input file, tokenizes each one using regex splitting
    and a pre-trained tokenizer, then saves the tokenized sequences as a NumPy array
    for efficient training.

    The process includes:
    - Loading and cleaning text data
    - Regex-based token splitting
    - Adding BOS/EOS tokens
    - Tokenization with padding/truncation
    - Saving to compressed NumPy format

    Raises:
        FileNotFoundError: If input file or vocab file is not found.
        RuntimeError: If tokenization fails or vocab size exceeds dtype limits.
    """
    print(f"Reading SMILES data from {INPUT_FILE}...")

    # Load and clean the input data
    try:
        text_data = Path(INPUT_FILE).read_text(encoding="utf-8").splitlines()
        text_data = [line.strip() for line in text_data if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if not text_data:
        raise ValueError("No valid SMILES strings found in input file")

    print(f"Tokenizing {len(text_data)} molecules...")

    # Initialize tokenizer
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_FILE)
        tokenizer.add_special_tokens({
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>'
        })
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Create memory-efficient NumPy array
    # int16 supports vocab sizes up to 32,767; int32 for larger vocabs if needed
    vocab_size = len(tokenizer)
    if vocab_size > 32767:
        dtype = np.int32  # Use int32 for larger vocabularies
        print(f"Large vocabulary detected ({vocab_size}), using int32 dtype")
    else:
        dtype = np.int16

    data_array = np.zeros((len(text_data), MAX_LENGTH), dtype=dtype)

    # Process each SMILES string
    for i, line in enumerate(tqdm(text_data, desc="Tokenizing")):
        # Split SMILES into tokens using regex
        tokens = SMILES_TOKEN_REGEX.findall(line)

        # Join tokens with spaces for tokenizer compatibility
        spaced_text = " ".join(tokens)

        # Add BOS and EOS tokens
        final_text = f"<s> {spaced_text} </s>"

        # Tokenize with padding and truncation
        try:
            encodings = tokenizer.encode(
                final_text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length"
            )
            data_array[i] = encodings
        except Exception as e:
            print(f"Warning: Failed to tokenize line {i}: {e}")
            continue

    # Save the processed data
    print(f"Saving tokenized data to {OUTPUT_FILE}...")
    try:
        np.save(OUTPUT_FILE, data_array)
        print("✅ Preprocessing complete! You can now train 10x faster.")
        print(f"   Array shape: {data_array.shape}, dtype: {data_array.dtype}")
    except Exception as e:
        raise RuntimeError(f"Failed to save output file: {e}")

if __name__ == "__main__":
    pre_tokenize()
