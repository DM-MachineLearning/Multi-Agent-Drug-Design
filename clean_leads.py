import pandas as pd
import os

def clean_and_count(file_path, output_path):
    """
    Reads a CSV, counts 'Latent_Vector_Only' rows, removes them,
    and drops the 'captions' column.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Input file '{file_path}' not found.")
        return

    print(f"📂 Reading {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 1. Count Total Rows
    total_rows = len(df)

    # 2. Count "Ghost" Rows (Latent_Vector_Only)
    if 'smiles' in df.columns:
        ghost_mask = df['smiles'] == 'Latent_Vector_Only'
        num_ghosts = ghost_mask.sum()
        # Filter them out
        clean_df = df[~ghost_mask].copy()
    else:
        print("⚠️ Warning: 'smiles' column not found. Skipping row filtering.")
        clean_df = df.copy()
        num_ghosts = 0

    # 3. Remove 'captions' column
    dropped_captions = False
    if 'captions' in clean_df.columns:
        clean_df.drop(columns=['captions'], inplace=True)
        dropped_captions = True

    # 4. Save Clean Version
    clean_df.to_csv(output_path, index=False)

    # --- REPORT ---
    print("\n" + "="*40)
    print("🧹 CSV CLEANING REPORT")
    print("="*40)
    print(f"Original Rows:         {total_rows}")
    print(f"❌ 'Latent' Rows:      {num_ghosts} (Removed)")
    print(f"✂️ 'captions' Column:   {'Removed' if dropped_captions else 'Not Found'}")
    print(f"✅ Final Valid Rows:   {len(clean_df)}")
    print("-" * 40)
    print(f"💾 Clean file saved to: {output_path}")
    print("="*40)

if __name__ == "__main__":
    # INPUT_FILE = "outputs/successful_molecules.csv"
    # INPUT_FILE = "outputs/exploration_updateMeanVar.csv"
    INPUT_FILE = "outputs/exploration_updateMeanVar_50update.csv"
    # OUTPUT_FILE = "outputs/successful_molecules_CLEAN.csv"
    # OUTPUT_FILE = "outputs/exploration_updateMeanVar_CLEAN.csv"
    OUTPUT_FILE = "outputs/exploration_updateMeanVar_50update_CLEAN.csv"
    
    clean_and_count(INPUT_FILE, OUTPUT_FILE)