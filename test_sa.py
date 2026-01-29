import os
import requests
import gzip
import pickle
from rdkit import Chem

# 1. Ensure Dependencies exist locally
SA_REPO_URL = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/"
for f in ["sascorer.py", "fpscores.pkl.gz"]:
    if not os.path.exists(f):
        print(f"⬇️ Downloading {f}...")
        r = requests.get(SA_REPO_URL + f)
        with open(f, 'wb') as file:
            file.write(r.content)

# 2. Force import from LOCAL directory to avoid using a broken system version
import sys
sys.path.insert(0, os.getcwd()) 
import sascorer

def test_sa():
    print("\n🧪 SA SCORE BENCHMARK")
    print("--------------------------------------------------")
    print(f"{'MOLECULE':<15} | {'EXPECTED':<10} | {'CALCULATED'}")
    print("--------------------------------------------------")

    benchmarks = [
        ("Benzene", "c1ccccc1", "1.0 - 1.5"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "1.5 - 2.0"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "2.0 - 2.5"),
        ("Taxol (Complex)", "CC(=O)OC1C2=C(C)C(CC(O)(C(OC(=O)c3ccccc3)C4C(O)C4(OC(=O)C)C2(C)C)C1(O)C(=O)c5ccccc5)OC(=O)C(O)C(NC(=O)c6ccccc6)c7ccccc7", "> 4.5"),
    ]

    for name, smi, expected in benchmarks:
        mol = Chem.MolFromSmiles(smi)
        score = sascorer.calculateScore(mol)
        print(f"{name:<15} | {expected:<10} | {score:.3f}")

    print("--------------------------------------------------")
    print("✅ IF 'CALCULATED' matches 'EXPECTED', your plot is VALID.")
    print("⚠️ IF all scores are exactly 1.0 or very high, the pickle file failed to load.")

if __name__ == "__main__":
    test_sa()