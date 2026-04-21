import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import joblib

def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def train_teacher_models(data_root, output_dir="teacher_models"):
    os.makedirs(output_dir, exist_ok=True)
    tasks = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    results = []
    for task in tasks:
        print(f"📖 Processing Task: {task}")
        task_dir = os.path.join(data_root, task)
        # Find train and test files
        train_file = [f for f in os.listdir(task_dir) if "train" in f][0]
        test_file = [f for f in os.listdir(task_dir) if "test" in f][0]
        
        df_train = pd.read_csv(os.path.join(task_dir, train_file))
        df_test = pd.read_csv(os.path.join(task_dir, test_file))
        
        # Get labels
        label_col = "bioclass" if "bioclass" in df_train.columns else df_train.columns[-1]
        
        def prepare_data(df):
            fps, labels = [], []
            for _, row in df.iterrows():
                fp = smiles_to_fp(row['SMILES'])
                if fp is not None:
                    fps.append(fp)
                    labels.append(row[label_col])
            return np.array(fps), np.array(labels)
        
        X_train, y_train = prepare_data(df_train)
        X_test, y_test = prepare_data(df_test)
        
        if len(np.unique(y_train)) < 2:
            print(f"⚠️ Skipping {task} - only one class.")
            continue
            
        # Train Teacher
        model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        # Evaluate
        probs = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        print(f"✅ {task} | AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}")
        
        # Save model
        joblib.dump(model, os.path.join(output_dir, f"{task}_teacher.pkl"))
        
        results.append({
            "Task": task,
            "AUC-ROC": auc_roc,
            "PR-AUC": pr_auc
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_dir, "teacher_performance.csv"), index=False)
    print("\n--- Teacher Model Summary ---")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    train_teacher_models("Models/AdmetClassifier/Auto_ML_dataset")
