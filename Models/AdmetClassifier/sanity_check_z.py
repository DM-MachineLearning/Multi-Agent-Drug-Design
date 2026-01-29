import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# Load your latent data
print("📥 Loading data...")
data_pack = torch.load("admet_latent_train_bidirec.pt", weights_only=False)
samples = data_pack['data']
task_list = data_pack['tasks']

# Let's target the "Problem Child": CYP2D6_inhibition
target_task = "BBBP"
task_idx = task_list.index(target_task)

print(f"🕵️‍♂️ Investigating Task: {target_task} (Index {task_idx})")

# Extract X (latent z) and y (label) ONLY for this task
X = []
y = []

for item in samples:
    # item['task_idx'] is a scalar tensor, check if it matches
    if item['task_idx'] == task_idx:
        X.append(item['z'])
        y.append(item['y'])

X = np.array(X)
y = np.array(y)

print(f"📊 Found {len(X)} samples for {target_task}")
print(f"   Positives: {np.sum(y==1)} | Negatives: {np.sum(y==0)}")

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost (The Gold Standard)
print("\n🚀 Training XGBoost Baseline...")
model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    eval_metric='logloss',
    scale_pos_weight=(len(y) - sum(y)) / sum(y) # Handle imbalance automatically
)

model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print("\n" + "="*40)
print(f"RESULT FOR {target_task}")
print("="*40)
print(f"ACCURACY: {acc:.4f}")
print(f"AUC-ROC:  {auc:.4f}")
print(f"F1 SCORE: {f1:.4f}")
print("="*40)

if f1 < 0.60:
    print("\n❌ VERDICT: The Latent Space (z) is the bottleneck.")
    print("   XGBoost (which usually beats NNs on tabular data) failed.")
    print("   This means 'z' does not contain enough info to distinguish Active vs Inactive.")
    print("   SOLUTION: You need to improve your VAE or use Morgan Fingerprints.")
else:
    print("\n✅ VERDICT: The Latent Space is GOOD.")
    print("   Since XGBoost learned it, the Neural Net *should* be able to.")