import sys
import os
import re
import time
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) < 2:
    print("❌ Usage: python3 Generators/plot_metrics.py <path_to_training_log>")
    sys.exit(1)

log_path = sys.argv[1]
save_path = Path(log_path).parent / "live_recon_plot.png"
log_pattern = re.compile(r"Epoch\s+(\d+).*Recon:\s+([\d.]+)")

print(f"📈 Headless live plotting started. Checking {save_path} every 5 seconds.")
print("Press Ctrl+C to stop.")

last_mtime = 0

try:
    while True:
        try:
            # Check if the file has been modified since we last checked
            current_mtime = os.path.getmtime(log_path)
        except FileNotFoundError:
            print(f"Waiting for log file: {log_path}...")
            time.sleep(5)
            continue
            
        if current_mtime > last_mtime:
            epochs = []
            recon_losses = []
            
            with open(log_path, 'r') as f:
                for line in f:
                    match = log_pattern.search(line)
                    if match:
                        epochs.append(int(match.group(1)))
                        recon_losses.append(float(match.group(2)))
                        
            if epochs:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, recon_losses, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=4)
                
                plt.xlim(1, 100)
                max_recon = max(recon_losses)
                plt.ylim(0, max_recon * 1.05) 
                
                plt.title('🔴 LIVE: Transformer VAE Reconstruction Loss', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Reconstruction Loss', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"[{time.strftime('%X')}] Plot updated (Latest Epoch: {epochs[-1]})")
                
            last_mtime = current_mtime
            
        time.sleep(5)
except KeyboardInterrupt:
    print("\n🛑 Live plotting stopped.")
