# # # import pickle
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import matplotlib.gridspec as gridspec
# # # from rdkit import Chem
# # # from rdkit.Chem.Draw import rdMolDraw2D
# # # from sklearn.decomposition import PCA
# # # import seaborn as sns
# # # import os

# # # # Create output directory for your paper assets
# # # OUTPUT_DIR = "Paper_Assets"
# # # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # # Set academic style
# # # sns.set_theme(style="whitegrid", context="paper")
# # # plt.rcParams.update({'font.family': 'serif', 'font.size': 14})

# # # def load_trajectory(path="optimization_trajectory.pkl"):
# # #     with open(path, "rb") as f:
# # #         return pickle.load(f)

# # # def save_molecule_svg(mol, filename, score, step):
# # #     """
# # #     Generates an SVG file for a molecule using the C++ engine directly.
# # #     Bypasses the Python PIL/Cairo requirement completely.
# # #     """
# # #     if mol is None: return
    
# # #     # Use the SVG Drawer - No Cairo dependency
# # #     drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    
# # #     # specific drawing options for publication quality
# # #     opts = drawer.drawOptions()
# # #     opts.clearBackground = False
# # #     opts.addStereoAnnotation = True
    
# # #     drawer.DrawMolecule(mol)
# # #     drawer.FinishDrawing()
# # #     svg = drawer.GetDrawingText()
    
# # #     # Add a title/score to the SVG text manually if needed, 
# # #     # but for papers, you usually want just the molecule.
# # #     with open(filename, 'w') as f:
# # #         f.write(svg)

# # # def generate_paper_figures(history):
# # #     print(f"📂 Generatings assets in '{OUTPUT_DIR}/'...")
    
# # #     # --- Data Extraction ---
# # #     steps = [h['step'] for h in history]
# # #     best_scores = [h['score'] for h in history]
# # #     avg_scores = [h['avg_score'] for h in history]
# # #     z_vectors = np.concatenate([h['z_vector'] for h in history], axis=0)
    
# # #     # ==========================================
# # #     # FIGURE 1: QUANTITATIVE METRICS (PNG)
# # #     # ==========================================
# # #     fig = plt.figure(figsize=(12, 5))
# # #     gs = gridspec.GridSpec(1, 2)
    
# # #     # Panel A: Optimization Curve
# # #     ax1 = fig.add_subplot(gs[0, 0])
# # #     ax1.plot(steps, best_scores, label='Best Agent', color='#C0392B', linewidth=2.5)
# # #     ax1.plot(steps, avg_scores, label='Swarm Avg', color='#2980B9', linestyle='--')
# # #     ax1.fill_between(steps, avg_scores, best_scores, color='#2980B9', alpha=0.1)
# # #     ax1.set_title("(A) Optimization Trajectory", loc='left', fontweight='bold')
# # #     ax1.set_xlabel("Steps")
# # #     ax1.set_ylabel("Target Probability")
# # #     ax1.legend()
# # #     ax1.grid(True, alpha=0.3)
    
# # #     # Panel B: PCA Manifold
# # #     ax2 = fig.add_subplot(gs[0, 1])
# # #     pca = PCA(n_components=2)
# # #     z_pca = pca.fit_transform(z_vectors)
    
# # #     # Plot path
# # #     sc = ax2.scatter(z_pca[:, 0], z_pca[:, 1], c=steps, cmap='viridis', 
# # #                      s=60, edgecolors='k', linewidth=0.5, alpha=0.8)
# # #     ax2.plot(z_pca[:, 0], z_pca[:, 1], c='black', alpha=0.2)
    
# # #     # Start/End annotations
# # #     ax2.text(z_pca[0,0], z_pca[0,1], "START", fontweight='bold', ha='right')
# # #     ax2.text(z_pca[-1,0], z_pca[-1,1], "GOAL", fontweight='bold', color='#27AE60')
    
# # #     cbar = plt.colorbar(sc, ax=ax2)
# # #     cbar.set_label('Step')
# # #     ax2.set_title("(B) Latent Space Navigation", loc='left', fontweight='bold')
# # #     ax2.set_xlabel("PC 1")
# # #     ax2.set_ylabel("PC 2")
    
# # #     plt.tight_layout()
# # #     plot_path = os.path.join(OUTPUT_DIR, "Figure_Optimization_Plots.png")
# # #     plt.savefig(plot_path, dpi=300)
# # #     print(f"✅ Saved Plots: {plot_path}")
    
# # #     # ==========================================
# # #     # FIGURE 2: MOLECULAR EVOLUTION (SVGs)
# # #     # ==========================================
# # #     # Select 5 evenly spaced molecules
# # #     indices = np.linspace(0, len(history) - 1, 5, dtype=int)
    
# # #     print("✅ Saving Molecular Structures (SVG):")
# # #     for i, idx in enumerate(indices):
# # #         smi = history[idx]['smiles']
# # #         score = history[idx]['score']
# # #         mol = Chem.MolFromSmiles(smi)
        
# # #         fname = f"Step_{history[idx]['step']:03d}_Score_{int(score*100)}.svg"
# # #         fpath = os.path.join(OUTPUT_DIR, fname)
        
# # #         save_molecule_svg(mol, fpath, score, history[idx]['step'])
# # #         print(f"   -> {fname}")

# # #     print("\n🚀 DONE! Download the 'Paper_Assets' folder to your local machine.")
# # #     print("   Use the PNG for the data panels and the SVGs for the structure panel in LaTeX.")

# # # if __name__ == "__main__":
# # #     if not os.path.exists("optimization_trajectory.pkl"):
# # #         print("❌ Error: optimization_trajectory.pkl not found!")
# # #     else:
# # #         traj_data = load_trajectory()
# # #         generate_paper_figures(traj_data)

# # import pickle
# # import numpy as np
# # from rdkit import Chem
# # from rdkit.Chem.Draw import rdMolDraw2D
# # import html

# # # --- CONFIG ---
# # INPUT_FILE = "optimization_trajectory.pkl"
# # OUTPUT_FILE = "trajectory_view.html"

# # def load_data():
# #     with open(INPUT_FILE, "rb") as f:
# #         return pickle.load(f)

# # def mol_to_svg(smi, width=300, height=300):
# #     if not smi: return ""
# #     mol = Chem.MolFromSmiles(smi)
# #     if not mol: return "Invalid Molecule"
    
# #     drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
# #     opts = drawer.drawOptions()
# #     opts.clearBackground = False
    
# #     # Make it look scientific (black and white with clear atoms)
# #     drawer.DrawMolecule(mol)
# #     drawer.FinishDrawing()
# #     svg = drawer.GetDrawingText()
    
# #     # Clean up SVG for embedding (remove namespace clutter if needed, but usually fine)
# #     return svg

# # def generate_html(history):
# #     # Select 5 evenly spaced steps
# #     indices = np.linspace(0, len(history) - 1, 5, dtype=int)
    
# #     # Start HTML
# #     html_content = f"""
# #     <!DOCTYPE html>
# #     <html lang="en">
# #     <head>
# #         <meta charset="UTF-8">
# #         <style>
# #             body {{
# #                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
# #                 background-color: #f8f9fa;
# #                 padding: 40px;
# #                 text-align: center;
# #             }}
# #             h1 {{ color: #2c3e50; margin-bottom: 10px; }}
# #             p.sub {{ color: #7f8c8d; margin-bottom: 40px; font-size: 1.1em; }}
            
# #             .container {{
# #                 display: flex;
# #                 flex-direction: row;
# #                 justify-content: center;
# #                 align-items: center;
# #                 gap: 20px;
# #                 flex-wrap: wrap;
# #             }}
            
# #             .card {{
# #                 background: white;
# #                 border-radius: 12px;
# #                 box-shadow: 0 4px 15px rgba(0,0,0,0.1);
# #                 padding: 20px;
# #                 width: 320px;
# #                 transition: transform 0.2s;
# #                 border-top: 5px solid #ccc;
# #             }}
            
# #             .card:hover {{ transform: translateY(-5px); }}
            
# #             /* Dynamic border colors based on score */
# #             .score-low {{ border-color: #e74c3c; }}    /* Red */
# #             .score-mid {{ border-color: #f1c40f; }}    /* Yellow */
# #             .score-high {{ border-color: #2ecc71; }}   /* Green */
            
# #             .step-label {{
# #                 font-size: 0.9em;
# #                 text-transform: uppercase;
# #                 letter-spacing: 1px;
# #                 color: #95a5a6;
# #                 margin-bottom: 5px;
# #             }}
            
# #             .score-val {{
# #                 font-size: 1.8em;
# #                 font-weight: bold;
# #                 color: #2c3e50;
# #             }}
            
# #             .arrow {{
# #                 font-size: 2em;
# #                 color: #bdc3c7;
# #                 font-weight: bold;
# #             }}

# #             .svg-container {{
# #                 margin: 15px 0;
# #             }}
# #         </style>
# #     </head>
# #     <body>

# #         <h1>De Novo Optimization Trajectory</h1>
# #         <p class="sub">Evolution of molecular structure via Gradient Ascent in Latent Space</p>

# #         <div class="container">
# #     """

# #     # Loop through selected steps
# #     for i, idx in enumerate(indices):
# #         data = history[idx]
# #         step = data['step']
# #         score = data['score']
# #         smiles = data['smiles']
        
# #         # Color coding class
# #         if score < 0.4: color_class = "score-low"
# #         elif score < 0.8: color_class = "score-mid"
# #         else: color_class = "score-high"
        
# #         # SVG
# #         svg_image = mol_to_svg(smiles)
        
# #         # Card HTML
# #         html_content += f"""
# #             <div class="card {color_class}">
# #                 <div class="step-label">Step {step}</div>
# #                 <div class="score-val">{score*100:.1f}%</div>
# #                 <div class="svg-container">{svg_image}</div>
# #                 <div style="font-size:0.8em; color:#777; word-break:break-all;">{smiles[:30]}...</div>
# #             </div>
# #         """
        
# #         # Arrow (if not last)
# #         if i < len(indices) - 1:
# #             html_content += '<div class="arrow">➔</div>'

# #     # Close HTML
# #     html_content += """
# #         </div>
# #     </body>
# #     </html>
# #     """

# #     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
# #         f.write(html_content)
    
# #     print(f"✅ Generated HTML view: {OUTPUT_FILE}")

# # if __name__ == "__main__":
# #     try:
# #         data = load_data()
# #         generate_html(data)
# #     except FileNotFoundError:
# #         print("❌ optimization_trajectory.pkl not found! Please run discovery.py first.")

# import pickle
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem.Draw import rdMolDraw2D
# from xhtml2pdf import pisa  # Library for HTML -> PDF
# import os

# # --- CONFIG ---
# INPUT_FILE = "optimization_trajectory.pkl"
# HTML_FILE = "smart_trajectory.html"
# PDF_FILE = "smart_trajectory.pdf"

# def load_data():
#     with open(INPUT_FILE, "rb") as f:
#         return pickle.load(f)

# def mol_to_svg_text(smi):
#     """Generates a clean SVG string for the molecule"""
#     if not smi: return ""
#     mol = Chem.MolFromSmiles(smi)
#     if not mol: return "Invalid"
    
#     drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
#     opts = drawer.drawOptions()
#     opts.clearBackground = False
    
#     drawer.DrawMolecule(mol)
#     drawer.FinishDrawing()
#     return drawer.GetDrawingText()

# def generate_smart_outputs(history):
#     # 1. SMART CROP: Find where we first hit >99%
#     saturation_idx = len(history) - 1
#     for i, data in enumerate(history):
#         if data['score'] >= 0.99:
#             saturation_idx = i
#             break
            
#     print(f"📉 Detected saturation at Step {history[saturation_idx]['step']}.")
#     print(f"   -> Zooming visuals into range [0 - {history[saturation_idx]['step']}]")
    
#     # 2. Select 5 evenly spaced frames from START to SATURATION
#     # This ensures we see the "climb" in detail
#     indices = np.linspace(0, saturation_idx, 5, dtype=int)
    
#     # 3. Build HTML content
#     cards_html = ""
#     for i, idx in enumerate(indices):
#         data = history[idx]
#         step = data['step']
#         score = data['score']
#         smiles = data['smiles']
        
#         # Color logic
#         border_color = "#e74c3c" # Red
#         if score > 0.4: border_color = "#f1c40f" # Yellow
#         if score > 0.8: border_color = "#2ecc71" # Green
        
#         svg = mol_to_svg_text(smiles)
        
#         arrow = '<div class="arrow">➔</div>' if i < len(indices)-1 else ''
        
#         cards_html += f"""
#         <div class="card-wrapper">
#             <div class="card" style="border-top: 5px solid {border_color};">
#                 <div class="step">Step {step}</div>
#                 <div class="score">{score*100:.1f}%</div>
#                 <div class="mol">{svg}</div>
#                 <div class="smi">{smiles[:25]}...</div>
#             </div>
#             {arrow}
#         </div>
#         """

#     full_html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <style>
#             @page {{ size: landscape; margin: 1cm; }}
#             body {{ font-family: Helvetica, sans-serif; text-align: center; padding: 20px; }}
#             h1 {{ color: #2c3e50; }}
#             .container {{ 
#                 display: flex; 
#                 flex-direction: row; 
#                 justify-content: center; 
#                 align-items: center; 
#                 width: 100%;
#             }}
#             .card-wrapper {{ display: inline-block; vertical-align: middle; }}
#             .card {{
#                 background: #fff;
#                 width: 220px; 
#                 padding: 15px;
#                 margin: 10px;
#                 border-radius: 8px;
#                 box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#                 display: inline-block;
#             }}
#             .step {{ color: #7f8c8d; font-size: 10pt; text-transform: uppercase; }}
#             .score {{ color: #2c3e50; font-size: 18pt; font-weight: bold; margin: 5px 0; }}
#             .arrow {{ font-size: 24pt; color: #bdc3c7; display: inline-block; margin: 0 10px; }}
#             .smi {{ font-family: monospace; font-size: 8pt; color: #95a5a6; word-wrap: break-word; }}
#         </style>
#     </head>
#     <body>
#         <h1>Optimization Trajectory (Detailed View)</h1>
#         <p>Active learning phase from initialization to convergence (Step 0 - {history[saturation_idx]['step']})</p>
#         <br>
#         <div class="container">
#             {cards_html}
#         </div>
#     </body>
#     </html>
#     """
    
#     # 4. Save HTML
#     with open(HTML_FILE, "w", encoding="utf-8") as f:
#         f.write(full_html)
#     print(f"✅ HTML saved: {HTML_FILE}")

#     # 5. Convert to PDF
#     try:
#         with open(PDF_FILE, "wb") as pdf_out:
#             pisa_status = pisa.CreatePDF(full_html, dest=pdf_out)
            
#         if not pisa_status.err:
#             print(f"✅ PDF saved: {PDF_FILE}")
#         else:
#             print("❌ PDF generation failed.")
#     except Exception as e:
#         print(f"⚠️  PDF Error: {e}")
#         print("   (You can still open the HTML file and 'Print to PDF' manually)")

# if __name__ == "__main__":
#     if os.path.exists(INPUT_FILE):
#         data = load_data()
#         generate_smart_outputs(data)
#     else:
#         print("❌ Data file not found.")

import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from xhtml2pdf import pisa
import os

# --- CONFIG ---
INPUT_FILE = "optimization_trajectory.pkl"
HTML_FILE = "focused_trajectory.html"
PDF_FILE = "focused_trajectory.pdf"

def load_data():
    with open(INPUT_FILE, "rb") as f:
        return pickle.load(f)

def mol_to_svg_text(smi):
    if not smi: return ""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return "Invalid Molecule"
    
    # Draw slightly smaller to ensure it fits inside borders
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.05 # Add internal padding so atoms don't touch edges
    
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def generate_focused_outputs(history):
    # 1. FIND THE MOMENT OF SUCCESS
    success_idx = -1
    for i, data in enumerate(history):
        if data['score'] >= 0.99:
            success_idx = i
            break
            
    if success_idx == -1:
        success_idx = len(history) - 1 # Use last step if never hits 100%
        
    print(f"🎯 Success detected at Step {history[success_idx]['step']}")
    
    # 2. SELECT THE "TRANSITION WINDOW" (Last 5 steps ending at success)
    # If success is at index 24, we want indices: 20, 21, 22, 23, 24
    start_idx = max(0, success_idx - 4)
    indices = range(start_idx, success_idx + 1)
    
    print(f"   -> Zooming in on Steps: {[history[i]['step'] for i in indices]}")

    # 3. BUILD HTML
    cards_html = ""
    for i, idx in enumerate(indices):
        data = history[idx]
        step = data['step']
        score = data['score']
        smiles = data['smiles']
        
        # Color Coding
        border_color = "#e74c3c" # Red (Fail)
        if score > 0.1: border_color = "#f39c12" # Orange (Progress)
        if score > 0.5: border_color = "#f1c40f" # Yellow (Good)
        if score > 0.9: border_color = "#2ecc71" # Green (Success)
        
        svg = mol_to_svg_text(smiles)
        
        # Arrow logic
        arrow_html = '<div class="arrow">➔</div>' if i < len(indices)-1 else ''
        
        cards_html += f"""
        <div class="card-wrapper">
            <div class="card" style="border-top: 6px solid {border_color};">
                <div class="step">Step {step}</div>
                <div class="score">{score*100:.1f}%</div>
                <div class="mol-container">{svg}</div>
                <div class="smi" title="{smiles}">{smiles[:20]}...</div>
            </div>
            {arrow_html}
        </div>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @page {{ size: landscape; margin: 0.5cm; }}
            body {{ 
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
                text-align: center; 
                padding: 20px;
                background-color: #ffffff;
            }}
            h1 {{ color: #2c3e50; margin-bottom: 5px; }}
            p.sub {{ color: #7f8c8d; font-size: 12px; margin-bottom: 30px; }}
            
            .container {{ 
                display: flex; 
                flex-direction: row; 
                justify-content: center; 
                align-items: center; 
                width: 100%;
                flex-wrap: nowrap; /* Forces them to stay in one line */
            }}
            
            .card-wrapper {{ 
                display: inline-block; 
                vertical-align: middle; 
                margin: 0 5px;
            }}
            
            .card {{
                background: #fff;
                width: 200px; /* Fixed width to prevent overflow */
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            
            .step {{ 
                color: #95a5a6; 
                font-size: 10px; 
                font-weight: bold; 
                text-transform: uppercase; 
                letter-spacing: 1px;
            }}
            
            .score {{ 
                color: #2c3e50; 
                font-size: 24px; 
                font-weight: 800; 
                margin: 5px 0 10px 0;
            }}
            
            /* Important: Force SVG to fit container */
            .mol-container svg {{
                width: 100%;
                height: auto;
                max-height: 180px;
            }}
            
            .arrow {{ 
                font-size: 24px; 
                color: #dfe6e9; 
                display: inline-block; 
                margin: 0 8px;
                font-weight: bold;
            }}
            
            .smi {{ 
                margin-top: 10px;
                font-family: 'Consolas', monospace; 
                font-size: 9px; 
                color: #bdc3c7; 
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
                max-width: 100%;
            }}
        </style>
    </head>
    <body>
        <h1>Critical Optimization Phase</h1>
        <p class="sub">Frame-by-frame visualization of the final convergence (Steps {[history[i]['step'] for i in indices]})</p>
        
        <div class="container">
            {cards_html}
        </div>
    </body>
    </html>
    """
    
    # 4. Save HTML
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"✅ View saved: {HTML_FILE} (Open this in browser!)")

    # 5. Convert to PDF (Optional convenience)
    try:
        with open(PDF_FILE, "wb") as pdf_out:
            pisa_status = pisa.CreatePDF(full_html, dest=pdf_out)
        if not pisa_status.err:
            print(f"✅ PDF saved: {PDF_FILE}")
    except Exception as e:
        print(f"Note: PDF generation skipped ({e}). Use the HTML.")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        data = load_data()
        generate_focused_outputs(data)
    else:
        print("❌ Data file not found. Run discovery.py first.")