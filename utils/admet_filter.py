"""
ADMET filter using admet_ai (https://github.com/swansonk14/admet_ai).

Reads a CSV of generated SMILES, runs ADMET predictions, then evaluates
each molecule against the hard and soft thresholds defined in
configs/PropertyConfig.yaml.

A molecule PASSES if:
  1. It passes ALL hard filters.
  2. It passes at least SOFT_PASS_REQUIRED (default=5) of the 9 soft filters.

Usage:
    python utils/admet_filter.py --input test.csv [--smiles_col smiles]
                                 [--output results.csv] [--soft_required 5]
                                 [--config configs/PropertyConfig.yaml]
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Mapping from PropertyConfig.yaml property names → admet_ai column names
# and any scale-adjusted thresholds.
#
# For probability-based properties (0-1), the config threshold applies as-is.
# For continuous-valued properties, a biologically appropriate threshold is
# used instead and noted below.
# ---------------------------------------------------------------------------

# (admet_ai_column, adjusted_threshold, direction)
# direction: "low"  → value must be < threshold to PASS
#            "high" → value must be > threshold to PASS
ADMET_COLUMN_MAP = {
    # --- Hard filters ---
    # hERG inhibition probability (0-1); target=low → pass if < 0.3
    "hERG_inhibition": ("hERG", 0.3, "low"),

    # CYP3A4 inhibition probability (0-1); target=low → pass if < 0.4
    "CYP3A4_inhibition": ("CYP3A4_Veith", 0.4, "low"),

    # --- Soft filters ---
    # Blood-brain barrier penetration probability (0-1); target=low (non-CNS)
    "BBBP": ("BBB_Martins", 0.3, "low"),

    # Caco-2 permeability [log(cm/s)]; target=high.
    # -5.15 log(cm/s) ≈ 7×10⁻⁶ cm/s is a standard high-permeability cutoff.
    # (Config threshold 0.6 was probability-based; this replaces it.)
    "Caco2_permeability": ("Caco2_Wang", -5.15, "high"),

    # Human Liver Microsome (HLM) stability via microsomal clearance [mL/min/kg].
    # Target=high stability → low clearance.  < 30 mL/min/kg = stable.
    # (Config threshold 0.6 was probability-based; this replaces it.)
    "HLM_stability": ("Clearance_Microsome_AZ", 30.0, "low"),

    # Rat Liver Microsome (RLM) stability: no direct RLM column in admet_ai;
    # hepatocyte clearance [mL/min/10⁶ cells] used as proxy. < 30 = stable.
    "RLM_stability": ("Clearance_Hepatocyte_AZ", 30.0, "low"),

    # P-glycoprotein substrate probability (0-1); target=low
    "P-gp_substrate": ("Pgp_Broccatelli", 0.4, "low"),

    # CYP inhibition probabilities (0-1); target=low
    "CYP1A2_inhibition": ("CYP1A2_Veith", 0.4, "low"),
    "CYP2C9_inhibition": ("CYP2C9_Veith", 0.4, "low"),
    "CYP2C19_inhibition": ("CYP2C19_Veith", 0.4, "low"),
    "CYP2D6_inhibition": ("CYP2D6_Veith", 0.4, "low"),
}

HARD_FILTER_PROPS = ["hERG_inhibition", "CYP3A4_inhibition"]
SOFT_FILTER_PROPS = [
    "BBBP", "Caco2_permeability", "HLM_stability", "RLM_stability",
    "P-gp_substrate", "CYP1A2_inhibition", "CYP2C9_inhibition",
    "CYP2C19_inhibition", "CYP2D6_inhibition",
]


def load_property_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def passes_filter(value: float, threshold: float, direction: str) -> bool:
    if direction == "low":
        return value < threshold
    return value > threshold


def evaluate_molecules(
    df_preds: pd.DataFrame,
    soft_required: int = 5,
) -> pd.DataFrame:
    """
    Add per-property pass/fail columns and overall pass columns to df_preds.

    Returns the augmented dataframe.
    """
    results = df_preds.copy()

    for prop, (col, threshold, direction) in ADMET_COLUMN_MAP.items():
        if col not in results.columns:
            print(f"  [WARN] Column '{col}' not found in predictions (skipping '{prop}')")
            results[f"pass_{prop}"] = None
            continue
        results[f"pass_{prop}"] = results[col].apply(
            lambda v: passes_filter(v, threshold, direction)
        )

    # Hard filter: must pass ALL
    hard_cols = [f"pass_{p}" for p in HARD_FILTER_PROPS if f"pass_{p}" in results]
    results["pass_hard_filters"] = results[hard_cols].all(axis=1)

    # Soft filter: must pass >= soft_required out of 9
    soft_cols = [
        f"pass_{p}" for p in SOFT_FILTER_PROPS
        if f"pass_{p}" in results and results[f"pass_{p}"].notna().all()
    ]
    results["soft_passes"] = results[soft_cols].sum(axis=1)
    results["pass_soft_filters"] = results["soft_passes"] >= soft_required

    # Overall pass: hard AND soft
    results["pass_all"] = results["pass_hard_filters"] & results["pass_soft_filters"]

    return results


def print_summary(results: pd.DataFrame, soft_required: int) -> None:
    n_total = len(results)
    n_hard = results["pass_hard_filters"].sum()
    n_soft = results["pass_soft_filters"].sum()
    n_all = results["pass_all"].sum()

    print("\n" + "=" * 60)
    print("ADMET FILTER SUMMARY")
    print("=" * 60)
    print(f"  Total molecules evaluated : {n_total}")
    print(f"  Pass hard filters (all 2) : {n_hard:>5}  ({100*n_hard/n_total:.1f}%)")
    print(f"  Pass soft filters (≥{soft_required}/9)   : {n_soft:>5}  ({100*n_soft/n_total:.1f}%)")
    print(f"  Pass ALL filters          : {n_all:>5}  ({100*n_all/n_total:.1f}%)")

    print("\nPer-property pass rates:")
    all_props = HARD_FILTER_PROPS + SOFT_FILTER_PROPS
    for prop in all_props:
        col = f"pass_{prop}"
        if col not in results or results[col].isna().all():
            print(f"  {'[HARD]' if prop in HARD_FILTER_PROPS else '[SOFT]'} {prop:<28} N/A")
            continue
        tag = "[HARD]" if prop in HARD_FILTER_PROPS else "[SOFT]"
        n = results[col].sum()
        admet_col, threshold, direction = ADMET_COLUMN_MAP[prop]
        print(
            f"  {tag} {prop:<28} {n:>5}/{n_total}  "
            f"({100*n/n_total:5.1f}%)  "
            f"[{admet_col} {direction}<{'' if direction=='low' else '>'}{threshold}]"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ADMET filter using admet_ai")
    parser.add_argument("--input", required=True, help="Input CSV with SMILES")
    parser.add_argument("--smiles_col", default="smiles", help="Name of the SMILES column")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <input>_admet_filtered.csv)"
    )
    parser.add_argument(
        "--soft_required", type=int, default=5,
        help="Number of soft filters a molecule must pass (default: 5)"
    )
    parser.add_argument(
        "--config", default="configs/PropertyConfig.yaml",
        help="Path to PropertyConfig.yaml"
    )
    args = parser.parse_args()

    # --- Load config (used for reference; thresholds handled via ADMET_COLUMN_MAP) ---
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_property_config(str(config_path))
        print(f"Loaded property config: {config_path}")
    else:
        print(f"[WARN] Config not found at {config_path}; using built-in defaults.")

    # --- Load input CSV ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if args.smiles_col not in df.columns:
        print(f"[ERROR] Column '{args.smiles_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    smiles_list = df[args.smiles_col].dropna().tolist()
    print(f"Loaded {len(smiles_list)} SMILES from {input_path}")

    # --- Run admet_ai predictions ---
    print("Running admet_ai predictions (this may take a moment)...")
    from admet_ai import ADMETModel  # imported here to avoid slow load on --help

    model = ADMETModel()
    preds_df = model.predict(smiles=smiles_list)  # returns DataFrame indexed by SMILES

    # Merge original dataframe columns back in
    preds_df = preds_df.reset_index().rename(columns={"index": args.smiles_col})
    merged = df.merge(preds_df, on=args.smiles_col, how="left")

    # --- Evaluate ---
    results = evaluate_molecules(merged, soft_required=args.soft_required)

    # --- Summary ---
    print_summary(results, soft_required=args.soft_required)

    # --- Save output ---
    output_path = args.output or str(input_path.with_name(input_path.stem + "_admet_filtered.csv"))
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
