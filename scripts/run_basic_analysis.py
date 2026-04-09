from pathlib import Path
import helper_functions as helper
import pandas as pd

# ===== CONFIG =====
BASE_DIR = Path(r"C:\Users\20223231\OneDrive - TU Eindhoven\Documents\Word documenten\Master Jaar 1\Q4 - Stage\Github EWS-Pipeline-FEM")

MODEL_DIR = BASE_DIR / "density_models" / "density_100" / "output"

STEP_MIN = 0
STEP_MAX = 20

# ===== RUN =====
print(">>> Starting basic analysis")

# Build summary table
df = helper.build_summary_table(MODEL_DIR, STEP_MIN, STEP_MAX)

print(f"Processed {len(df)} timesteps")

# Save CSV
output_file = MODEL_DIR.parent / "summary_statistics.csv"
df.to_csv(output_file, index=False)

print(f"Saved to: {output_file}")