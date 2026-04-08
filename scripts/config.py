"""
Configuration file for breast biomechanics analysis.

Defines paths, model directories, analysis parameters, tissue/model labels,
color schemes, and literature reference values for use throughout the pipeline.
"""
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(
    "/Users/ryanengels/Documents/limebv-ews_fem_pipeline-e5fe835fd1bf")

# Directories for each model to be compared in the analysis
MODELS_TO_COMPARE = {
    "Homogeneous_baseline": BASE_DIR / "output",
    "Radial_density": BASE_DIR / "heterogeneous_density/output",
    "Radial_coef": BASE_DIR / "heterogeneous_c/output",
    "Lobular_only": BASE_DIR / "full_glandular/output",
    "Heterogeneity_and_lobular": BASE_DIR / "gradient_lobules/output",
    "full_adipose_density": BASE_DIR / "full_homogeneous_density/output",
    "full_glandular_density": BASE_DIR / "full_glandular_density/output",
    "full_density_and_coefs": BASE_DIR / "full_density_and_coefs/output",

}

# ==================== ANALYSIS PARAMETERS ====================
STEP_MIN = 0         # First time step to include
STEP_MAX = 20        # Last time step (exclusive)
STEP = 3             # Time step for landmark tracking
TARGET_Y = 0.039     # Y-coordinate for extracting horizontal plane landmarks

# ==================== LABELS ====================
# Tissue type labels by part ID
TISSUE_LABELS = {
    1: "Glandular",
    2: "Adipose"
}

# Human-readable model labels for plots and tables
MODEL_LABELS = {
    "Homogeneous_baseline": "Baseline\n(Homogeneous)",
    "Radial_density": "Radial Gradient\n(Density only)",
    "Radial_coef": "Radial Gradient\n(coef only)",
    "Lobular_only": "Lobular\n(Glandular only)",
    "Heterogeneity_and_lobular": "Full Model\n(Radial + Lobular)",
    "full_adipose_density": "Adipose-only Density",
    "full_glandular_density": "Glandular-only Density ",
    "full_density_and_coefs": "Complete\nHomogeneous",
}

# Color codes for each model (used in visualizations)
MODEL_COLORS = {
    "Homogeneous_baseline": "#1f77b4",
    "Radial_density": "#ff7f0e",
    "Radial_coef": "#17becf",
    "Lobular_only": "#2ca02c",
    "Heterogeneity_and_lobular": "#d62728",
    "full_adipose_density": "#9467bd",
    "full_glandular_density": "#8c564b",
    "full_density_and_coefs": "#e377c2",
}

# ==================== LITERATURE REFERENCE VALUES ====================
# Reference values from published studies for validation/comparison
LITERATURE_REFS = {
    "Chen_2025_Running_6kmh": {
        "Adipose_max_kPa": 10.56,
        "Glandular_max_kPa": 3.89,
        "displacement_vertical_cm": (4.2, 11.0),
    },
    "Chen_2025_Jumping": {
        "Adipose_max_kPa": 16.24,
        "Glandular_max_kPa": 4.22,
    },
    "General_Literature": {
        "vertical_displacement_cm": (4.2, 11.0),
        "medial_lateral_cm": (1.8, 6.2),
        "anterior_posterior_cm": (2.2, 5.9),
        "vertical_acceleration_g": (2.8, 4.87),
    }
}
