"""
Main analysis pipeline for VTK-based tissue mechanics data.

Processes simulation outputs to compute statistics,
and generates visualizations and summary tables.

Outputs:
  - figures (*.png) in figures/ subfolders
  - scalar summary table: figures/vm_trend.csv
"""

from pathlib import Path
import pandas as pd
from . import helper_functions as helper
from . import visualization as viz
from . import config
from .mesh_stability_analysis import run_mesh_analysis

if __name__ == "__main__":
    """
    Run the full analysis pipeline:
    - Extract surface nodes and landmarks
    - Compute and save displacement data
    - Generate summary statistics and visualizations for each model
    - Compare stress and displacement metrics across models
    """

    # Set up input and outputs paths
    vtk_dir = config.MODELS_TO_COMPARE["Heterogeneity_and_lobular"]
    feb_path = vtk_dir.parent / "gradient_lobules.feb"
    fig_dir = config.BASE_DIR / f"figures/{vtk_dir.parent.name}"
    all_models_fig_dir = config.BASE_DIR / "figures/comparison_all_models"

    # Ensure output dir exists
    all_models_fig_dir.mkdir(parents=True, exist_ok=True)

    # Extract surface nodes from FEB file
    print("\n[1/13] Extracting surface nodes from FEB file")
    surface_nodes = helper.get_surface_nodes(feb_path)
    print(f"Found {len(surface_nodes)} surface nodes")

    # Extract landmarks based on target Y coordinate
    print(f"\n[2/13] Extracting landmarks (target Y = {config.TARGET_Y})")
    landmarks = helper.extract_surface_nodes(feb_path,
                                             target_y=config.TARGET_Y)
    if not landmarks:
        print("No landmarks found!")

    # Extract displacements at landmarks for specified step
    print(
        f"\n[3/13] Extracting displacements at landmarks (step {config.STEP})")
    df_landmarks = helper.extract_landmark_displacements(
        config.MODELS_TO_COMPARE,
        landmarks,
        surface_nodes,
        config.STEP,
        relative_to_baseline=True
    )

    if df_landmarks.empty:
        print("No displacement data extracted!")

    # Loop over models and generate outputs
    first_model = list(config.MODELS_TO_COMPARE.values())[0]
    times = helper.get_common_times(first_model, config.STEP_MIN,
                                    config.STEP_MAX)

    for model in config.MODELS_TO_COMPARE:
        fig_dir = config.BASE_DIR / f"figures/{config.MODELS_TO_COMPARE[model].parent.name}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Build summary statistics table
        print("\n[4/13] Building summary table")
        df = helper.build_summary_table(config.MODELS_TO_COMPARE[model],
                                        config.STEP_MIN, config.STEP_MAX)
        print(f"processed {len(df)} time steps")

        # Plot stress evolution over time
        print("\n[5/13] Plotting stress evolution")
        viz.plot_stress_evolution(df, fig_dir, times, model,
                                  config.MODEL_LABELS)

        # Plot tissue stress comparison
        print("\n[6/13] Plotting tissue comparison")
        viz.plot_tissue_comparison(df, config.TISSUE_LABELS, fig_dir, times,
                                   model,
                                   config.MODEL_LABELS)
        del df  # Free memory

        # Plot surface displacement evolution
        print("\n[7/13] Plotting surface displacement evolution...")
        viz.plot_surface_displacement_evolution(
            config.MODELS_TO_COMPARE[model], surface_nodes,
            config.STEP_MIN, config.STEP_MAX,
            fig_dir, times, config.MODEL_LABELS, model)

        # Save landmark displacement data to CSV
        print("\n[8/13] Saving landmark displacement data")
        output_csv = all_models_fig_dir / f"landmark_displacements_step{config.STEP}.csv"
        df_landmarks.to_csv(output_csv, index=False)

        # Create specific markers visualizations
        print("\n[9/13] Creating markers visualizations")
        viz.plot_landmark_spatial(config.MODELS_TO_COMPARE[model], landmarks,
                                  surface_nodes, config.STEP,
                                  fig_dir, config.MODEL_LABELS, model)

        viz.plot_landmark_comparison(df_landmarks, all_models_fig_dir,
                                     config.MODEL_LABELS, config.MODEL_COLORS)

    del landmarks  # Free memory

    # Analyse stress and displacement metrics for all models
    print("\n[10/13] Analyzing stress in all models")
    stress_results = {}
    for model_name, vtk_dir in config.MODELS_TO_COMPARE.items():
        print(f"Processing {model_name}...")
        summary_csv = vtk_dir.parent / "summary_statistics.csv"
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            print(f"Loaded existing summary")
        else:
            print(f"Building summary table...")
            df = helper.build_summary_table(vtk_dir, config.STEP_MIN,
                                            config.STEP_MAX)
            df.to_csv(summary_csv, index=False)
            print(f"Saved summary table")

        stress_results[model_name] = {
            "stress": helper.extract_peak_stress_by_tissue(df),
            "displacement": helper.extract_displacement_metrics(
                vtk_dir, feb_path, config.STEP_MIN, config.STEP_MAX)
        }

    # Create spatial displacement comparison maps
    print(
        f"\n[11/13] Creating spatial displacement maps for step {config.STEP}")
    viz.plot_spatial_displacement_comparison(config.MODELS_TO_COMPARE,
                                             surface_nodes,
                                             config.STEP,
                                             all_models_fig_dir,
                                             config.MODEL_LABELS)

    # Create scalar comparison plots
    print("\n[12/13] Creating scalar comparison plots")
    viz.plot_stress_comparison(stress_results, all_models_fig_dir,
                               config.MODEL_LABELS, config.MODEL_COLORS,
                               config.TISSUE_LABELS,
                               config.LITERATURE_REFS)

    # Create tissue stress evolution comparison plots
    print("\n[13/13] Detailed tissue stress comparison")
    viz.plot_stress_evolution_comparison(config.MODELS_TO_COMPARE,
                                         all_models_fig_dir,
                                         config.STEP_MIN, config.STEP_MAX,
                                         config.MODEL_LABELS,
                                         config.MODEL_COLORS, times)

    print("\nAnalysis complete")
    # print(f"All outputs saved in {fig_dir.resolve()}")


# Additional analysis: mesh stability and quality metrics
if __name__ == "__main__":
    run_mesh_analysis()