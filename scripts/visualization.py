import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import meshio
import re
import helper_functions as helper


def plot_stress_evolution(df: pd.DataFrame, output_dir: Path,
                          times: np.ndarray, model_key: str,
                          model_labels: Dict):
    """
    Plot the evolution of von Mises stress statistics (mean, median, max) over time for a single model.

    Args:
        df: DataFrame with stress statistics per timestep.
        output_dir: Directory to save the plot.
        times: Array of time values.
        model_key: Model identifier.
        model_labels: Mapping of model keys to display names.
    """
    # Ensure times match dataframe length
    times = times[:len(df)]
    model_name = model_labels.get(model_key, model_key)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Mean and Median Stress
    ax1.plot(times, df["vm_mean"], "o-", label="Mean", linewidth=2)
    ax1.plot(times, df["vm_median"], "s-", label="Median", linewidth=2)
    ax1.fill_between(times, df["vm_mean"] - df["vm_std"],
                     df["vm_mean"] + df["vm_std"],
                     alpha=0.3, label="±1 SD")
    ax1.set_ylabel("Von Mises Stress (Pa)")
    ax1.set_title(f"Von Mises Stress evolution {model_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Maximum Stress
    ax2.plot(times, df["vm_max"], "r^-", label="Maximum", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Von Mises Stress (Pa)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "stress_evolution.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_tissue_comparison(df: pd.DataFrame, tissue_labels: Dict,
                           output_dir: Path, times: np.ndarray,
                           model_key: str, model_labels: Dict):
    """
    Plot mean von Mises stress for each tissue type over time.

    Args:
        df: DataFrame with stress statistics per tissue.
        tissue_labels: Mapping of part numbers to tissue names.
        output_dir: Directory to save the plot.
        times: Array of time values.
        model_key: Model identifier.
        model_labels: Mapping of model keys to display names.
    """
    # Find columns for each tissue type
    parts_cols = [col for col in df.columns if
                  col.startswith("part") and col.endswith("_vm_mean")]

    if len(parts_cols) < 2:
        print("Not enough parts to plot tissue comparison")
        return

    times = times[:len(df)]
    model_name = model_labels.get(model_key, model_key)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Plot each tissue's stress comparison
    for col in parts_cols:
        part_num = int(col.replace("part", "").split("_")[0])
        part_name = tissue_labels[part_num]
        ax.plot(times, df[col], "o-", label=part_name, linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Von Mises Stress (Pa)")
    ax.set_title(f"Stress per Tissue type {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tissue_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_surface_displacement_evolution(vtk_dir: Path, surface_nodes: set,
                                        step_min: int, step_max: int,
                                        output_dir: Path, times: np.ndarray,
                                        model_labels: Dict, model_key: str):
    """
    Plot the evolution of surface node displacement statistics over time.

    Args:
        vtk_dir: Directory containing VTK files.
        surface_nodes: Set of node IDs on the surface.
        step_min: First timestep to include.
        step_max: Last timestep (exclusive).
        output_dir: Directory to save plots and CSV.
        times: Array of time values.
        model_labels: Mapping of model keys to display names.
        model_key: Model identifier.
    """
    # Gather VTK files in step range
    vtks = []
    for x in vtk_dir.glob("*.vtk"):
        m = re.search(r"\.(\d+)\.vtk", x.name)
        if m:
            step = int(m.group(1))
            if step_min <= step < step_max:
                vtks.append((step, x))
    vtks = sorted(vtks)

    stats = []
    for step_num, vtkp_path in vtks:
        mesh = meshio.read(vtkp_path)
        U = mesh.point_data.get("displacement")
        if U is None:
            continue

        U_mag = np.linalg.norm(U, axis=1)
        # Convert node IDs to 0-based indices and filter valid ones
        surface_indices = np.array([nid - 1 for nid in surface_nodes
                                    if 0 <= nid - 1 < len(mesh.points)])
        surf_disp_mag = U_mag[surface_indices] * 1000

        stats.append({
            "step": step_num,
            "mean": surf_disp_mag.mean(),
            "median": np.median(surf_disp_mag),
            "max": surf_disp_mag.max(),
            "std": surf_disp_mag.std()
        })

    df_disp = pd.DataFrame(stats)

    if len(df_disp) != len(times):
        print(
            f"Warning: Mismatch - {len(df_disp)} VTK files but {len(times)} time points")
    # Truncate times to match data
    times = times[:len(df_disp)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot mean and median displacement with ±SD shading
    ax1.plot(times, df_disp["mean"], "o-", label="Mean", linewidth=2)
    ax1.plot(times, df_disp["median"], "s-", label="Median",
             linewidth=2)
    ax1.fill_between(times,
                     df_disp["mean"] - df_disp["std"],
                     df_disp["mean"] + df_disp["std"],
                     alpha=0.3, label="+- SD")
    ax1.set_ylabel("Displacement (mm)")
    model_name = model_labels.get(model_key, model_key)
    ax1.set_title(f"Surface displacement {model_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot max displacement
    ax2.plot(times, df_disp["max"], "^-", label="Max", linewidth=2,
             color="red")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Displacement (mm)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "surface_displacement_over_time.png", dpi=300,
                bbox_inches="tight")
    plt.close()

    # Save stats to CSV for further analysis
    df_disp.to_csv(output_dir / "surface_displacement_stats.csv", index=False)


def plot_landmark_spatial(vtk_dir: Path, landmarks: dict,
                          surface_nodes: set, step: int,
                          output_dir: Path, model_labels: Dict,
                          model_key: str):
    """
    Visualize spatial location and displacement of landmarks on the breast surface (top view).

    Args:
        vtk_dir: Directory containing VTK files.
        landmarks: Dict of landmark info (name -> {coords, node_id, ...}).
        surface_nodes: Set of node IDs on the surface.
        step: Timestep to visualize.
        output_dir: Directory to save figure.
        model_labels: Mapping of model keys to display names.
        model_key: Model identifier.
    """
    model_name = model_labels.get(model_key, model_key)

    vtk_files = list(vtk_dir.glob(f"*.{int(step)}.vtk"))
    if not vtk_files:
        print(f"Warning: No VTK file found for {model_name} at step {step}")
        return

    mesh = meshio.read(vtk_files[0])
    U = mesh.point_data.get("displacement")
    if U is None:
        print(f"Warning: No displacement data found for {model_name}")
        return

    # Get surface nodes
    surface_indices = np.array([nid - 1 for nid in surface_nodes
                                if 0 <= nid - 1 < len(mesh.points)])

    surf_coords = mesh.points[surface_indices]
    surf_disp = np.linalg.norm(U[surface_indices], axis=1) * 1000  # mm

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot all surface nodes colored by displacement
    sc = ax.scatter(surf_coords[:, 0], surf_coords[:, 2],
                    # X-Z plane (top view)
                    c=surf_disp, s=5, cmap='viridis', alpha=0.4,
                    vmin=0, vmax=surf_disp.max())

    # Highlight landmarks with distinct colors
    landmark_colors = {
        'nipple': 'red',
        'left': 'cyan',
        'right': 'magenta',
        'superior': 'lime',
        'inferior': 'orange'
    }

    for landmark_name, landmark_info in landmarks.items():
        coords = landmark_info['coords']
        color = landmark_colors.get(landmark_name, 'white')
        ax.scatter(coords[0], coords[2], s=200, c=color,
                   edgecolors='black', linewidths=2,
                   marker='o', zorder=10, label=landmark_name)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.set_title(f"{model_name}", fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Displacement (mm)", rotation=270, labelpad=15)

    # Save figure
    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
    plt.tight_layout()
    plt.savefig(
        output_dir / f"landmark_spatial_top_{safe_model_name}.png",
        dpi=300, bbox_inches='tight')
    plt.close()


def plot_landmark_comparison(df: pd.DataFrame, output_dir: Path,
                             model_labels: Dict, model_colors: Dict):
    """
    Create a bar chart comparing landmark displacements across models.

    Args:
        df: DataFrame with landmark displacement data.
        output_dir: Directory to save the plot.
        model_labels: Mapping of model keys to display names.
        model_colors: Mapping of model keys to color codes.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    landmarks = df['landmark'].unique()
    models = df['model'].unique()
    x = np.arange(len(landmarks))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        # Get displacement for each landmark (assumes one value per landmark/model)
        displacements = [
            model_data[model_data['landmark'] == lm]['displacement_mm'].values[
                0]
            for lm in landmarks
        ]

        offset = (i - len(models) / 2 + 0.5) * width
        color = model_colors.get(model, f'C{i}')
        label = model_labels.get(model, model)

        ax.bar(x + offset, displacements, width, label=label,
               color=color, alpha=0.8)

    ax.set_xlabel("Landmark")
    ax.set_ylabel("Displacement Magnitude (mm)")
    ax.set_title("Landmark Displacement Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(landmarks, rotation=20, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "landmark_comparison.png", dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_spatial_displacement_comparison(model_dirs: dict,
                                         surface_nodes: set,
                                         step: int,
                                         output_dir: Path, model_labels: dict):
    """
    Create spatial displacement maps (top view: X-Z) for each model,
    arranged in a grid for visual comparison.

    Args:
        model_dirs: Dict of model names to VTK directories.
        surface_nodes: Set of node IDs on the surface.
        step: Timestep to visualize.
        output_dir: Directory to save the figure.
        model_labels: Mapping of model keys to display names.
    """
    n_models = len(model_dirs)
    ncols = 3
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    all_disp_data = []

    print(f"Loading displacement data for {n_models} models...")
    for model_name, vtk_dir in model_dirs.items():
        vtk_files = list(vtk_dir.glob(f"*.{int(step)}.vtk"))
        if not vtk_files:
            continue

        mesh = meshio.read(vtk_files[0])
        U = mesh.point_data.get("displacement")
        if U is None:
            continue

        coords = mesh.points
        surface_indices = np.array([nid - 1 for nid in surface_nodes
                                    if 0 <= nid - 1 < len(coords)])

        surf_coords = coords[surface_indices]
        surf_U = U[surface_indices]
        surf_disp_total = np.linalg.norm(surf_U, axis=1) * 1000  # mm

        all_disp_data.append({
            "model": model_name,
            "coords": surf_coords,
            "total": surf_disp_total,
        })
        print(f"Loaded {model_name}")

    if not all_disp_data:
        print("No displacement data found!")
        return

    vmax = max([d["total"].max() for d in all_disp_data])
    print(f"    Displacement range: 0 - {vmax:.2f} mm")

    for idx, data in enumerate(all_disp_data):
        ax = axes[idx]
        model_label = model_labels.get(data["model"], data["model"])
        sc = ax.scatter(data["coords"][:, 0], data["coords"][:, 2],
                        # X-Z plane
                        c=data["total"], s=20, cmap="plasma",
                        vmin=0, vmax=vmax)
        ax.set_xlabel("X (m)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Z (m)", fontsize=11, fontweight='bold')
        ax.set_title(
            f"{model_label}\n{data['total'].mean():.2f}±{data['total'].std():.2f} mm",
            fontsize=12, fontweight='bold')
        ax.set_aspect("equal")

    # Hide unused axes if fewer models than grid slots
    for ax in axes[len(all_disp_data):]:
        ax.axis('off')

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Total Surface Displacement (mm)', fontsize=12,
                 fontweight='bold')

    plt.suptitle(f'Surface Displacement Patterns - Step {step}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_dir / f"spatial_displacement_step{step}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_stress_comparison(results: dict, output_dir: Path, model_labels: Dict,
                           model_colors: Dict, tissue_labels: Dict,
                           literature_refs: Dict):
    """
    Create a bar chart comparing peak stress by tissue type across models.

    Args:
        results: Dict of model results with stress metrics.
        output_dir: Directory to save the plot.
        model_labels: Mapping of model keys to display names.
        model_colors: Mapping of model keys to color codes.
        tissue_labels: Mapping of tissue keys to display names.
        literature_refs: Reference values (not used in plot).
    """
    models = list(results.keys())
    tissue_types = ["Tissue_1", "Tissue_2"]  # Glandular, Adipose

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(tissue_types))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        # Get peak stress for each tissue type
        values = [results[model]["stress"]["max_stress_kPa"].get(t, 0)
                  for t in tissue_types]
        label = model_labels.get(model, model)
        color = model_colors.get(model, None)
        ax.bar(x + i * width, values, width, label=label, alpha=0.8,
               color=color)

    ax.set_xlabel('Tissue Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Stress (kPa)', fontsize=12, fontweight='bold')
    ax.set_title('Peak Stress by Tissue Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(
        [tissue_labels[int(t.split('_')[1])] for t in tissue_types])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True,
              fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_scalars.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_stress_evolution_comparison(model_dirs: dict, output_dir: Path,
                                     step_min: int, step_max: int,
                                     model_labels: Dict,
                                     model_colors: Dict, times: np.ndarray):
    """
    Plot the evolution of mean and peak von Mises stress over time for multiple models.

    Args:
        model_dirs: Dict of model names to VTK directories.
        output_dir: Directory to save plots.
        step_min: First timestep to include.
        step_max: Last timestep (exclusive).
        model_labels: Mapping of model keys to display names.
        model_colors: Mapping of model keys to color codes.
        times: Array of time values.
    """
    # Mean Stress
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    # Peak Stress
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

    for model_name, vtk_dir in model_dirs.items():
        # Use cached summary if available, else build it
        summary_csv = vtk_dir.parent / "summary_statistics.csv"

        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
        else:
            print(f"  Warning: No summary for {model_name}, building now...")
            df = helper.build_summary_table(vtk_dir, step_min, step_max)
            df.to_csv(summary_csv, index=False)

        df = df[(df["step"] >= step_min) & (df["step"] < step_max)]

        model_times = times[:len(df)]
        label = model_labels.get(model_name, model_name)
        color = model_colors.get(model_name, None)

        # Plot mean stress
        ax1.plot(model_times, df["vm_mean"], "o-", label=label,
                 linewidth=2, color=color, alpha=0.8, markersize=4)

        # Plot max stress
        ax2.plot(model_times, df["vm_max"], "^-", label=label,
                 linewidth=2, color=color, alpha=0.8, markersize=4)

    # Configure Figure 1 (Mean Stress)
    ax1.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Mean von Mises Stress (Pa)", fontsize=12,
                   fontweight='bold')
    ax1.set_title("Mean Stress Evolution", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True,
               fontsize=10)

    fig1.tight_layout()
    fig1.savefig(output_dir / "stress_evolution_mean.png",
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Configure Figure 2 (Peak Stress)
    ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Max von Mises Stress (Pa)", fontsize=12, fontweight='bold')
    ax2.set_title("Peak Stress Evolution", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True,
               fontsize=10)

    fig2.tight_layout()
    fig2.savefig(output_dir / "stress_evolution_peak.png",
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
