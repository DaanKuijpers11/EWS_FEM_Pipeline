import re
from pathlib import Path
import numpy as np
import pandas as pd
import meshio
import gc
from typing import Dict, List, Tuple


def von_mises_any(sig):
    """
    Calculate von Mises stress from a stress tensor.

    Supports input shapes:
        - (n, 3, 3): full tensor
        - (n, 6): Voigt notation [sxx, syy, szz, sxy, syz, sxz]

    Args:
        sig: Stress tensor array.

    Returns:
        Von Mises stress as a 1D numpy array.
    """
    sig = np.asarray(sig)
    # Handle full tensor (n, 3, 3)
    if sig.ndim == 3 and sig.shape[1:] == (3, 3):
        sxx, syy, szz = sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2]
        sxy = 0.5 * (sig[:, 0, 1] + sig[:, 1, 0])
        sxz = 0.5 * (sig[:, 0, 2] + sig[:, 2, 0])
        syz = 0.5 * (sig[:, 1, 2] + sig[:, 2, 1])
    # Handle Voigt notation (n, 6)
    elif sig.ndim == 2 and sig.shape[1] == 6:
        sxx, syy, szz, sxy, syz, sxz = [sig[:, i] for i in range(6)]
    else:
        raise ValueError(f"Unsupported stress shape: {sig.shape}")

    # standard von Mises formula for 3D stress
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                 + 6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))
    return np.sqrt(np.maximum(vm2, 0.0))


def list_vtks(vtk_dir: Path, step_min: int, step_max: int) -> List[Path]:
    """
    List all VTK files in a directory within a step range, sorted by step number.

    Args:
        vtk_dir: Directory containing VTK files.
        step_min: Minimum step (inclusive).
        step_max: Maximum step (exclusive).

    Returns:
        List of VTK file paths sorted by step.
    """
    vtks = []
    for p in vtk_dir.glob("*.vtk"):
        m = re.search(r"\.(\d+)\.vtk$", p.name)
        if m:
            step_num = int(m.group(1))
            if step_min <= step_num < step_max:
                vtks.append((step_num, p))
    # Sort by step number
    return [x for _, x in sorted(vtks)]


def get_baseline_displacement(vtk_dir: Path, surface_nodes: set):
    """
    Get displacement at step 0 (gravity equilibrium) for baseline subtraction.

    Args:
        vtk_dir: Directory containing VTK files.
        surface_nodes: Set of surface node IDs.

    Returns:
        baseline_disp_total: Displacement magnitudes at each surface node.
        baseline_disp_vertical: Y-displacement at each surface node.
        baseline_mean: Mean displacement magnitude.
    """
    vtks = list(vtk_dir.glob("*.0.vtk"))

    if not vtks:
        print("WARNING: No step 0 VTK file found! Cannot establish baseline.")
        return None, None, None

    mesh_baseline = meshio.read(vtks[0])
    U_baseline = mesh_baseline.point_data.get("displacement")
    if U_baseline is None:
        print("WARNING: No displacement data in baseline VTK!")
        return None, None, None

    # Get surface node indices (convert 1-based node IDs to 0-based array indices)
    surface_indices = np.array([nid - 1 for nid in surface_nodes
                                if 0 <= nid - 1 < len(mesh_baseline.points)])

    # Extract baseline displacements
    baseline_disp_total = np.linalg.norm(U_baseline[surface_indices], axis=1)
    baseline_disp_vertical = U_baseline[surface_indices, 1]  # Y-component
    baseline_mean = np.mean(baseline_disp_total)

    print(
        f"  Baseline (step 0) mean displacement: {baseline_mean * 1000:.2f} mm")

    return baseline_disp_total, baseline_disp_vertical, baseline_mean


def read_step(vtk_path: Path) -> Dict:
    """
    Extract relevant statistics from a single VTK file.

    Args:
        vtk_path: Path to the VTK file.

    Returns:
        Dictionary of statistics for the timestep.
    """
    mesh = meshio.read(vtk_path)
    # Use tetra10 if available, otherwise first cell type
    cell_type = "tetra10" if "tetra10" in mesh.cells_dict else \
        list(mesh.cells_dict)[0]

    # Stress data and compute von Mises
    stress = mesh.cell_data_dict["stress"][cell_type]
    vm = von_mises_any(stress)

    # Relative Volume (J) data
    J = mesh.cell_data_dict.get("relative_volume", {}).get(cell_type, None)
    J = J.ravel() if J is not None else np.ones_like(vm)

    # Part id
    pid = mesh.cell_data_dict.get("part_id", {}).get(cell_type, None)
    pid = pid.ravel().astype(int) if pid is not None else np.full(vm.size, -1)

    # Displacement
    U = mesh.point_data.get("displacement", None)
    U_mag = np.linalg.norm(U, axis=1) if U is not None else None

    # Extract step number from filename
    step = int(re.search(r"\.(\d+)\.vtk$", vtk_path.name).group(1))

    # Statistics per step
    stats = {
        "step": step,
        "filename": vtk_path.name,
        "n_elements": len(vm),
        "vm_mean": float(np.mean(vm)),
        "vm_median": float(np.median(vm)),
        "vm_max": float(np.max(vm)),
        "vm_std": float(np.std(vm)),
        "J_mean": float(np.mean(J)),
        "J_median": float(np.median(J)),
        "J_std": float(np.std(J)),
    }

    # Statistics per tissue type
    unique_parts = np.unique(pid)
    for part_id in unique_parts:
        mask = (pid == part_id)
        part_name = f"part{part_id}"
        stats[f"{part_name}_vm_mean"] = float(np.mean(vm[mask]))
        stats[f"{part_name}_vm_max"] = float(np.max(vm[mask]))
        stats[f"{part_name}_J_mean"] = float(np.mean(J[mask]))
        stats[f"{part_name}_n_elem"] = int(np.sum(mask))

    # Displacement statistics
    if U_mag is not None:
        stats["disp_mean"] = float(np.mean(U_mag))
        stats["disp_max"] = float(np.max(U_mag))

    return stats


def get_surface_nodes(feb_path: Path) -> set:
    """
    Extract surface node IDs from a .feb file.

    Args:
        feb_path: Path to the .feb file.

    Returns:
        Set of surface node IDs (integers).
    """
    start_marker = '<Elements type="tri6" name="skin_part">'
    end_marker = "</Elements>"
    num_re = re.compile(r"\d+")
    unique_nodes = set()
    in_elem = False

    with feb_path.open() as file:
        for line in file:
            s = line.strip()
            if s == start_marker:
                in_elem = True
                continue
            if s == end_marker:
                in_elem = False
                continue
            if in_elem:
                # Try to extract node numbers from XML like lines
                m = re.search(r">([^<]+)<", line)
                if m:
                    for n in num_re.findall(m.group(1)):
                        unique_nodes.add(int(n))
                else:
                    for n in num_re.findall(line):
                        unique_nodes.add(int(n))
    return unique_nodes


def build_summary_table(vtk_dir: Path, step_min: int,
                        step_max: int) -> pd.DataFrame:
    """
    Build a summary table of statistics for all timesteps in a directory.

    Args:
        vtk_dir: Directory containing VTK files.
        step_min: Minimum step (inclusive).
        step_max: Maximum step (exclusive).

    Returns:
        DataFrame with statistics for each timestep.
    """
    vtk_files = list_vtks(vtk_dir, step_min, step_max)
    if not vtk_files:
        raise FileNotFoundError(f"No vtk files found in {vtk_dir}")

    print(f"  Building summary table for {len(vtk_files)} vtk files")
    stats_list = []
    for i, vtk_path in enumerate(vtk_files):
        if (i + 1) % 5 == 0:
            print(f"    Processing {i + 1}/{len(vtk_files)}")
        stats = read_step(vtk_path)
        stats_list.append(stats)
        gc.collect()  # Explicitly free memory after each line

    df = pd.DataFrame(stats_list).sort_values("step").reset_index(drop=True)
    return df


def extract_peak_stress_by_tissue(df: pd.DataFrame) -> Dict:
    """
    Extract peak von Mises stress for each tissue type across all timesteps.

    Args:
        df: DataFrame with per-timestep statistics.

    Returns:
        Dictionary with peak stress, mean stress, and step at peak for each tissue.
    """
    results = {
        "step_at_peak": {},
        "max_stress_kPa": {},
        "mean_stress_kPa": {},
    }

    # Find all columns for tissue von Mises stress
    tissue_cols = [col for col in df.columns if
                   "part" in col and "_vm_" in col]

    for col in tissue_cols:
        if "_vm_max" in col:
            part_num = int(col.replace("part", "").split("_")[0])
            max_val = df[col].max() / 1000  # Convert Pa to kPa
            max_step = df.loc[df[col].idxmax(), "step"]

            mean_col = col.replace("_vm_max", "_vm_mean")
            mean_val = df[
                           mean_col].max() / 1000 if mean_col in df.columns else None

            tissue_name = f"Tissue_{part_num}"
            results["max_stress_kPa"][tissue_name] = max_val
            results["step_at_peak"][tissue_name] = max_step
            if mean_val:
                results["mean_stress_kPa"][tissue_name] = mean_val

    return results


def extract_displacement_metrics(vtk_dir: Path, feb_path: Path, step_min: int,
                                 step_max: int) -> Dict:
    """
    Extract displacement metrics relative to gravity equilibrium (step 0).

    Args:
        vtk_dir: Directory containing VTK files.
        feb_path: Path to the .feb file.
        step_min: Minimum step (inclusive).
        step_max: Maximum step (exclusive).

    Returns:
        Dictionary with relative and absolute displacement metrics.
    """
    surface_nodes = get_surface_nodes(feb_path)
    vtks = list_vtks(vtk_dir, step_min, step_max)
    if not vtks:
        return {}

    # Get baseline (step 0 = end of gravity phase, start of jump)
    baseline_total, baseline_vertical, baseline_mean = get_baseline_displacement(
        vtk_dir, surface_nodes)
    if baseline_total is None:
        print("WARNING: Proceeding without baseline subtraction!")
        baseline_total = 0
        baseline_vertical = 0
        baseline_mean = 0

    max_disp_relative = 0
    max_disp_step = 0
    all_displacements_relative = []
    all_displacements_absolute = []

    for vtk_path in vtks:
        mesh = meshio.read(vtk_path)
        U = mesh.point_data.get("displacement")
        if U is None:
            continue

        surface_indices = np.array([nid - 1 for nid in surface_nodes
                                    if 0 <= nid - 1 < len(mesh.points)])

        # Calculate absolute displacement
        U_vertical_abs = np.abs(U[surface_indices, 1])  # meters
        U_total_abs = np.linalg.norm(U[surface_indices], axis=1)  # meters

        # Calculate relative displacement (deviation from gravity equilibrium)
        if isinstance(baseline_vertical, np.ndarray):
            U_vertical_relative = np.abs(
                U_vertical_abs - np.abs(baseline_vertical))
        else:
            U_vertical_relative = U_vertical_abs

        if isinstance(baseline_total, np.ndarray):
            U_total_relative = np.abs(U_total_abs - baseline_total)
        else:
            U_total_relative = U_total_abs

        # Track maximum relative vertical displacement
        current_max = U_vertical_relative.max() * 100  # convert to cm
        if current_max > max_disp_relative:
            max_disp_relative = current_max
            max_disp_step = int(
                re.search(r"\.(\d+)\.vtk$", vtk_path.name).group(1))

        # Store all values
        all_displacements_relative.extend(U_total_relative * 100)  # cm
        all_displacements_absolute.extend(U_total_abs * 100)  # cm

    return {
        # PRIMARY METRICS (relative to gravity)
        "peak_vertical_relative_cm": max_disp_relative,
        "mean_displacement_relative_cm": np.mean(all_displacements_relative),
        "median_displacement_relative_cm": np.median(
            all_displacements_relative),
        "std_displacement_relative_cm": np.std(all_displacements_relative),

        # ABSOLUTE METRICS (from undeformed state)
        "peak_vertical_absolute_cm": max_disp_relative + (
            baseline_mean * 100 if baseline_mean else 0),
        "mean_displacement_absolute_cm": np.mean(all_displacements_absolute),
        "baseline_gravity_cm": baseline_mean * 100 if baseline_mean else 0,

        # METADATA
        "peak_step": max_disp_step,
    }


def extract_surface_nodes(feb_path: Path, target_y=0.035):
    """
    Extract surface nodes at key anatomical positions from a .feb file.

    Args:
        feb_path: Path to the .feb file.
        target_y: Y-coordinate for horizontal plane landmarks.

    Returns:
        Dictionary of landmark names to node IDs and coordinates.
    """
    start_marker = '<Nodes name="Object01">'
    end_marker = "</Nodes>"
    id_re = re.compile(r'id="(\d+)"')
    coord_re = re.compile(r"(?<=>)([^<]+)(?=</)")

    nodes = {}
    in_elem = False

    with feb_path.open() as file:
        for line in file:
            s = line.strip()
            if s == start_marker:
                in_elem = True
                continue
            if s == end_marker:
                in_elem = False
                break
            if in_elem:
                id_match = id_re.search(line)
                coord_match = coord_re.search(line)
                if id_match and coord_match:
                    node_id = int(id_match.group(1))
                    coords = [float(c) for c in
                              coord_match.group(1).split(",")]
                    nodes[node_id] = tuple(coords)

    selected = {}

    # Nipple (maximum Y)
    nipple_id = max(nodes.items(), key=lambda n: n[1][1])
    selected['nipple'] = {'id': nipple_id[0], 'coords': nipple_id[1]}

    tolerance_y = 0.005
    target_y_nodes = {nid: coords for nid, coords in nodes.items()
                      if abs(coords[1] - target_y) < tolerance_y}
    if not target_y_nodes:
        return selected

    # Left side
    left_id = min(target_y_nodes.items(), key=lambda n: n[1][0])
    selected['left'] = {'id': left_id[0], 'coords': left_id[1]}
    left_z = left_id[1][2]

    # Right side
    tolerance_z = 0.01
    right_candidates = {nid: coords for nid, coords in target_y_nodes.items()
                        if abs(coords[2] - left_z) < tolerance_z and coords[
                            0] > 0}
    if right_candidates:
        right_id = max(right_candidates.items(), key=lambda n: n[1][0])
    else:
        right_id = max(target_y_nodes.items(), key=lambda n: n[1][0])
    selected['right'] = {'id': right_id[0], 'coords': right_id[1]}

    # Superior
    top_id = max(target_y_nodes.items(), key=lambda n: n[1][2])
    selected['superior'] = {'id': top_id[0], 'coords': top_id[1]}
    top_x = top_id[1][0]

    # Inferior
    tolerance_x = 0.01
    bottom_candidates = {nid: coords for nid, coords in target_y_nodes.items()
                         if abs(coords[0] - top_x) < tolerance_x and coords[
                             2] < 0}
    if bottom_candidates:
        bottom_id = min(bottom_candidates.items(), key=lambda n: n[1][2])
    else:
        bottom_id = min(target_y_nodes.items(), key=lambda n: n[1][2])
    selected['inferior'] = {'id': bottom_id[0], 'coords': bottom_id[1]}

    return selected


def extract_landmark_displacements(model_dirs: dict, landmarks: dict,
                                   surface_nodes: set, step: int,
                                   relative_to_baseline: bool = True):
    """
    Extract displacement at specific landmarks for all models.

    Args:
        model_dirs: Dict of model names to VTK directories.
        landmarks: Dict of landmark names to node info.
        surface_nodes: Set of surface node IDs.
        step: VTK step number to analyze.
        relative_to_baseline: If True, subtract baseline (step 0) displacement.

    Returns:
        DataFrame with displacement at each landmark for each model.
    """
    results = []

    for model_name, vtk_dir in model_dirs.items():
        # Get baseline if requested
        baseline_U = None
        if relative_to_baseline:
            baseline_vtks = list(vtk_dir.glob("*.0.vtk"))
            if baseline_vtks:
                mesh_baseline = meshio.read(baseline_vtks[0])
                baseline_U = mesh_baseline.point_data.get("displacement")
                print(f"  {model_name}: Using baseline from step 0")
            else:
                print(f"  WARNING: {model_name} has no step 0 baseline!")

        # Get displacement at target step
        vtk_files = list(vtk_dir.glob(f"*.{int(step)}.vtk"))
        if not vtk_files:
            print(
                f"  WARNING: No VTK file found for {model_name} at step {step}")
            continue

        mesh = meshio.read(vtk_files[0])
        U = mesh.point_data.get("displacement")
        if U is None:
            print(f"  WARNING: No displacement data for {model_name}")
            continue

        # Extract displacement at each landmark
        for landmark_name, landmark_info in landmarks.items():
            idx = landmark_info["id"] - 1  # Convert to 0-based index
            if idx >= len(U):
                print(f"  WARNING: Index {idx} out of bounds for {model_name}")
                continue

            # Get displacement vector
            disp_vector = U[idx]
            # Subtract baseline if available
            if baseline_U is not None and idx < len(baseline_U):
                disp_vector = disp_vector - baseline_U[idx]
            disp_vector = disp_vector * 1000  # Convert to mm
            disp_mag = np.linalg.norm(disp_vector)

            results.append({
                "model": model_name,
                "landmark": landmark_name,
                "displacement_mm": disp_mag,
                "disp_x_mm": disp_vector[0],
                "disp_y_mm": disp_vector[1],
                "disp_z_mm": disp_vector[2],
                "position_x": landmark_info["coords"][0],
                "position_y": landmark_info["coords"][1],
                "position_z": landmark_info["coords"][2],
                "relative_to_baseline": relative_to_baseline,
            })

    return pd.DataFrame(results)


def extract_time_from_vtk(vtk_path: Path) -> float:
    """
    Extract simulation time from the VTK file header.

    Args:
        vtk_path: Path to the VTK file.

    Returns:
        Time value as float, or 0.0 if not found.
    """
    with open(vtk_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 5:
                break  # Only check first few lines
            if line.startswith('time'):
                return float(line.split()[1].strip())
    return 0.0


def get_common_times(vtk_dir: Path, step_min: int, step_max: int):
    """
    Extract time array for all VTK files in a directory within a step range.

    Args:
        vtk_dir: Directory containing VTK files.
        step_min: Minimum step (inclusive).
        step_max: Maximum step (exclusive).

    Returns:
        Numpy array of time values.
    """
    vtks = []
    for p in vtk_dir.glob("*.vtk"):
        m = re.search(r"(\d+)\.vtk$", p.name)
        if m and step_min <= int(m.group(1)) < step_max:
            vtks.append((int(m.group(1)), p))

    # Sort by step number
    vtks = [p for _, p in sorted(vtks, key=lambda t: t[0])]
    times = np.array([extract_time_from_vtk(p) for p in vtks])
    return times
