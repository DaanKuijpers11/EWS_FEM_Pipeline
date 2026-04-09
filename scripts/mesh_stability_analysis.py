    """
    Mesh stability analysis for density sweep.

    Evaluates:
    - Mesh generation success
    - FEBio simulation success
    - Number of elements
    - Number of nodes
    - Number of VTK output files

    Outputs:
    - CSV summary table: mesh_stability_results.csv
    """

    from pathlib import Path
    import pandas as pd
    import xml.etree.ElementTree as ET

    # ==================== CONFIG ====================
    print(">>> Script gestart")
    BASE_DIR = Path("C:\\Users\\20223231\\OneDrive - TU Eindhoven\\Documents\\Word documenten\\Master Jaar 1\\Q4 - Stage\\Github EWS-Pipeline-FEM")


    DENSITY_MODELS = {
        "density_100": BASE_DIR / "density_models" / "density_100",
        # "density_150": BASE_DIR / "density_models" / "density_150",
        # "density_200": BASE_DIR / "density_models" / "density_200",
        # "density_250": BASE_DIR / "density_models" / "density_250",
    }

    # ==================== HELPERS ====================

    def check_mesh_success(model_dir: Path):
        """Check if .feb file exists (mesh generated)."""
        feb_files = list(model_dir.glob("*.feb"))
        return len(feb_files) > 0, feb_files[0] if feb_files else None


    def check_febio_success(model_dir: Path):
        """Check if FEBio produced VTK files."""
        vtk_dir = model_dir / "output"
        vtk_files = list(vtk_dir.glob("*.vtk"))
        return len(vtk_files) > 0, len(vtk_files)


    def extract_mesh_info(feb_path: Path):
        """Extract number of nodes and elements from FEB file."""
        try:
            tree = ET.parse(feb_path)
            root = tree.getroot()

            nodes = root.findall(".//Nodes[@name='Object01']/node")
            n_nodes = len(nodes)

            elems_adipose = root.findall(".//Elements[@name='adipose_part']/elem")
            elems_glandular = root.findall(".//Elements[@name='glandular_part']/elem")

            n_elements = len(elems_adipose) + len(elems_glandular)

            return n_nodes, n_elements

        except Exception as e:
            print(f"Error reading {feb_path}: {e}")
            return None, None


    # ==================== MAIN ====================

    def run_mesh_analysis():
        print(">>> Script gestart")
        results = []

        print("\n=== Mesh Stability Analysis ===")

        for model_name, model_dir in DENSITY_MODELS.items():
            print(f"\nProcessing: {model_name}")

            density_value = int(model_name.split("_")[1])

            mesh_ok, feb_path = check_mesh_success(model_dir)
            febio_ok, n_vtks = check_febio_success(model_dir)

            n_nodes, n_elements = (None, None)
            if feb_path:
                n_nodes, n_elements = extract_mesh_info(feb_path)

            results.append({
                "model": model_name,
                "mesh_success": mesh_ok,
                "febio_success": febio_ok,
                "n_nodes": n_nodes,
                "n_elements": n_elements,
                "n_vtk_files": n_vtks,
                "mesh_density": density_value,
            })

            print(f"  Mesh success: {mesh_ok}")
            print(f"  FEBio success: {febio_ok}")
            print(f"  Nodes: {n_nodes}")
            print(f"  Elements: {n_elements}")
            print(f"  VTK files: {n_vtks}")

        # Save results
        df = pd.DataFrame(results)
        output_file = BASE_DIR / "mesh_stability_results.csv"
        df.to_csv(output_file, index=False)

        print("\nSaved results to:")
        print(output_file)

        return df


    if __name__ == "__main__":
        run_mesh_analysis()