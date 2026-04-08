import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from ews_fem_pipeline.prepare_simulation import MeshParts, Settings, FEBElement, Constants, BoundaryCondition, \
    Loads, write_nodes_to_xml, write_elements_to_xml, write_xml


def add_density_map_math(root, settings, mesh):
    """
    Compute and write per-node density values to the FEBio file for visualization.

    Evaluates the spatially-varying density field at each mesh node by:
    1. Determining which tissue each node belongs to
    2. Computing density based on tissue baseline + heterogeneity patterns
    3. Writing node-by-node density values to a <NodeData> XML section

    This allows ParaView and FEBio Studio to visualize the density distribution
    as a color-mapped field on the mesh.

    Args:
        root: Root XML element of the .feb file
        settings: Simulation settings containing material properties and heterogeneity
        mesh: Mesh data containing node coordinates and tissue connectivity

    Note:
        - Nodes shared between tissues will have density overwritten by the last tissue processed
        - Processing order: skin → glandular → adipose
        - Heterogeneity patterns (lobules + radial) are evaluated numerically at each node
    """
    # Extract node information from mesh
    node_coords = mesh.nodes.coords # (n_nodes, 3) array of (x,y,z) positions
    node_tags = mesh.nodes.tags # Node IDs from gmsh
    n_nodes = len(node_coords)

    # Initialize density array (filled per tissue)
    density_values = np.zeros(n_nodes, dtype=float)

    # Create mapping from node ID to array index for fast lookup
    node_tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

    # Compute density for each tissue's nodes
    tissues = mesh.tissue_parts

    # Process tissues in order (later tissues overwrite shared nodes)
    for tissue_name, material in [
        ("skin", settings.material.skin),
        ("glandular", settings.material.glandular),
        ("adipose", settings.material.adipose)
    ]:
        tissue = getattr(tissues, tissue_name)

        # Get all unique node IDs belonging to this tissue
        tissue_nodes_flat = tissue.nodes.flatten()  # Elements have multiple nodes
        unique_tissue_nodes = np.unique(tissue_nodes_flat)

        # Extract baseline density and heterogeneity config
        base_density = float(material.density)
        h = getattr(material, "hetero", None)

        # Evaluate density at each node in this tissue
        for node_tag in unique_tissue_nodes:
            node_idx = node_tag_to_idx[int(node_tag)]
            x, y, z = node_coords[node_idx]

            # Start with baseline tissue density
            rho = base_density

            # Add heterogeneity contributions if enabled
            if h and h.enabled:
                # Add Gaussian lobule contributions
                for L in h.lobules:
                    lx, ly, lz = L.center
                    s = float(L.width)
                    amp = float(getattr(L, "amp_rho", 0.0))

                    if amp:
                        # Evaluate 3D Gaussian: amp * exp(-r²/σ²)
                        dist_sq = (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
                        rho += amp * np.exp(-dist_sq / (s ** 2))

                # Add radial background gradient (increases with distance from center)
                if h.radial_center is not None:
                    xr, yr, zr = h.radial_center
                    sR = float(h.radial_L)
                    aR = float(getattr(h, "radial_alpha_rho", 0.0))

                    if aR != 0.0:
                        # Evaluate radial pattern: alpha * (1 - exp(-r²/L²))
                        dist_sq = (x - xr) ** 2 + (y - yr) ** 2 + (z - zr) ** 2
                        rho += aR * (1 - np.exp(-dist_sq / (sR ** 2)))

            # Store computed density for this node
            density_values[node_idx] = rho

    # Write density values to XML as NodeData
    meshdata = ET.SubElement(root, "MeshData")
    nd = ET.SubElement(meshdata, "NodeData", name="density_map",
                       node_set="Object01")

    # Write individual <node lid="...">value</node> entries
    for tag, value in zip(node_tags, density_values):
        node_elem = ET.SubElement(nd, "node", lid=str(int(tag)))
        node_elem.text = f"{value:.6f}"

def write_to_feb(filepath: Path, mesh: MeshParts, settings: Settings):
    """
    This file writes the mesh and simulation settings to the .xml/.feb file using the settings from simulation_settings.py
    """


    FEB_element = FEBElement()
    ###############################################################################################################
    # Root #
    ########
    root = ET.Element('febio_spec', version="4.0")

    module_elem = FEB_element.module.to_xml(parent=root)

    ##############################################################################################################
    # Globals #
    ###########
    globals_elem = FEB_element.globals.to_xml(parent=root)

    ##############################################################################################################
    # Constants #
    #############
    constants_elem = FEB_element.constants.to_xml(parent=globals_elem)
    Constants().to_xml(parent=constants_elem)

    #################################################################################################################
    # Materials #
    #############
    material_elem = FEB_element.material.to_xml(parent=root)

    skin_elem = FEB_element.skin.to_xml(parent=material_elem)
    settings.material.skin.to_xml(parent=skin_elem, tumor=settings.material.tumor.skin)

    adipose_elem = FEB_element.adipose.to_xml(parent=material_elem)
    settings.material.adipose.to_xml(parent=adipose_elem, tumor=settings.material.tumor.adipose)

    glandualar_elem = FEB_element.glandular.to_xml(parent=material_elem)
    settings.material.glandular.to_xml(parent=glandualar_elem, tumor=settings.material.tumor.glandular)

    #################################################################################################################
    # Mesh #
    ########
    mesh_elem = FEB_element.mesh.to_xml(parent=root)

    # Nodes
    write_nodes_to_xml(parent=mesh_elem, mesh=mesh)

    # Elements
    write_elements_to_xml(parent=mesh_elem, mesh=mesh)

    # Mass damping and gravity
    FEB_element.mass_damping.to_xml(parent=mesh_elem)
    FEB_element.gravitational_acceleration.to_xml(parent=mesh_elem)

    #################################################################################################################
    # Mesh domains #
    ################
    mesh_domains_elem = FEB_element.mesh_domains.to_xml(parent=root)

    # Shell domain
    shell_elem = FEB_element.shell_domain.to_xml(parent=mesh_domains_elem)
    FEB_element.shell_thickness.to_xml(parent=shell_elem)

    # Solid domain
    FEB_element.solid_domain_glandular.to_xml(parent=mesh_domains_elem)
    FEB_element.solid_domain_adipose.to_xml(parent=mesh_domains_elem)

    # Add all material property maps for visualizations in paraview
    add_density_map_math(root, settings, mesh)

    # Loads
    loads_elem = FEB_element.loads.to_xml(parent=root)

    body_load_elem = FEB_element.body_load1.to_xml(parent=loads_elem)
    FEB_element.force.to_xml(parent=body_load_elem)

    #################################################################################################################
    # Steps #
    #########
    step_elem = FEB_element.step.to_xml(parent=root)

    #################################################################################################################
    # Step 1 #
    ##########
    step1_elem = FEB_element.step1.to_xml(parent=step_elem)

    # Control
    control1_elem = FEB_element.control.to_xml(parent=step1_elem)
    control1_field = settings.simulation.control_step1.to_xml(parent=control1_elem)

    # Time Stepper
    timestepper1_elem = FEB_element.time_stepper.to_xml(parent=control1_elem)
    timestepper1_field = settings.simulation.timestepper_step1.to_xml(parent=timestepper1_elem)

    # Solver
    solver1_elem = FEB_element.solver.to_xml(parent=control1_elem)
    solver1_field = settings.simulation.solver_step1.to_xml(parent=solver1_elem)

    # qn_method
    qnmethod1_elem = FEB_element.qn_method.to_xml(parent=solver1_elem)
    settings.simulation.qnmethod_step1.to_xml(parent=qnmethod1_elem)

    # Boundary
    boundary1_elem = FEB_element.boundary.to_xml(parent=step1_elem)
    boundary1_field = FEB_element.boundary_zero_displacement.to_xml(parent=boundary1_elem)
    BoundaryCondition().zero_displacement.to_xml(parent=boundary1_field)

    #################################################################################################################
    # Step 2 #
    ##########
    step2_elem = FEB_element.step2.to_xml(parent=step_elem)

    # Control
    control2_elem = FEB_element.control.to_xml(parent=step2_elem)
    control3_field = settings.simulation.control_step2.to_xml(parent=control2_elem)

    # Time Stepper
    timestepper2_elem = FEB_element.time_stepper.to_xml(parent=control2_elem)
    timestepper2_field = settings.simulation.timestepper_step2.to_xml(parent=timestepper2_elem)

    # Solver
    solver2_elem = FEB_element.solver.to_xml(parent=control2_elem)
    solver2_field = settings.simulation.solver_step2.to_xml(parent=solver2_elem)

    # qn_method
    qnmethod2_elem = FEB_element.qn_method.to_xml(parent=solver2_elem)
    settings.simulation.qnmethod_step2.to_xml(parent=qnmethod2_elem)

    # Boundary
    boundary2_elem = FEB_element.boundary.to_xml(parent=step2_elem)

    boundary2_field = FEB_element.boundary_parabolic_trajectory.to_xml(parent=boundary2_elem)
    BoundaryCondition().prescribed_displacement.to_xml(parent=boundary2_field)

    boundary2_field = FEB_element.boundary_only_z_displacement.to_xml(parent=boundary2_elem)
    BoundaryCondition().only_z_displacement.to_xml(parent=boundary2_field)

    # Loads
    loads2_elem = FEB_element.loads.to_xml(parent=step2_elem)
    bodyload2_elem = FEB_element.body_load2.to_xml(parent=loads2_elem)
    Loads().to_xml(parent=bodyload2_elem)


    #################################################################################################################
    # Load Data #
    #############
    loaddata_elem = FEB_element.load_data.to_xml(parent=root)

    # Load Controller |points for increasing gravity
    settings.simulation.gravity.to_xml(parent = loaddata_elem)

    # Load Controller | points for parabola trajectory of breast
    settings.simulation.parabolic_jump.to_xml(parent = loaddata_elem)

    # Load Controller | points for output
    settings.simulation.animation.to_xml(parent = loaddata_elem)

    #################################################################################################################
    # Output #
    ##########
    output_elem = FEB_element.output.to_xml(parent=root)
    settings.simulation.output.to_xml(parent=output_elem, filepath=filepath)

    #################################################################################################################
    # Write to .feb #
    #################

    write_xml(root=root, filepath=filepath)

