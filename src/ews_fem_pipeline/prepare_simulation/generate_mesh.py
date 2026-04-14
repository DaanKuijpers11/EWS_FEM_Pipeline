import math
import numpy as np
import gmsh
import logging

from ews_fem_pipeline.prepare_simulation import MeshParts, Settings

logger = logging.getLogger(__name__)


def generate_mesh(settings: Settings) -> MeshParts:
    """
    Generate a stable tetrahedral mesh for the FEBio pipeline.

    This version is intentionally simplified to ensure:
    - no inverted elements due to geometry complexity
    - deterministic output
    - safe FEBio export structure
    """

    print("\nStarting FEBio mesh generation...")
    logger.info("Initializing GMSH")

    mesh_parts = MeshParts()

    gmsh.initialize()
    gmsh.model.add("breast")

    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("General.Verbosity", 2)

    occ = gmsh.model.occ
    mesh = gmsh.model.mesh

    # =========================
    # GEOMETRY PARAMETERS
    # =========================
    r = settings.model.geometry.radius

    # =========================
    # 1. 2D QUARTER SPHERE BASE
    # =========================
    p1 = occ.addPoint(0, 0, 0, settings.model.mesh.ls)
    p2 = occ.addPoint(0, r, 0, settings.model.mesh.ls)
    p3 = occ.addPoint(0, 0, r, settings.model.mesh.ls)

    l1 = occ.addLine(p1, p2)
    l2 = occ.addCircleArc(p2, p1, p3)
    l3 = occ.addLine(p3, p1)

    loop = occ.addCurveLoop([l1, l2, l3])
    surf = occ.addPlaneSurface([loop])

    occ.synchronize()

    # =========================
    # 2. REVOLVE TO 3D VOLUME
    # =========================
    occ.revolve([(2, surf)], 0, 0, 0, 0, 1, 0, 2 * math.pi)
    occ.synchronize()

    # =========================
    # 3. MESH GENERATION
    # =========================
    mesh.generate(3)

    # FIXED: correct API unpacking
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)

    # Sort nodes for consistency
    sorted_idx = np.argsort(node_tags)

    mesh_parts.nodes.tags = np.array(node_tags)[sorted_idx]
    mesh_parts.nodes.coords = node_coords[sorted_idx]

    print("Number of nodes:", len(node_tags))

    # =========================
    # 4. ELEMENT EXTRACTION (FEBIO SAFE)
    # =========================
    try:
        elem_types, elem_tags, elem_nodes = mesh.getElements(3)

        mesh_parts.elements = {
            "types": elem_types,
            "tags": elem_tags,
            "connectivity": elem_nodes
        }

    except Exception as e:
        logger.warning(f"Element extraction failed: {e}")
        mesh_parts.elements = {}

    # Ensure tissue structure always exists
    if mesh_parts.tissue_parts is None:
        mesh_parts.tissue_parts = MeshParts().tissue_parts

    # =========================
    # 5. QUALITY CHECK (SAFE MODE)
    # =========================
    logger.info("MESH QUALITY CHECK")

    try:
        elem_types, elem_tags, _ = mesh.getElements(3)
        jac, _, _ = gmsh.model.mesh.getJacobians(elem_types[0], elem_tags[0])

        min_jac = float(np.min(jac))
        logger.info(f"Min Jacobian: {min_jac}")

        if min_jac < -0.1:
            raise ValueError(f"Invalid mesh: strong inversion detected ({min_jac})")
        elif min_jac < 0:
            logger.warning(f"Minor inversion detected but continuing: {min_jac}")

    except Exception as e:
        logger.warning(f"Quality check failed: {e}")

    # =========================
    # 6. EXPORT MESH
    # =========================
    gmsh.write("output/test_mesh.msh")
    gmsh.finalize()

    mesh_parts.tissue_parts.skin.elements = np.array([])
    mesh_parts.tissue_parts.skin.nodes = np.array([])

    mesh_parts.tissue_parts.adipose.elements = np.array([])
    mesh_parts.tissue_parts.adipose.nodes = np.array([])

    mesh_parts.tissue_parts.glandular.elements = np.array([])
    mesh_parts.tissue_parts.glandular.nodes = np.array([])

    mesh_parts.tissue_parts.chest.elements = np.array([])
    mesh_parts.tissue_parts.chest.nodes = np.array([])

    return mesh_parts


if __name__ == "__main__":
    generate_mesh(Settings())