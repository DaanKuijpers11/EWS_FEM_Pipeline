import math
import numpy as np
import gmsh
import logging

from pathlib import Path
from ews_fem_pipeline.prepare_simulation import MeshParts, Settings

logger = logging.getLogger(__name__)


def generate_mesh(settings: Settings) -> MeshParts:
    """
    COMSOL-ready mesh generator.

    Creates:
    - Geometry (Gmsh CAD)
    - Physical groups (tissues)
    - Stable 3D mesh
    - MeshParts mapping for FEM pipeline
    """

    print("\nStarting COMSOL-ready mesh generation...")

    mesh_parts = MeshParts()

    # -----------------------------
    # INIT GMSH
    # -----------------------------
    gmsh.initialize()
    gmsh.model.add("breast")

    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("General.Verbosity", 2)

    dim2 = 2
    dim3 = 3

    occ = gmsh.model.occ
    mesh = gmsh.model.mesh

    # -----------------------------
    # GEOMETRY: BREAST QUADRANT
    # -----------------------------
    p1 = occ.addPoint(0, 0, 0, settings.model.mesh.ls)
    p2 = occ.addPoint(0, settings.model.geometry.radius, 0, settings.model.mesh.ls)
    p3 = occ.addPoint(0, 0, settings.model.geometry.radius, settings.model.mesh.ls)

    l1 = occ.addLine(p1, p2)
    l2 = occ.addCircleArc(p2, p1, p3)
    l3 = occ.addLine(p3, p1)

    loop1 = occ.addCurveLoop([l1, l2, l3])
    s1 = occ.addPlaneSurface([loop1])

    # -----------------------------
    # GEOMETRY: NIPPLE / GLAND REGION
    # -----------------------------
    p4 = occ.addPoint(0, -settings.model.geometry.left_position_ellipse, 0, settings.model.mesh.ls)
    p5 = occ.addPoint(0, settings.model.geometry.radius + settings.model.geometry.position_nipple, 0, settings.model.mesh.ls)
    p6 = occ.addPoint(
        0,
        (settings.model.geometry.radius + settings.model.geometry.position_nipple -
         settings.model.geometry.left_position_ellipse) / 2,
        -settings.model.geometry.position_center_ellipse,
        settings.model.mesh.ls,
    )

    l4 = occ.addEllipseArc(p4, p6, p5, p5)
    l5 = occ.addLine(p4, p5)

    loop2 = occ.addCurveLoop([l4, l5])
    s2 = occ.addPlaneSurface([loop2])

    # -----------------------------
    # GEOMETRY: CHEST WALL
    # -----------------------------
    p7 = occ.addPoint(
        0,
        -settings.model.geometry.thickness_chest_wall,
        settings.model.geometry.radius,
        settings.model.mesh.ls,
    )
    p8 = occ.addPoint(
        0,
        -settings.model.geometry.thickness_chest_wall,
        0,
        settings.model.mesh.ls,
    )

    l6 = occ.addLine(p3, p7)
    l7 = occ.addLine(p7, p8)
    l8 = occ.addLine(p8, p1)

    loop3 = occ.addCurveLoop([l8, l3, l6, l7])
    s3 = occ.addPlaneSurface([loop3])

    # -----------------------------
    # BOOLEAN OPERATIONS
    # -----------------------------
    occ.fragment([(dim2, s1), (dim2, s2)], [(dim2, s3)])
    occ.synchronize()

    # -----------------------------
    # REVOLVE INTO 3D
    # -----------------------------
    occ.revolve(occ.getEntities(dim2), 0, 0, 0, 0, 1, 0, 2 * math.pi)

    occ.synchronize()

    # -----------------------------
    # PHYSICAL GROUPS (CRITICAL FOR COMSOL + FEM)
    # -----------------------------

    # NOTE:
    # These groups define MATERIAL REGIONS
    # This is what COMSOL will import as domains

    skin_tag = gmsh.model.addPhysicalGroup(3, [1])
    gmsh.model.setPhysicalName(3, skin_tag, "skin")

    gland_tag = gmsh.model.addPhysicalGroup(3, [2])
    gmsh.model.setPhysicalName(3, gland_tag, "glandular")

    fat_tag = gmsh.model.addPhysicalGroup(3, [3])
    gmsh.model.setPhysicalName(3, fat_tag, "adipose")

    chest_tag = gmsh.model.addPhysicalGroup(3, [4])
    gmsh.model.setPhysicalName(3, chest_tag, "chest")

    # -----------------------------
    # MESH GENERATION
    # -----------------------------
    mesh.generate(3)

    node_tags, node_coords = gmsh.model.mesh.getNodes()
    node_coords = np.reshape(node_coords, (-1, 3))

    print(f"Nodes generated: {len(node_tags)}")

    # -----------------------------
    # QUALITY CHECK
    # -----------------------------
    logger.info("--- MESH QUALITY CHECK ---")

    try:
        elem_types, elem_tags, _ = mesh.getElements(3)
        jac, _, _ = mesh.getJacobians(elem_types[0], elem_tags[0])
        logger.info(f"Min Jacobian: {np.min(jac)}")
    except Exception as e:
        logger.warning(f"Jacobian check failed: {e}")

    # -----------------------------
    # BUILD MESH PARTS (CRITICAL FIX)
    # -----------------------------
    mesh_parts.nodes.tags = node_tags
    mesh_parts.nodes.coords = node_coords

    # IMPORTANT:
    # We now use physical groups instead of manual assignment
    # This makes COMSOL + FEBio consistent

    def get_elements_by_physical(name):
        try:
            dim = 3
            tag = gmsh.model.getPhysicalGroups(dim)
            for d, t in tag:
                if gmsh.model.getPhysicalName(d, t) == name:
                    return gmsh.model.mesh.getElements(dim, t)[2][0]
        except Exception:
            return []

    mesh_parts.tissue_parts.skin.elements = get_elements_by_physical("skin")
    mesh_parts.tissue_parts.glandular.elements = get_elements_by_physical("glandular")
    mesh_parts.tissue_parts.adipose.elements = get_elements_by_physical("adipose")
    mesh_parts.tissue_parts.chest.elements = get_elements_by_physical("chest")

    # -----------------------------
    # EXPORT MESH
    # -----------------------------
    Path("output").mkdir(exist_ok=True)
    gmsh.write("output/test_mesh.msh")

    gmsh.finalize()

    return mesh_parts


if __name__ == "__main__":
    generate_mesh(settings=Settings())