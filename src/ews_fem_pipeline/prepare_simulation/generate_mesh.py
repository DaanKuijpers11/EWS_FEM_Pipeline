import math
import numpy as np

import gmsh

from ews_fem_pipeline.prepare_simulation import MeshParts, Settings

def generate_mesh(settings: Settings()):
    """
    In this function, the mesh is generated from settings extracted from the Settings class.
    These settings are explained in detail in model_settings.py under MeshSettings and GeometrySettings.
    """
    print("\nStarting mesh generation...")

    mesh_parts = MeshParts()

    gmsh.initialize()
    gmsh.model.add("breast")

    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("General.Verbosity", 2)

    dim0, dim1, dim2, dim3 = 0, 1, 2, 3

    build = gmsh.model.occ
    mesh = gmsh.model.mesh

    #############################
    # Construct breast quadrant #
    #############################

    p1 = build.addPoint(0, 0, 0, settings.model.mesh.ls, 1)
    p2 = build.addPoint(0, settings.model.geometry.radius, 0, settings.model.mesh.ls, 2)
    p3 = build.addPoint(0, 0, settings.model.geometry.radius, settings.model.mesh.ls, 3)

    l1 = build.addLine(p1, p2, 1)
    l2 = build.addCircleArc(p2, p1, p3, 2)
    l3 = build.addLine(p3, p1, 3)

    loop1 = build.addCurveLoop([l1, l2, l3], 1)
    s1 = build.addPlaneSurface([loop1], 1)

    p4 = build.addPoint(0, -settings.model.geometry.left_position_ellipse, 0, settings.model.mesh.ls, 4)
    p5 = build.addPoint(0, settings.model.geometry.radius + settings.model.geometry.position_nipple, 0, settings.model.mesh.ls, 5)
    p6 = build.addPoint(
        0,
        (settings.model.geometry.radius + settings.model.geometry.position_nipple -
         settings.model.geometry.left_position_ellipse) / 2,
        -settings.model.geometry.position_center_ellipse,
        settings.model.mesh.ls,
        6,
    )

    l4 = build.addEllipseArc(p4, p6, p5, p5, 4)
    l5 = build.addLine(p4, p5, 5)

    loop2 = build.addCurveLoop([l4, l5], 2)
    s2 = build.addPlaneSurface([loop2], 2)

    # Back side
    p7 = build.addPoint(0, -settings.model.geometry.thickness_chest_wall, settings.model.geometry.radius, settings.model.mesh.ls, 7)
    p8 = build.addPoint(0, -settings.model.geometry.thickness_chest_wall, 0, settings.model.mesh.ls, 8)

    l6 = build.addLine(p3, p7, 6)
    l7 = build.addLine(p7, p8, 7)
    l8 = build.addLine(p8, p1, 8)

    loop3 = build.addCurveLoop(([l8, l3, l6, l7]))
    s3 = build.addPlaneSurface([loop3])

    build.fragment([(dim2, s1), (dim2, s2)], [(dim2, s3)])

    #############################
    # Rebuild geometry cleanup  #
    #############################

    all_points = build.getEntities(dim0)
    all_lines = build.getEntities(dim1)
    all_surfaces = build.getEntities(dim2)

    for _, idx in all_points:
        globals()[f"p{idx}"] = idx

    for _, idx in all_lines:
        globals()[f"l{idx}"] = idx

    build.remove(all_surfaces)

    build.remove(
        [(dim1, l3), (dim1, l4), (dim1, l5), (dim1, l7),
         (dim1, l8), (dim1, l9), (dim1, l11), (dim1, l12), (dim1, l14)]
    )

    build.remove([(dim0, p6), (dim0, p10), (dim0, p12), (dim0, p14)])

    l3 = build.addLine(p16, p13, 3)
    l4 = build.addLine(p8, p15, 4)

    ###############################################
    # Revolve into 3D
    ###############################################

    build.revolve(build.getEntities(dim1), 0, 0, 0, 0, 1, 0, 2 * math.pi)

    tissues = mesh_parts.tissue_parts

    surfloop_gland = build.addSurfaceLoop([1, 2, 5])
    surfloop_fat = build.addSurfaceLoop([1, 2, 3, 4, 6])

    tissues.skin.tags = [9, 10]
    tissues.chest.tags = 11
    tissues.glandular.tags = build.addVolume([surfloop_gland])
    tissues.adipose.tags = build.addVolume([surfloop_fat])

    build.fragment([(dim3, 1)], [(dim3, 2)])
    build.remove(build.getEntities(dim2))
    build.remove(build.getEntities(dim1))
    build.remove(build.getEntities(dim0))

    build.synchronize()

    ####################
    # Generate mesh
    ####################

    for curve in build.getEntities(dim1):
        length = build.getMass(dim1, curve[1])
        mesh.setTransfiniteCurve(curve[1], int(settings.model.mesh.density * length))

    mesh.generate(dim3)
    mesh.setOrder(settings.model.mesh.order)

    if settings.model.mesh.optimize:
        mesh.optimize("HighOrder" if settings.model.mesh.order > 1 else "")

    #############################
    # MESH VALIDATION
    #############################

    logger.info("--- MESH QUALITY CHECK ---") # does not work yet for all loggers

    min_jac = None
    min_q = None

    try:
        elem_types, elem_tags, _ = mesh.getElements(dim3)
        jacobians, _, _ = mesh.getJacobians(elem_types[0], elem_tags[0])
        min_jac = float(np.min(jacobians))
    except Exception as e:
        logger.warning(f"Jacobian check failed: {e}")

    try:
        qualities = mesh.getElementQualities()
        if qualities:
            qualities = np.array(qualities)
            min_q = float(np.min(qualities))
            mean_q = float(np.mean(qualities))
            bad_ratio = float(np.sum(qualities < 0.1) / len(qualities))

            logger.info(f"Min quality: {min_q}")
            logger.info(f"Mean quality: {mean_q}")
            logger.info(f"Bad elements (<0.1): {bad_ratio*100:.2f}%")
    except Exception as e:
        logger.warning(f"Quality check failed: {e}")

    if min_jac is not None:
        logger.info(f"Min Jacobian: {min_jac}")
        if min_jac <= 0:
            raise ValueError(f"Invalid mesh: negative Jacobian ({min_jac})")

    if min_q is not None and min_q < 0.05:
        logger.warning("Very low element quality detected")

    #########################################
    # Extract nodes
    #########################################

    node_tags, node_coords = gmsh.model.mesh.getNodes()[0:2]
    node_coords = np.reshape(node_coords, (-1, 3))

    sorted_ids = node_tags.argsort()
    mesh_parts.nodes.tags = node_tags[sorted_ids]
    mesh_parts.nodes.coords = node_coords[sorted_ids]

    gmsh.write(str(Path("output") / "test_mesh.msh"))

    gmsh.finalize()


    return mesh_parts

if __name__ == "__main__":
    generate_mesh()