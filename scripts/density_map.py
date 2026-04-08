import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET


def create_density_vtu(feb_path: Path, output_dir: Path):
    """
    Extract mesh and density data from a .feb file and export as a VTU file for visualization.

    Reads the 3D volumetric mesh (nodes and elements) from the FEBio input file,
    extracts the density field from the MeshData section, and
    writes everything to a VTK Unstructured Grid (.vtu) file. This allows visualization
    of the heterogeneous density distribution in ParaView or other VTK-compatible tools.

    Args:
        feb_path: Path to the .feb input file
        output_dir: Directory where the output .vtu file will be saved

    Note:
        - Only includes adipose and glandular tissue volumes (not skin surface)
        - Uses default density of 911.0 kg/m³ for any nodes without explicit values
        - Supports both tet4 and tet10 element types
    """
    # Parse the .feb file
    tree = ET.parse(feb_path)
    root = tree.getroot()

    # Extract nodes
    nodes_elem = root.find(".//Nodes[@name='Object01']")
    node_coords = []
    node_ids = []
    node_id_to_idx = {}  # Map node ID (from .feb) to array index

    for idx, node in enumerate(nodes_elem.findall("node")):
        node_id = int(node.get("id"))
        coords = [float(x) for x in node.text.split(",")]
        node_ids.append(node_id)
        node_coords.append(coords)
        node_id_to_idx[node_id] = idx

    node_coords = np.array(node_coords)

    # Extract volumetric elements (tetrahedral cells)
    elements = []
    cell_types = []

    # Get adipose tissue elements
    adipose_elems = root.find(".//Elements[@name='adipose_part']")
    elem_type = adipose_elems.get("type")

    # Map FEBio element types to VTK cell type codes
    vtk_cell_type = {"tet4": 10, "tet10": 24}[elem_type]
    nodes_per_elem = {"tet4": 4, "tet10": 10}[elem_type]

    for elem in adipose_elems.findall("elem"):
        node_list = [int(x) for x in elem.text.split(",")]
        # Convert node IDs to indices for vtk
        node_indices = [node_id_to_idx[nid] for nid in node_list]
        elements.append(node_indices)
        cell_types.append(vtk_cell_type)

    # Get glandular tissue elements
    glandular_elems = root.find(".//Elements[@name='glandular_part']")
    for elem in glandular_elems.findall("elem"):
        node_list = [int(x) for x in elem.text.split(",")]
        node_indices = [node_id_to_idx[nid] for nid in node_list]
        elements.append(node_indices)
        cell_types.append(vtk_cell_type)

    # Extract density values from MeshData section
    meshdata = root.find(".//MeshData")
    nodedata = meshdata.find(".//NodeData[@name='density_map']")

    density_lookup = {}
    for node in nodedata.findall("node"):
        lid = int(node.get("lid"))  # Node ID in .feb file
        value = float(node.text)    # Density value in kg/m³
        density_lookup[lid] = value

    # Map density values to node array (use default 911.0 if missing)
    density_values = np.array(
        [density_lookup.get(nid, 911.0) for nid in node_ids])

    # Build VTU XML structure (VTK Unstructured Grid format)
    vtu_root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1",
                          byte_order="LittleEndian")
    ugrid = ET.SubElement(vtu_root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(len(node_coords)),
                          NumberOfCells=str(len(elements)))

    # Node coordinates (Points)
    points = ET.SubElement(piece, "Points")
    points_array = ET.SubElement(points, "DataArray",
                                 type="Float32",
                                 NumberOfComponents="3",
                                 format="ascii")
    points_str = "\n".join([f"{x} {y} {z}" for x, y, z in node_coords])
    points_array.text = "\n" + points_str + "\n"

    # Element connectivity (Cells)
    cells = ET.SubElement(piece, "Cells")

    # Connectivity: node indices for each element
    connectivity = ET.SubElement(cells, "DataArray",
                                 type="Int32",
                                 Name="connectivity",
                                 format="ascii")
    conn_str = "\n".join([" ".join(map(str, elem)) for elem in elements])
    connectivity.text = "\n" + conn_str + "\n"

    # Offsets: cumulative count of nodes (for indexing into connectivity array)
    offsets = ET.SubElement(cells, "DataArray",
                            type="Int32",
                            Name="offsets",
                            format="ascii")
    offset_values = [nodes_per_elem * (i + 1) for i in range(len(elements))]
    offsets.text = "\n" + " ".join(map(str, offset_values)) + "\n"

    # Types: VTK cell type code for each element
    types = ET.SubElement(cells, "DataArray",
                          type="UInt8",
                          Name="types",
                          format="ascii")
    types.text = "\n" + " ".join(map(str, cell_types)) + "\n"

    # Nodal scalar data (density field)
    point_data = ET.SubElement(piece, "PointData", Scalars="density_map")
    density_array = ET.SubElement(point_data, "DataArray",
                                  type="Float32",
                                  Name="density_map",
                                  format="ascii")
    density_str = "\n".join([f"{v:.6f}" for v in density_values])
    density_array.text = "\n" + density_str + "\n"

    # Write VTU file
    output_file = output_dir / "density_map.vtu"
    tree = ET.ElementTree(vtu_root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"✓ Created {output_file}")
    print(f"  Nodes: {len(node_coords)}")
    print(f"  Elements: {len(elements)}")
    print(
        f"  Density range: [{density_values.min():.2f}, {density_values.max():.2f}]")

    return output_file


# Usage:
if __name__ == "__main__":
    feb_path = Path("/Users/ryanengels/Documents/limebv-ews_fem_pipeline-e5fe835fd1bf/gradient_lobules/gradient_lobules.feb")
    # feb_path = Path("/Users/ryanengels/Documents/limebv-ews_fem_pipeline-e5fe835fd1bf/heterogenous_1/all_default_settings.feb")
    output_dir = feb_path.parent / "output"

    create_density_vtu(feb_path, output_dir)