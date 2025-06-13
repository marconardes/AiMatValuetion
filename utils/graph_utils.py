# utils/graph_utils.py
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element # For electronegativity, though site.specie.X should work

def structure_to_graph(structure: Structure) -> dict:
    """
    Converts a pymatgen Structure object into a graph representation.

    Args:
        structure (pymatgen.core.structure.Structure): Input crystal structure.

    Returns:
        dict: A dictionary containing graph nodes, edges, and their counts.
              Keys: "nodes", "edges", "num_nodes", "num_edges".
    """
    nodes = []
    edges = []

    # 1. Create nodes
    for i, site in enumerate(structure):
        electronegativity = None
        try:
            electronegativity = site.specie.X
        except AttributeError:
            # Fallback if .X is not available or specie is not a standard element
            # For pymatgen > 2023.x.x, Element(site.specie.symbol).X might be more robust
            # but site.specie.X should work for common elements.
            try:
                element = Element(site.specie.symbol)
                electronegativity = element.X
            except Exception: # Broad exception to catch any issue with Element lookup
                electronegativity = 0.0 # Default value if electronegativity cannot be found
                print(f"Warning: Could not determine electronegativity for site {i} ({site.specie.symbol}). Defaulting to 0.0.")


        node = {
            "atomic_number": site.specie.number,
            "electronegativity": electronegativity,
            "original_site_index": i
        }
        nodes.append(node)

    # 2. Determine edges
    # Using a cutoff radius of 3.0 Angstroms as specified
    # get_all_neighbors returns a list of lists; each inner list contains Neighbor objects for a site
    all_neighbors = structure.get_all_neighbors(r=3.0)

    for i, site_neighbors in enumerate(all_neighbors):
        for neighbor_info in site_neighbors:
            j_index = neighbor_info.index # neighbor_info is a Neighbor object, .index is the site index

            # To avoid duplicate edges (i,j) and (j,i) and self-loops (i,i)
            if j_index > i:
                edge = {
                    "source_node_index": i,
                    "target_node_index": j_index,
                    "distance": neighbor_info.nn_distance
                }
                edges.append(edge)

    return {
        "nodes": nodes,
        "edges": edges,
        "num_nodes": len(nodes),
        "num_edges": len(edges)
    }

if __name__ == '__main__':
    # Example Usage (requires pymatgen to be installed)
    # You would typically run this from a context where pymatgen is available
    # and you have a Structure object.

    # Create a simple Si structure for testing
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    # Example: Silicon structure
    # For a real test, ensure you have a valid Structure object.
    # This is a placeholder for testing the function if run directly.
    try:
        print("Attempting to create a sample Si structure for testing structure_to_graph...")
        lattice = Lattice.cubic(5.43)  # Silicon lattice constant
        species = ["Si", "Si"]
        coords = [[0, 0, 0], [0.25, 0.25, 0.25]] # Diamond structure basis
        si_structure = Structure(lattice, species, coords)

        print(f"Sample structure created: {si_structure.formula}")

        graph_data = structure_to_graph(si_structure)
        print("\nGraph Data:")
        print(f"Number of nodes: {graph_data['num_nodes']}")
        for node in graph_data['nodes']:
            print(node)

        print(f"\nNumber of edges: {graph_data['num_edges']}")
        for edge in graph_data['edges']:
            print(edge)

    except ImportError:
        print("Pymatgen not found. Skipping example usage in structure_to_graph.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
