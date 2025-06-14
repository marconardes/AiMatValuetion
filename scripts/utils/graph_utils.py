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
        electronegativity_val = None
        try:
            electronegativity_val = site.specie.X
        except AttributeError:
            try:
                element = Element(site.specie.symbol) # Ensure Element is imported
                electronegativity_val = element.X
            except Exception: # Broad exception for Element lookup issues
                electronegativity_val = 0.0 # Default value
                print(f"Warning: Could not determine electronegativity for site {i} ({site.specie.symbol}). Defaulting to 0.0.")

        final_en = None
        if electronegativity_val is not None:
            try:
                final_en = float(electronegativity_val)
            except (TypeError, ValueError): # Handle if it's not float-convertible
                final_en = 0.0 # Fallback
                print(f"Warning: Electronegativity value '{electronegativity_val}' for site {i} could not be converted to float. Defaulting to 0.0.")
        # If electronegativity_val was None, final_en remains None

        node = {
            "atomic_number": int(site.specie.number),
            "electronegativity": final_en,
            "original_site_index": int(i) # i from enumerate is already int
        }
        nodes.append(node)

    # 2. Determine edges
    cutoff = 3.0  # Define the cutoff radius
    all_neighbors = structure.get_all_neighbors(r=cutoff) # Still useful for candidate pairs
    added_edges = set()

    for i, site_neighbors_of_i in enumerate(all_neighbors):
        for neighbor_info in site_neighbors_of_i:
            j_index = neighbor_info.index

            edge_key = tuple(sorted((i, j_index)))

            if i != j_index and edge_key not in added_edges:
                # Get the true shortest distance between original sites i and j_index
                true_shortest_distance = structure.get_distance(i, j_index)

                # Only add if this shortest distance is within the cutoff
                if true_shortest_distance <= cutoff:
                    edges.append({
                        "source_node_index": int(edge_key[0]),
                        "target_node_index": int(edge_key[1]),
                        "distance": float(true_shortest_distance)
                    })
                    added_edges.add(edge_key)

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
