# utils/schema.py

# DATA_SCHEMA defines the structure and description of data fetched from Materials Project
# and some derived properties. It's used as a reference in fetching and processing stages.
DATA_SCHEMA = {
    "material_id": "Materials Project ID (e.g., mp-123)",
    "supercon_composition": "Original SuperCon composition string (e.g., 'Ba0.6K0.4Fe2As2')",
    "cif_string": "Crystallographic Information File as a string",
    "band_gap_mp": "Band Gap (eV) directly from MP",
    "formation_energy_per_atom_mp": "Formation Energy (eV/atom) directly from MP",
    "dos_object_mp": "The CompleteDOS object from MP (temporary, for processing)",
    "is_metal": "Boolean (True if band_gap_mp == 0, False otherwise)",
    "dos_at_fermi": "Value of total DOS at Fermi level (eV⁻¹). 0 or N/A for insulators.",
    "formula_pretty": "Reduced chemical formula (e.g., 'Si', 'GaAs')",
    "num_elements": "Number of distinct elements in the material",
    "elements": "Comma-separated string of elements (e.g., 'Ga,As')",
    "density_pg": "Density (g/cm³) calculated by pymatgen",
    "volume_pg": "Cell Volume (Å³) calculated by pymatgen",
    "volume_per_atom_pg": "Volume per atom (Å³/atom) calculated by pymatgen",
    "spacegroup_number_pg": "Space Group Number calculated by pymatgen",
    "crystal_system_pg": "Crystal System (e.g., 'cubic', 'tetragonal') from pymatgen",
    "lattice_a_pg": "Lattice parameter a (Å)",
    "lattice_b_pg": "Lattice parameter b (Å)",
    "lattice_c_pg": "Lattice parameter c (Å)",
    "lattice_alpha_pg": "Lattice angle alpha (°)",
    "lattice_beta_pg": "Lattice angle beta (°)",
    "lattice_gamma_pg": "Lattice angle gamma (°)",
    "num_sites_pg": "Number of atomic sites in the unit cell from pymatgen",
    "target_band_gap": "Final Band Gap (eV) for ML",
    "target_formation_energy": "Final Formation Energy (eV/atom) for ML",
    "target_is_metal": "Final 'is_metal' boolean for ML",
    "target_dos_at_fermi": "Final 'dos_at_fermi' for ML",
    "graph_nodes": "List of dictionaries, each representing an atom with its features (atomic_number, electronegativity, etc.).",
    "graph_edges": "List of dictionaries, each representing a bond/connection with its features (e.g., source/target node indices, distance).",
    "num_graph_nodes": "Number of nodes (atoms) in the graph.",
    "num_graph_edges": "Number of edges (bonds/connections) in the graph."
}

# MANUAL_ENTRY_CSV_HEADERS defines the expected column order and names for CSV files
# that the ManualEntryTab in the GUI might interact with (e.g., saving new entries).
MANUAL_ENTRY_CSV_HEADERS = [
    "material_id", "band_gap_mp", "formation_energy_per_atom_mp", "is_metal", "dos_at_fermi",
    "formula_pretty", "num_elements", "elements", "density_pg", "volume_pg", "volume_per_atom_pg",
    "spacegroup_number_pg", "crystal_system_pg", "lattice_a_pg", "lattice_b_pg", "lattice_c_pg",
    "lattice_alpha_pg", "lattice_beta_pg", "lattice_gamma_pg", "num_sites_pg",
    "num_graph_nodes", "num_graph_edges",
    "target_band_gap", "target_formation_energy", "target_is_metal", "target_dos_at_fermi"
]
