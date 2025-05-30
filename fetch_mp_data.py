import os
import json
import warnings
from mp_api.client import MPRester
from pymatgen.core import Composition, Structure # Ensure Structure is imported

# DATA_SCHEMA definition remains the same
DATA_SCHEMA = {
    "material_id": "Materials Project ID (e.g., mp-123)",
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
    "target_dos_at_fermi": "Final 'dos_at_fermi' for ML"
}

def fetch_data(max_total_materials=50):
    api_key = os.environ.get("MP_API_KEY")
    if api_key is None:
        warnings.warn("MP_API_KEY environment variable not set. Proceeding with anonymous access.")

    raw_materials_data = []

    # Fields for the initial summary search
    summary_fields = [
        "material_id", "formula_pretty", "nelements",
        "band_gap", "formation_energy_per_atom"
    ]

    # Define criteria sets for different numbers of elements for Python-side filtering
    criteria_sets = [
        {"target_n_elements": 2, "limit_per_set": 20, "description": "binary Fe compounds"},
        {"target_n_elements": 3, "limit_per_set": 20, "description": "ternary Fe compounds"},
        {"target_n_elements": 4, "limit_per_set": 10, "description": "quaternary Fe compounds"},
        {"target_n_elements": 1, "limit_per_set": 5, "description": "elemental Fe"},
    ]

    # Cache for initial summary query results
    summary_docs_cache = None

    with MPRester(api_key=api_key) as mpr:
        print("Fetching initial candidate materials (Fe-containing with band_gap data)...")
        try:
            # Step 1: Initial query using summary.search
            # Query for Fe-containing materials that have a band_gap calculated (0 to 100 eV is a wide range)
            # The mp-api client handles pagination by default. Default limit is 1000.
            summary_docs_cache = mpr.materials.summary.search(
                elements=["Fe"],
                band_gap=(0, 100), # Filter for materials with calculated band gaps
                fields=summary_fields
            )
            if not summary_docs_cache:
                summary_docs_cache = [] # Ensure it's an empty list if None
            print(f"Found {len(summary_docs_cache)} initial Fe-containing candidates with band gap data.")
        except Exception as e:
            warnings.warn(f"API call for initial summary search failed: {e}")
            summary_docs_cache = [] # Ensure it's an empty list on failure

        if not summary_docs_cache:
            print("No initial candidate materials found. Exiting.")
            return

        # Step 2: Iterate through criteria sets, filter candidates, and fetch details
        for criteria_set in criteria_sets:
            if len(raw_materials_data) >= max_total_materials:
                print(f"Reached overall target of {len(raw_materials_data)}/{max_total_materials} materials. Stopping.")
                break

            target_n_elements = criteria_set["target_n_elements"]
            limit_per_set = criteria_set["limit_per_set"]
            description = criteria_set["description"]

            print(f"\nProcessing for {description} (target {target_n_elements} elements)...")

            materials_added_this_set = 0
            for summary_doc in summary_docs_cache:
                if len(raw_materials_data) >= max_total_materials: break
                if materials_added_this_set >= limit_per_set: break

                # Python-side filtering for number of elements
                num_doc_elements = summary_doc.nelements if hasattr(summary_doc, 'nelements') and summary_doc.nelements is not None \
                                   else len(Composition(summary_doc.formula_pretty).elements)

                if num_doc_elements != target_n_elements:
                    continue

                material_id = str(summary_doc.material_id)
                print(f"  Fetching details for {material_id} ({summary_doc.formula_pretty})...")

                try:
                    # Fetch structure
                    structure = mpr.get_structure_by_material_id(material_id)
                    cif_string = structure.to(fmt="cif") if structure else None

                    # Fetch DOS
                    dos = mpr.get_dos_by_material_id(material_id)
                    dos_dict = dos.as_dict() if dos else None

                    material_entry = {
                        "material_id": material_id,
                        "cif_string": cif_string,
                        "band_gap_mp": summary_doc.band_gap,
                        "formation_energy_per_atom_mp": summary_doc.formation_energy_per_atom,
                        "dos_object_mp": dos_dict,
                        "formula_pretty_mp": summary_doc.formula_pretty,
                        "nelements_mp": num_doc_elements
                    }
                    raw_materials_data.append(material_entry)
                    materials_added_this_set += 1

                except Exception as e:
                    warnings.warn(f"  Failed to fetch details for {material_id}: {e}")

            print(f"Added {materials_added_this_set} materials for {description}. Total collected: {len(raw_materials_data)}")

    print(f"\nTotal materials collected after all processing: {len(raw_materials_data)}")

    if raw_materials_data:
        filename = "mp_raw_data.json"
        print(f"Saving raw data to {filename}...")
        try:
            with open(filename, 'w') as f:
                json.dump(raw_materials_data, f, indent=4)
            print(f"Successfully saved data for {len(raw_materials_data)} materials to {filename}.")
        except Exception as e:
            print(f"Error saving data to JSON: {e}")
    else:
        print("No data collected to save.")

if __name__ == "__main__":
    fetch_data(max_total_materials=50)
