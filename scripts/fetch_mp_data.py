import os
import json
import warnings
from utils.config_loader import load_config
from mp_api.client import MPRester
from pymatgen.core import Composition, Structure # Ensure Structure is imported

# DATA_SCHEMA is now imported from utils.schema

# This script fetches data from the Materials Project (MP).
# Usage of the MP API is considered optional for this project.
# If an MP_API_KEY is not provided in config.yml or environment variables,
# the script will attempt to proceed with anonymous access via the mp-api client,
# which may have limitations or fail for queries requiring authentication.

def fetch_data(max_total_materials_arg=50): # Renamed arg to avoid conflict with loaded var
    full_config = load_config() # Use the new centralized loader

    if not full_config: # load_config now returns {} on failure/not found
        warnings.warn("Failed to load or parse config.yml. Using default script parameters or environment variables.", UserWarning)
        mp_api_key_from_config = None
        fetch_config_params = {} # Use a different name to avoid confusion with outer scope 'fetch_config' if any
    else:
        mp_api_key_from_config = full_config.get("mp_api_key")
        fetch_config_params = full_config.get("fetch_data", {})

    # Get API key: Prioritize config, then environment variable
    api_key = mp_api_key_from_config if mp_api_key_from_config and mp_api_key_from_config != "YOUR_MP_API_KEY" else os.environ.get("MP_API_KEY")

    if not api_key:
        warnings.warn("MP_API_KEY not found in config.yml or environment variables, or is set to the placeholder. Proceeding with anonymous access if possible.", UserWarning)
        # Depending on mp_api strictness, this might still fail later.

    # Get other parameters from config, with defaults from original script or function arg
    # Retrieve max_total_materials from config, using function arg as ultimate fallback
    max_total_materials_config = fetch_config_params.get('max_total_materials', max_total_materials_arg)
    output_filename = fetch_config_params.get('output_filename', "data/mp_raw_data.json") # Default from original script

    # Define criteria sets: Prioritize config, then script defaults
    default_criteria_sets = [
        {"target_n_elements": 2, "limit_per_set": 20, "description": "binary Fe compounds"},
        {"target_n_elements": 3, "limit_per_set": 20, "description": "ternary Fe compounds"},
        {"target_n_elements": 4, "limit_per_set": 10, "description": "quaternary Fe compounds"},
        {"target_n_elements": 1, "limit_per_set": 5, "description": "elemental Fe"},
    ]
    criteria_sets = fetch_config_params.get('criteria_sets', default_criteria_sets)


    raw_materials_data = []

    # Fields for the initial summary search
    summary_fields = [
        "material_id", "formula_pretty", "nelements",
        "band_gap", "formation_energy_per_atom"
    ]

    # Cache for initial summary query results
    summary_docs_cache = None

    # Attempting to connect to Materials Project.
    # If api_key is None (due to not being found in config/env or being the placeholder),
    # mp-api client might attempt anonymous access or raise an error if queries require authentication.
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
        fetchAll = False
        if max_total_materials_config == -5:
            fetchAll = True
            print("Config 'max_total_materials' is -5. Fetching all matching materials, ignoring limits per set and overall total.")

        for criteria_set in criteria_sets:
            if not fetchAll and len(raw_materials_data) >= max_total_materials_config:
                print(f"Reached overall target of {len(raw_materials_data)}/{max_total_materials_config} materials. Stopping.")
                break

            target_n_elements = criteria_set["target_n_elements"]
            limit_per_set = criteria_set["limit_per_set"]
            description = criteria_set["description"]

            print(f"\nProcessing for {description} (target {target_n_elements} elements)...")

            materials_added_this_set = 0
            for summary_doc in summary_docs_cache:
                if not fetchAll and len(raw_materials_data) >= max_total_materials_config: break
                if not fetchAll and materials_added_this_set >= limit_per_set: break

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
        # output_filename is now sourced from config or default at the start of the function
        print(f"Saving raw data to {output_filename}...")
        try:
            with open(output_filename, 'w') as f:
                json.dump(raw_materials_data, f, indent=4)
            print(f"Successfully saved data for {len(raw_materials_data)} materials to {output_filename}.")
        except Exception as e:
            print(f"Error saving data to JSON: {e}")
    else:
        print("No data collected to save.")

if __name__ == "__main__":
    # The fetch_data function now loads its own config, so no need to pass max_total_materials from here
    # The fetch_data function now loads its own config.
    # Pass the default max_total_materials if you want it to be the ultimate fallback.
    fetch_data(max_total_materials_arg=50) # Original default was 50
