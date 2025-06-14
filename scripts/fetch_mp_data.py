import os
import json
import pandas as pd # Add this if not already present at the top
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

def get_supercon_compositions(csv_path="data/supercon_processed.csv"):
    """
    Reads unique compositions from the supercon_processed.csv file.
    """
    if not os.path.exists(csv_path):
        warnings.warn(f"Source CSV file for compositions not found: {csv_path}", UserWarning)
        # Depending on desired behavior, could try to run a prerequisite script here
        # For now, just return empty list if not found after warning.
        return []
    try:
        df = pd.read_csv(csv_path)
        if 'composition' not in df.columns:
            warnings.warn(f"'composition' column not found in {csv_path}. Cannot extract compositions.", UserWarning)
            return []
        unique_comps = df['composition'].unique().tolist()
        print(f"Read {len(unique_comps)} unique compositions from {csv_path}")
        return unique_comps
    except pd.errors.EmptyDataError:
        warnings.warn(f"Warning: The file {csv_path} is empty.", UserWarning)
        return []
    except Exception as e:
        warnings.warn(f"Error reading or processing {csv_path}: {e}", UserWarning)
        return []

def select_best_mp_entry(mp_entries_list):
    """
    Selects the best MP entry from a list based on stability criteria.
    mp_entries_list is a list of dictionaries (from Pydantic model_dump()).
    """
    if not mp_entries_list:
        return None

    valid_entries = []
    for entry_dict in mp_entries_list:
        # Basic check: ensure it's a dict and not deprecated (already filtered but good practice)
        if not isinstance(entry_dict, dict) or entry_dict.get('deprecated', False):
            continue
        valid_entries.append(entry_dict)

    if not valid_entries:
        return None

    # Sort by energy_above_hull (lower is better, None treated as high)
    # Then by formation_energy_per_atom (lower is better, None treated as high)
    infinity = float('inf')
    valid_entries.sort(key=lambda x: (
        x.get('energy_above_hull', infinity) if x.get('energy_above_hull') is not None else infinity,
        x.get('formation_energy_per_atom', infinity) if x.get('formation_energy_per_atom') is not None else infinity,
        # material_id can be a string like "mp-1234", so direct numeric sort isn't ideal without parsing.
        # For simplicity, if energies are identical, the first one after sorting by energy is taken.
    ))

    # Log which entry was chosen (optional)
    # best_choice = valid_entries[0]
    # print(f"    Selected MP Entry: {best_choice.get('material_id')} "
    #       f"(EAH: {best_choice.get('energy_above_hull', 'N/A')}, "
    #       f"Form.E/atom: {best_choice.get('formation_energy_per_atom', 'N/A')})")

    return valid_entries[0]

def fetch_data(max_total_materials_arg=50): # Renamed arg to avoid conflict with loaded var
    full_config = load_config() # Use the new centralized loader

    if not full_config: # load_config now returns {} on failure/not found
        warnings.warn("Failed to load or parse config.yml. Using default script parameters or environment variables.", UserWarning)
        mp_api_key_from_config = None
        fetch_config_params = {} # Use a different name to avoid confusion with outer scope 'fetch_config' if any
    else:
        mp_api_key_from_config = full_config.get("mp_api_key")
        fetch_config_params = full_config.get("fetch_data", {})

    supercon_compositions_csv_path = fetch_config_params.get('supercon_processed_csv_path', "data/supercon_processed.csv") # Make path configurable
    target_compositions = get_supercon_compositions(supercon_compositions_csv_path)

    output_filename = fetch_config_params.get('output_filename', "data/mp_raw_data_from_supercon.json") # Consider a new default output name
    if not target_compositions:
        print("No target compositions loaded from CSV. Exiting.")
        # Ensure output JSON is empty or not written if no compositions
        if os.path.exists(output_filename): # If an old file exists, maybe clear it or leave as is based on desired behavior.
            # For now, we'll just proceed and it will save an empty list if raw_materials_data remains empty.
            pass
        with open(output_filename, 'w') as f:
             json.dump([], f) # Write empty list if no targets
        print(f"Saved empty list to {output_filename} as no target compositions were found.")
        return # Exit if no compositions to process

    # Get API key: Prioritize config, then environment variable
    api_key = mp_api_key_from_config if mp_api_key_from_config and mp_api_key_from_config != "YOUR_MP_API_KEY" else os.environ.get("MP_API_KEY")

    if not api_key:
        warnings.warn("MP_API_KEY not found in config.yml or environment variables, or is set to the placeholder. Proceeding with anonymous access if possible.", UserWarning)
        # Depending on mp_api strictness, this might still fail later.

    # Get other parameters from config, with defaults from original script or function arg
    # Retrieve max_total_materials from config, using function arg as ultimate fallback
    max_total_materials_config = fetch_config_params.get('max_total_materials', max_total_materials_arg)
    # output_filename = fetch_config_params.get('output_filename', "data/mp_raw_data.json") # Default from original script
    # This line is now handled by the new block inserted above.

    # criteria_sets related variables removed as they are no longer used by the primary logic.

    # Fields for the initial summary search
    summary_fields = [
        "material_id", "formula_pretty", "nelements",
        "band_gap", "formation_energy_per_atom", "energy_above_hull", # Added energy_above_hull
        "volume", "density", "deprecated", "theoretical", "experimental_description" # Added more fields
    ]

    # summary_docs_cache removed as it's no longer used.

    # Attempting to connect to Materials Project.
    # If api_key is None (due to not being found in config/env or being the placeholder),
    # mp-api client might attempt anonymous access or raise an error if queries require authentication.
    with MPRester(api_key=api_key) as mpr:
        # Old Fe-based fetching logic (summary_docs_cache, criteria_sets loop) fully removed.
        # New logic starts here.
        all_materials_data_map = {} # This will store the final data

        # max_total_materials_config needs to be re-evaluated in this new context.
        # For now, we'll fetch for all target_compositions.
        # A global limit might still be useful but applied differently.
        # fetchAll variable might also be re-evaluated or removed if not fitting the new logic.

        # Ensure pymatgen.core.Composition is imported at the top of the file
        # from pymatgen.core import Composition

        processed_composition_count = 0

        for supercon_comp_str in target_compositions:
            # Optional: Implement a global material limit if needed
            # if not fetchAll and some_overall_counter >= max_total_materials_config:
            #     print(f"Reached overall material limit. Stopping.")
            #     break

            print(f"Processing SuperCon composition: {supercon_comp_str}")
            try:
                comp_obj = Composition(supercon_comp_str)
                elements_list = sorted([el.symbol for el in comp_obj.elements]) # Sorted list for consistent chemsys
                chemical_system = "-".join(elements_list)
                reduced_formula = comp_obj.reduced_formula # For potential filtering

                print(f"  Querying MP for chemical system: {chemical_system} (derived from {supercon_comp_str})...")

                # Query using chemsys. This is generally more robust for matching.
                current_mp_entries = mpr.materials.search(
                    chemsys=chemical_system,
                    fields=summary_fields
                )

                relevant_entries = []
                if current_mp_entries:
                    print(f"  Found {len(current_mp_entries)} MP entries for chemical system {chemical_system}.")

                    # Client-side filtering for entries that are not deprecated and are theoretical (calculated)
                    # Also, try to match the reduced formula as a heuristic
                    for doc in current_mp_entries:
                        if hasattr(doc, 'deprecated') and doc.deprecated:
                            continue # Skip deprecated materials
                        # if hasattr(doc, 'theoretical') and not doc.theoretical: # Only keep theoretical for now, unless specified otherwise
                        #     continue # Skip experimental if we want calculated entries for GNNs. Adjust if needed.

                        # Match reduced formula as a primary filter if possible
                        if hasattr(doc, 'formula_pretty') and Composition(doc.formula_pretty).reduced_formula == reduced_formula:
                            relevant_entries.append(doc.model_dump()) # Use model_dump() for pydantic models
                        # If no exact reduced formula match, we might consider all non-deprecated theoretical entries from the chemsys later
                        # For now, only take exact reduced formula matches.

                    # if relevant_entries: # Old logic
                    #     print(f"    Selected {len(relevant_entries)} entries matching reduced formula '{reduced_formula}' and filters.")
                    #     all_materials_data_map[supercon_comp_str] = relevant_entries # Old logic
                    if relevant_entries:
                        best_entry_dict = select_best_mp_entry(relevant_entries) # Call the new selection function
                        if best_entry_dict: # best_entry_dict is a dictionary from model_dump()
                            print(f"    Selected best MP entry: {best_entry_dict.get('material_id')} "
                                  f"(EAH: {best_entry_dict.get('energy_above_hull', 'N/A')}, "
                                  f"Form.E/atom: {best_entry_dict.get('formation_energy_per_atom', 'N/A')})")

                            # Now fetch detailed data for this selected entry
                            material_id_str = best_entry_dict.get('material_id') # material_id should be a string like "mp-xxxx"

                            if not material_id_str:
                                warnings.warn(f"    Material ID missing in selected entry for {supercon_comp_str}. Skipping detail fetch.")
                                # Store the summary data anyway, or decide to store None if details are critical
                                all_materials_data_map[supercon_comp_str] = best_entry_dict
                                # continue # Continue to next supercon_comp_str # This was a bug in prompt, continue is outside this if block
                            else:
                                try:
                                    print(f"      Fetching structure for {material_id_str}...")
                                    structure_obj = mpr.get_structure_by_material_id(material_id_str) # Fetches Structure object
                                    cif_string = structure_obj.to(fmt="cif") if structure_obj else None
                                    best_entry_dict['cif_string_mp'] = cif_string # Add to the dict
                                    if not cif_string:
                                        warnings.warn(f"      Could not get CIF string for {material_id_str}")

                                    print(f"      Fetching DOS for {material_id_str}...")
                                    dos_obj = mpr.get_dos_by_material_id(material_id_str) # Fetches DOS object
                                    dos_dict_data = dos_obj.as_dict() if dos_obj else None
                                    best_entry_dict['dos_object_mp'] = dos_dict_data # Add to the dict
                                    if not dos_dict_data:
                                        warnings.warn(f"      Could not get DOS for {material_id_str}")

                                except Exception as detail_exc:
                                    warnings.warn(f"      Error fetching details (structure/DOS) for {material_id_str}: {str(detail_exc)[:200]}")
                                    # Decide if you want to keep the summary data even if details fail
                                    if 'cif_string_mp' not in best_entry_dict: # if it failed before adding anything
                                        best_entry_dict['cif_string_mp'] = None
                                    if 'dos_object_mp' not in best_entry_dict:
                                        best_entry_dict['dos_object_mp'] = None

                            all_materials_data_map[supercon_comp_str] = best_entry_dict # Store the updated dict with details

                        else:
                            # This case means select_best_mp_entry returned None
                            print(f"    No suitable best entry found by select_best_mp_entry for {supercon_comp_str}.")
                            all_materials_data_map[supercon_comp_str] = None
                    else:
                        # If no exact formula match, consider a fallback or log as no suitable match
                        # For now, if no exact match, we store None, this can be refined in selection step.
                        print(f"    No entries matched reduced formula '{reduced_formula}' and filters for {chemical_system}.")
                        all_materials_data_map[supercon_comp_str] = None
                else:
                    all_materials_data_map[supercon_comp_str] = None
                    print(f"  No MP entries found for chemical system {chemical_system}.")

            except Exception as e:
                warnings.warn(f"  Error processing or querying for SuperCon composition {supercon_comp_str}: {str(e)[:500]}")
                all_materials_data_map[supercon_comp_str] = None

            processed_composition_count += 1
            # Optional: add a small delay to be polite to the API
            # import time
            # time.sleep(0.05) # 50 ms delay

        print(f"\nFinished querying MP for {processed_composition_count} SuperCon compositions.")

    # print(f"\nTotal materials collected after all processing: {len(raw_materials_data)}") # Old message

    if all_materials_data_map: # Check if the map is not empty
        print(f"Saving raw data to {output_filename}...")
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            with open(output_filename, 'w') as f:
                json.dump(all_materials_data_map, f, indent=4) # Save the dictionary

            # Calculate total number of MP materials fetched (now it's 1 per SuperCon comp if found)
            # total_mp_materials_fetched = sum(len(v) for v in all_materials_data_map.values() if v is not None) # Old way
            successfully_mapped_supercon_comps = sum(1 for v in all_materials_data_map.values() if v is not None)
            print(f"Successfully saved data for {successfully_mapped_supercon_comps} SuperCon compositions to {output_filename}.")
        except Exception as e:
            print(f"Error saving data to JSON: {e}")
    else:
        print("No data collected to save.")

if __name__ == "__main__":
    # The fetch_data function now loads its own config, so no need to pass max_total_materials from here
    # The fetch_data function now loads its own config.
    # Pass the default max_total_materials if you want it to be the ultimate fallback.
    fetch_data(max_total_materials_arg=50) # Original default was 50
