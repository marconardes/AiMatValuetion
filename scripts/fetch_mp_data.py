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
    Reads unique compositions and their critical temperatures (tc)
    from the supercon_processed.csv file.
    Returns a dictionary mapping composition string to tc value.
    """
    if not os.path.exists(csv_path):
        warnings.warn(f"Source CSV file for compositions and tc not found: {csv_path}", UserWarning)
        return {}
    try:
        df = pd.read_csv(csv_path)
        if 'composition' not in df.columns or 'tc' not in df.columns:
            warnings.warn(f"Required columns 'composition' and/or 'tc' not found in {csv_path}. Cannot extract composition-tc map.", UserWarning)
            return {}

        # Drop rows where 'composition' is NaN/empty, as they are not useful
        df.dropna(subset=['composition'], inplace=True)
        if df.empty:
            warnings.warn(f"No valid compositions found in {csv_path} after dropping NaN compositions.", UserWarning)
            return {}

        # Handle duplicate compositions: keep the first occurrence
        df_unique_comps = df.drop_duplicates(subset=['composition'], keep='first')

        comp_to_tc_map = {}
        for index, row in df_unique_comps.iterrows():
            comp = str(row['composition'])
            tc_val = row['tc']
            try:
                # Ensure tc_val is a float or None if it's NaN or unconvertible
                if pd.isna(tc_val):
                    comp_to_tc_map[comp] = None
                else:
                    comp_to_tc_map[comp] = float(tc_val)
            except ValueError:
                warnings.warn(f"Could not convert tc value '{tc_val}' to float for composition '{comp}'. Storing as None.", UserWarning)
                comp_to_tc_map[comp] = None

        print(f"Read {len(comp_to_tc_map)} unique compositions and their tc values from {csv_path}")
        return comp_to_tc_map
    except pd.errors.EmptyDataError:
        warnings.warn(f"Warning: The file {csv_path} is empty.", UserWarning)
        return {}
    except Exception as e:
        warnings.warn(f"Error reading or processing {csv_path}: {e}", UserWarning)
        return {}

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

    supercon_compositions_csv_path = fetch_config_params.get('supercon_processed_csv_path', "data/supercon_processed.csv")
    comp_to_tc_map = get_supercon_compositions(supercon_compositions_csv_path) # Renamed variable

    output_filename = fetch_config_params.get('output_filename', "data/mp_raw_data.json")
    if not comp_to_tc_map: # Check if the map is empty
        print("No composition-tc map loaded from CSV. Exiting.")
        # Ensure output JSON is empty (dictionary for map) or not written
        with open(output_filename, 'w') as f:
             json.dump({}, f) # Write empty dict if no targets
        print(f"Saved empty dictionary to {output_filename} as no composition-tc map was found.")
        return

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
    initial_search_fields = [
        "material_id", "formula_pretty", "nelements",
        "composition_reduced", "chemsys", "deprecated"
        # "elements" # Also available and could be useful for precise chemsys construction if needed
    ]

    detailed_summary_fields = [
        "material_id", "formula_pretty", "nelements", "band_gap",
        "formation_energy_per_atom", "energy_above_hull", "volume",
        "density", "deprecated", "theoretical",
        # Add any other fields that mpr.materials.summary.search supports and are needed
        # "symmetry", "structure" # Structure from summary might be minimal, full structure fetched later
    ]

    # summary_docs_cache removed as it's no longer used.

    # Attempting to connect to Materials Project.
    # If api_key is None (due to not being found in config/env or being the placeholder),
    # mp-api client might attempt anonymous access or raise an error if queries require authentication.
    with MPRester(api_key=api_key) as mpr:
        # Old Fe-based fetching logic (summary_docs_cache, criteria_sets loop) fully removed.
        # New logic starts here.
        supercon_to_initial_candidates = {} # Changed variable name for clarity in this step

        # max_total_materials_config needs to be re-evaluated in this new context.
        # For now, we'll fetch for all target_compositions.
        # A global limit might still be useful but applied differently.
        # fetchAll variable might also be re-evaluated or removed if not fitting the new logic.

        # Ensure pymatgen.core.Composition is imported at the top of the file
        # from pymatgen.core import Composition

        processed_composition_count = 0

        # Iterate over the keys (compositions) of the map
        for supercon_comp_str in comp_to_tc_map.keys():
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
                    fields=initial_search_fields
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

                    # Client-side filtering for entries that are not deprecated
                    # Also, try to match the reduced formula as a heuristic (though chemsys should be primary)
                    for doc in current_mp_entries:
                        if hasattr(doc, 'deprecated') and doc.deprecated:
                            continue # Skip deprecated materials

                        # Ensure 'composition_reduced' is present if we want to filter by it
                        # For now, we'll take all non-deprecated entries from the chemsys search
                        # and rely on 'select_best_mp_entry' later if further filtering by formula is needed on more detailed data.
                        # The main point of this initial search is to get material_ids for a given chemical system.

                        # Store the initial search result (minimal dict from model_dump)
                        relevant_entries.append(doc.model_dump())


                    if relevant_entries:
                        print(f"    Storing {len(relevant_entries)} initial candidates for SuperCon composition {supercon_comp_str} (chemsys: {chemical_system}).")
                        supercon_to_initial_candidates[supercon_comp_str] = relevant_entries
                    else:
                        # This means no non-deprecated entries were found for the chemsys
                        print(f"    No non-deprecated entries found for chemical system {chemical_system} (from SuperCon comp {supercon_comp_str}).")
                        supercon_to_initial_candidates[supercon_comp_str] = None
                else:
                    # This means mpr.materials.search returned nothing for the chemsys
                    supercon_to_initial_candidates[supercon_comp_str] = None
                    print(f"  No MP entries found for chemical system {chemical_system} (from SuperCon comp {supercon_comp_str}).")

            except Exception as e:
                warnings.warn(f"  Error processing or querying for SuperCon composition {supercon_comp_str}: {str(e)[:500]}")
                supercon_to_initial_candidates[supercon_comp_str] = None

            processed_composition_count += 1
            # Optional: add a small delay to be polite to the API
            # import time
            # time.sleep(0.05) # 50 ms delay

        # print(f"\nFinished querying MP for {processed_composition_count} SuperCon compositions.") # Moved into the next block

    # (This code goes after the first loop that populates supercon_to_initial_candidates)
    # ...
    print(f"\nFinished initial candidate search for {processed_composition_count} SuperCon compositions. Found candidates for {len(supercon_to_initial_candidates)} of them.")

    all_candidate_ids = set()
    if supercon_to_initial_candidates: # Ensure it's not empty before iterating
        for comp_str, candidates_list in supercon_to_initial_candidates.items():
            if candidates_list: # candidates_list is a list of dicts
                for candidate_dict in candidates_list:
                    if isinstance(candidate_dict, dict) and candidate_dict.get('material_id'):
                        all_candidate_ids.add(str(candidate_dict['material_id'])) # Ensure material_id is string

    detailed_summaries_map = {}
    if all_candidate_ids:
        unique_ids_list = list(all_candidate_ids)
        print(f"Found {len(unique_ids_list)} unique material IDs from initial search. Fetching detailed summaries...")

        try:
            # Fetch summary documents in batches if necessary, though summary.search handles lists of IDs
            # mp-api's summary.search can take a list of material_ids.
            # Default limit for summary.search is 1000. If more IDs, batching might be needed,
            # but for now, assume total unique IDs will be manageable in one call.
            batch_size = 500 # Example batch size if needed
            for i in range(0, len(unique_ids_list), batch_size):
                batch_ids = unique_ids_list[i:i + batch_size]
                print(f"  Fetching detailed summaries for batch {i//batch_size + 1} ({len(batch_ids)} IDs)...")
                summary_docs_batch = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=detailed_summary_fields
                )
                if summary_docs_batch:
                    for summary_doc in summary_docs_batch:
                        # Ensure material_id is string for key consistency
                        detailed_summaries_map[str(summary_doc.material_id)] = summary_doc.model_dump()
                print(f"    Fetched {len(summary_docs_batch) if summary_docs_batch else 0} summaries for this batch.")

            print(f"Successfully fetched {len(detailed_summaries_map)} detailed summaries for the unique material IDs.")
        except Exception as e:
            warnings.warn(f"Error fetching detailed summaries by material IDs: {str(e)[:500]}", UserWarning)
            # detailed_summaries_map will contain whatever was fetched before the error
    else:
        print("No unique material IDs found from initial search. Skipping detailed summary fetch.")

    # The variable `supercon_to_initial_candidates` (populated in the previous step)
    # and `detailed_summaries_map` (populated here) will be used in the *next* step (Adapt Main Loop)
    # to reconstruct the `all_materials_data_map` with the detailed entries.
    # For now, the script's main data structure to be saved at the end needs to be decided.
    # Let's temporarily adjust the saving part to save detailed_summaries_map for inspection,
    # or comment out saving until the next step where all_materials_data_map is correctly rebuilt.
    # For this subtask, we'll assume the final saving will be handled after all_materials_data_map is repopulated.
    # So, no changes to the saving block in THIS subtask.
    # print(f"\nTotal materials collected after all processing: {len(raw_materials_data)}") # Old message

    # New main processing loop using detailed_summaries_map
    print("\nProcessing SuperCon compositions with detailed summaries...")
    all_materials_data_map = {} # This will be the final map for saving

    for supercon_comp_str, initial_candidates_list in supercon_to_initial_candidates.items():
        print(f"Re-processing SuperCon composition for final selection: {supercon_comp_str}")

        if not initial_candidates_list: # Should be a list of minimal dicts
            all_materials_data_map[supercon_comp_str] = None
            print(f"  No initial candidates found for {supercon_comp_str}. Storing None.")
            continue

        current_detailed_candidates = []
        for initial_candidate_dict in initial_candidates_list:
            if not isinstance(initial_candidate_dict, dict): continue

            candidate_material_id = str(initial_candidate_dict.get('material_id'))
            detailed_summary_dict = detailed_summaries_map.get(candidate_material_id)

            if detailed_summary_dict:
                current_detailed_candidates.append(detailed_summary_dict)
            else:
                warnings.warn(f"  Could not find detailed summary for candidate ID {candidate_material_id} "
                              f"of SuperCon comp {supercon_comp_str}. This ID will be skipped.", UserWarning)

        if not current_detailed_candidates:
            all_materials_data_map[supercon_comp_str] = None
            print(f"  No detailed summaries found for any initial candidates of {supercon_comp_str}. Storing None.")
            continue

        best_entry_dict = select_best_mp_entry(current_detailed_candidates)

        if best_entry_dict:
            print(f"  Selected best MP entry: {best_entry_dict.get('material_id')} "
                  f"(EAH: {best_entry_dict.get('energy_above_hull', 'N/A')}, "
                  f"Form.E/atom: {best_entry_dict.get('formation_energy_per_atom', 'N/A')})")

            material_id_str = str(best_entry_dict.get('material_id'))

            if not material_id_str:
                warnings.warn(f"    Material ID missing in selected entry for {supercon_comp_str}. Storing summary data only.")
                all_materials_data_map[supercon_comp_str] = best_entry_dict
            else:
                # Fetch structure and DOS within the mpr context
                try:
                    print(f"    Fetching structure for {material_id_str}...")
                    structure_obj = mpr.get_structure_by_material_id(material_id_str)
                    cif_string = structure_obj.to(fmt="cif") if structure_obj else None
                    best_entry_dict['cif_string_mp'] = cif_string
                    if not cif_string: warnings.warn(f"      Could not get CIF string for {material_id_str}")

                    print(f"    Fetching DOS for {material_id_str}...")
                    dos_obj = mpr.get_dos_by_material_id(material_id_str)
                    dos_dict_data = dos_obj.as_dict() if dos_obj else None
                    best_entry_dict['dos_object_mp'] = dos_dict_data
                    if not dos_dict_data: warnings.warn(f"      Could not get DOS for {material_id_str}")
                except Exception as detail_exc:
                    warnings.warn(f"    Error fetching details (structure/DOS) for {material_id_str}: {str(detail_exc)[:200]}")
                    best_entry_dict.setdefault('cif_string_mp', None)
                    best_entry_dict.setdefault('dos_object_mp', None)

                # Add critical temperature to the best_entry_dict
                tc_value = comp_to_tc_map.get(supercon_comp_str) # Get tc for the current composition
                if tc_value is not None: # Only add if tc_value is not None (it could be None if NaN in CSV)
                    best_entry_dict['critical_temperature_tc'] = tc_value
                else:
                    # If tc_value is None from the map, explicitly set it or decide not to add the key
                    best_entry_dict['critical_temperature_tc'] = None # Or simply don't add the key if that's preferred

                all_materials_data_map[supercon_comp_str] = best_entry_dict
        else:
            print(f"  No suitable best entry found by select_best_mp_entry from {len(current_detailed_candidates)} detailed candidates for {supercon_comp_str}.")
            # Store None for the material data, but we might still want to store its Tc if available
            # However, current logic ties Tc to best_entry_dict. If best_entry_dict is None, Tc is not added here.
            # This is acceptable as process_raw_data expects material data or None.
            all_materials_data_map[supercon_comp_str] = None

    # End of the new main processing loop

    # Final Saving Logic - now saves all_materials_data_map
    if all_materials_data_map: # Check if the map is not empty
        print(f"\nSaving final processed data to {output_filename}...")
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            with open(output_filename, 'w') as f:
                json.dump(all_materials_data_map, f, indent=4) # Save the final dictionary

            successfully_mapped_supercon_comps = sum(1 for v in all_materials_data_map.values() if v is not None)
            print(f"Successfully saved final processed data for {successfully_mapped_supercon_comps} SuperCon compositions to {output_filename}.")
        except Exception as e:
            print(f"Error saving final data to JSON: {e}")
    else:
        print("No final processed data collected to save.")

if __name__ == "__main__":
    # The fetch_data function now loads its own config, so no need to pass max_total_materials from here
    # The fetch_data function now loads its own config.
    # Pass the default max_total_materials if you want it to be the ultimate fallback.
    fetch_data(max_total_materials_arg=50) # Original default was 50
