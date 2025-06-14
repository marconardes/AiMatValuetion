import json
import pandas as pd
import os

OQMD_RAW_FILE = "data/oqmd_data_raw.json"
PROCESSED_OQMD_FILE = "data/oqmd_processed.csv"

# Helper function to validate structural data within an OQMD record
def is_structural_data_valid(record):
    """
    Checks if an OQMD record contains potentially valid unit_cell and sites data.
    """
    if record is None:
        return False

    unit_cell_data = record.get('unit_cell')
    sites_data = record.get('sites')

    # Check unit_cell
    if not isinstance(unit_cell_data, list) or len(unit_cell_data) != 6:
        return False
    # Ensure all elements in unit_cell_data are numbers
    if not all(isinstance(x, (int, float)) for x in unit_cell_data):
        return False

    # Check sites: must be a non-empty list of dictionaries
    if not isinstance(sites_data, list) or not sites_data:
        return False

    for site in sites_data:
        if not isinstance(site, dict):
            return False
        # Check for mandatory keys 'species' and 'xyz'
        if 'species' not in site or 'xyz' not in site:
            return False
        # Check 'xyz': must be a list of 3 numbers
        if not isinstance(site['xyz'], list) or len(site['xyz']) != 3:
            return False
        if not all(isinstance(x, (int, float)) for x in site['xyz']):
            return False
        # Check 'species': must be a non-empty list of dictionaries
        if not isinstance(site['species'], list) or not site['species']:
            return False
        # Check the first entry in 'species' list: must be a dict and have 'element' key
        first_species_info = site['species'][0]
        if not isinstance(first_species_info, dict) or 'element' not in first_species_info:
            return False
    return True

def select_best_oqmd_entry(oqmd_records):
    if not oqmd_records:
        return None

    structurally_valid_records = []
    for record in oqmd_records:
        if not isinstance(record, dict): # Ensure record itself is a dict
            print(f"Warning: Skipping non-dictionary record in OQMD data: {str(record)[:100]}")
            continue
        if is_structural_data_valid(record): # Use the new helper function
            structurally_valid_records.append(record)
        # else: # Optional: log why a record was deemed structurally invalid
            # print(f"Debug: Record {record.get('entry_id', 'N/A')} ({record.get('name', 'N/A')}) from original list was skipped due to invalid structural data.")


    if not structurally_valid_records:
        # print("Debug: No structurally valid records found after filtering.") # Optional debug
        return None

    infinity = float('inf')
    structurally_valid_records.sort(key=lambda x: (
        x.get('delta_e', infinity) if x.get('delta_e') is not None else infinity,
        x.get('stability', infinity) if x.get('stability') is not None else infinity,
        x.get('entry_id', 0)
    ))

    return structurally_valid_records[0]

def process_oqmd_data():
    print(f"Starting processing of {OQMD_RAW_FILE}...")

    if not os.path.exists(OQMD_RAW_FILE):
        print(f"ERROR: Raw OQMD data file not found: {OQMD_RAW_FILE}")
        print("Please run `scripts/fetch_oqmd_data.py` first to generate this file.")
        # Optionally, try to run fetch_oqmd_data.py
        if os.path.exists("scripts/fetch_oqmd_data.py"):
            print("Attempting to run scripts/fetch_oqmd_data.py...")
            try:
                # Ensure fetch_oqmd_data.py is executable for the os.system call
                os.chmod("scripts/fetch_oqmd_data.py", 0o755)
                exit_code = os.system("python scripts/fetch_oqmd_data.py") # This will use the current settings in that script
                if exit_code != 0:
                    print(f"scripts/fetch_oqmd_data.py exited with error code {exit_code}. Cannot proceed.")
                    return
                if not os.path.exists(OQMD_RAW_FILE): # Check again
                    print(f"{OQMD_RAW_FILE} still not found. Cannot proceed.")
                    return
                print(f"{OQMD_RAW_FILE} generated.")
            except Exception as e:
                print(f"Error running scripts/fetch_oqmd_data.py: {e}")
                return
        else:
            print("scripts/fetch_oqmd_data.py not found. Cannot generate missing input.")
            return


    try:
        with open(OQMD_RAW_FILE, "r") as f:
            raw_data_map = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing {OQMD_RAW_FILE}: {e}")
        return

    processed_data_list = []
    print(f"Processing data for {len(raw_data_map)} SuperCon compositions...")

    for supercon_comp, oqmd_records in raw_data_map.items():
        if oqmd_records is None or not isinstance(oqmd_records, list) or not oqmd_records:
            # print(f"No OQMD records found for SuperCon composition: {supercon_comp}. Skipping.")
            processed_data_list.append({'supercon_composition': supercon_comp}) # Add with Nones for other fields
            continue

        best_entry = select_best_oqmd_entry(oqmd_records)

        if best_entry:
            # Ensure unit_cell and sites are stored as JSON strings if they are complex objects (lists/dicts)
            unit_cell_json = None
            if best_entry.get('unit_cell') is not None:
                try:
                    unit_cell_json = json.dumps(best_entry['unit_cell'])
                except TypeError:
                    unit_cell_json = str(best_entry['unit_cell']) # Fallback to string if not JSON serializable

            sites_json = None
            if best_entry.get('sites') is not None:
                try:
                    sites_json = json.dumps(best_entry['sites'])
                except TypeError:
                    sites_json = str(best_entry['sites']) # Fallback

            processed_entry = {
                'supercon_composition': supercon_comp,
                'oqmd_entry_id': best_entry.get('entry_id'),
                'oqmd_formula': best_entry.get('name'), # OQMD uses 'name' for formula
                'oqmd_spacegroup': best_entry.get('spacegroup'),
                'oqmd_delta_e': best_entry.get('delta_e'),
                'oqmd_stability': best_entry.get('stability'),
                'oqmd_band_gap': best_entry.get('band_gap'),
                'oqmd_prototype': best_entry.get('prototype'),
                'oqmd_unit_cell_json': unit_cell_json,
                'oqmd_sites_json': sites_json,
                'oqmd_icsd_id': best_entry.get('icsd_id')
            }
            processed_data_list.append(processed_entry)
        else:
            # print(f"Could not select a best entry for SuperCon composition: {supercon_comp} from {len(oqmd_records)} records. Skipping.")
            processed_data_list.append({'supercon_composition': supercon_comp}) # Add with Nones

    if not processed_data_list:
        print("No data was processed. Output file will not be created.")
        return

    df_processed = pd.DataFrame(processed_data_list)

    # Define column order for consistency
    column_order = [
        'supercon_composition', 'oqmd_entry_id', 'oqmd_formula', 'oqmd_spacegroup',
        'oqmd_delta_e', 'oqmd_stability', 'oqmd_band_gap', 'oqmd_prototype',
        'oqmd_unit_cell_json', 'oqmd_sites_json', 'oqmd_icsd_id'
    ]
    df_processed = df_processed.reindex(columns=column_order)

    try:
        df_processed.to_csv(PROCESSED_OQMD_FILE, index=False)
        print(f"Successfully saved processed OQMD data to {PROCESSED_OQMD_FILE}")
        print(f"Total SuperCon compositions processed: {len(df_processed)}")
        print(f"Number of entries with OQMD data: {df_processed['oqmd_entry_id'].notna().sum()}")
    except Exception as e:
        print(f"Error saving data to {PROCESSED_OQMD_FILE}: {e}")

if __name__ == "__main__":
    process_oqmd_data()
