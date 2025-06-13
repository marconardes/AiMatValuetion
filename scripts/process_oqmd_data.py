import json
import pandas as pd
import os

OQMD_RAW_FILE = "data/oqmd_data_raw.json"
PROCESSED_OQMD_FILE = "data/oqmd_processed.csv"

def select_best_oqmd_entry(oqmd_records):
    if not oqmd_records:
        return None

    # Filter out records that might be missing critical data for selection
    # For example, ensure 'delta_e' and 'stability' can be accessed or default them
    valid_records = []
    for record in oqmd_records:
        try:
            # Ensure essential keys for sorting exist, can be None if that's acceptable for sorting logic
            record.get('delta_e')
            record.get('stability')
            valid_records.append(record)
        except AttributeError: # If a record isn't a dict, skip it
            print(f"Warning: Skipping non-dictionary record in OQMD data: {str(record)[:100]}")
            continue

    if not valid_records:
        return None

    # Sort by 'delta_e' (formation energy), then by 'stability' (hull distance)
    # Lower delta_e is generally more stable. Lower (closer to zero or more negative) stability is better.
    # Handle cases where these keys might be missing or are None by providing a default for sorting
    # A large number for None delta_e/stability will push them to the end of the sort (less preferred)
    infinity = float('inf')

    valid_records.sort(key=lambda x: (
        x.get('delta_e', infinity) if x.get('delta_e') is not None else infinity,
        x.get('stability', infinity) if x.get('stability') is not None else infinity,
        x.get('entry_id', 0) # As a final tie-breaker, though less physically meaningful
    ))

    return valid_records[0]

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
