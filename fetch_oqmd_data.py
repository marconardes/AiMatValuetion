import pandas as pd
import requests
import time
import json
import os
import re
from pymatgen.core import Composition, DummySpecies
from pymatgen.core.periodic_table import Element

# This script fetches data from the Open Quantum Materials Database (OQMD)
# primarily to find complementary material properties and crystal structures
# for compositions identified in the SuperCon dataset.
#
# The current query strategy uses an 'element_set' filter based on the
# elements present in each SuperCon composition. This means that for a
# SuperCon composition like 'BaTiO3', the script will query OQMD for all
# entries containing Ba, Ti, and O (e.g., BaTiO3, BaO, TiO2, Ba2TiO4, etc.).
# This approach gathers a broad set of related materials.
#
# Exact compositional matching or selection of the most relevant OQMD entry
# for a given SuperCon composition is typically handled in a downstream
# data processing step (e.g., in `process_oqmd_data.py`).

# Constants
SUPERCON_PROCESSED_FILE = "supercon_processed.csv" # Assumes this file exists from previous step
OQMD_BASE_URL = "http://oqmd.org/oqmdapi/formationenergy"
REQUEST_FIELDS = "name,entry_id,spacegroup,ntypes,band_gap,delta_e,stability,prototype,unit_cell,sites,icsd_id"
REQUEST_DELAY = 1 # seconds, to be polite to the API
OQMD_OUTPUT_FILE = "oqmd_data_raw.json"
LIMIT_COMPOSITIONS_TO_QUERY = 7 # TEMPORARY OVERRIDE FOR SUBTASK RUN

def get_unique_compositions(file_path):
    print(f"Reading unique compositions from {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        # Attempt to run the prerequisite script if supercon_processed.csv is missing
        # This makes the script more robust if run standalone after a git pull for example
        print(f"Attempting to generate {file_path} by running process_supercon_raw.py...")
        if os.path.exists("process_supercon_raw.py"):
            try:
                run_prereq_script_command = "python process_supercon_raw.py"
                exit_code = os.system(run_prereq_script_command)
                if exit_code != 0:
                    print(f"process_supercon_raw.py failed with exit code {exit_code}. Cannot proceed.")
                    return []
                if not os.path.exists(file_path): # Check again
                    print(f"{file_path} still not found after running prerequisite script. Cannot proceed.")
                    return []
                print(f"{file_path} generated successfully.")
            except Exception as e:
                print(f"Error running process_supercon_raw.py: {e}. Cannot proceed.")
                return []
        else:
            print("process_supercon_raw.py not found. Cannot generate missing input file. Please ensure it exists.")
            return []


    try:
        df = pd.read_csv(file_path)
        unique_comps = df['composition'].unique().tolist()
        print(f"Found {len(unique_comps)} unique compositions.")
        return unique_comps
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")
        return []

def get_elements_from_composition(composition_str):
    # Same cleaning logic as before to make it Pymatgen-friendly
    cleaned_comp = re.sub(r'\(.*?\)', '', composition_str)
    suffixes_to_remove = ['-Y', '-delta', '-X', 'x=', '-x', '+x', '-d', '+d', '(vac)']
    for suffix in suffixes_to_remove:
        cleaned_comp = cleaned_comp.replace(suffix, '')
    cleaned_comp = cleaned_comp.strip()

    if not cleaned_comp:
        # print(f"Warning: Composition '{composition_str}' became empty after cleaning.")
        return None
    try:
        parsed_comp = Composition(cleaned_comp, allow_dummy=True)
        elements = []
        valid_composition = True
        for el in parsed_comp.elements:
            if isinstance(el, DummySpecies):
                # For element_set query, we cannot include dummy species.
                # So, if any unignorable dummy is present, this composition cannot be used for element_set query.
                if str(el) not in ['X', 'M', 'RE', 'Ln', 'A', 'B']: # Common placeholders we might ignore for formula reconstruction
                    # print(f"Warning: Composition '{composition_str}' (cleaned: '{cleaned_comp}') contains unhandled DummySpecies '{el}'. Cannot form element_set.")
                    valid_composition = False
                    break
                # else:
                    # print(f"Ignoring placeholder DummySpecies '{el}' for element_set from '{cleaned_comp}'")
            else:
                elements.append(el.symbol)

        if not valid_composition or not elements:
            # print(f"Could not extract valid elements for element_set query from '{composition_str}' (cleaned: '{cleaned_comp}')")
            return None

        return sorted(list(set(elements))) # Return sorted list of unique element symbols

    except Exception as e:
        # print(f"Warning: Pymatgen parsing failed for '{composition_str}' (cleaned: '{cleaned_comp}') for element extraction: {e}")
        # Fallback regex if pymatgen fails completely on the cleaned string.
        # This tries to find capitalized elements.
        formula_parts = re.findall(r'([A-Z][a-z]?)', cleaned_comp) # Only extract element symbols
        if formula_parts:
            # print(f"  Fallback: extracted elements using regex: {formula_parts}")
            return sorted(list(set(formula_parts))) # Unique sorted list

        print(f"Warning: Could not extract elements from '{composition_str}' (cleaned: '{cleaned_comp}') even after fallbacks.")
        return None

def fetch_oqmd_data_for_elements(original_composition_str, elements_list):
    if not elements_list:
        print(f"No elements provided for original composition: {original_composition_str}. Skipping OQMD query.")
        return None

    element_set_str = ",".join(elements_list)
    print(f"Fetching OQMD data for element set: {element_set_str} (from original: {original_composition_str})...")

    params = {
        'filter': f'element_set={element_set_str}',
        'fields': REQUEST_FIELDS,
        'limit': 50, # OQMD default limit
        'format': 'json'
    }

    oqmd_data_for_set = []
    page_num = 0
    # Max pages to avoid overly broad queries in testing; OQMD default limit is 50, max is not explicitly stated but often around 200-1000 for APIs
    # For an element_set query, number of entries could be very large.
    max_pages_for_element_set = 10 # Fetch up to 10 pages (e.g., 10*50=500 entries if limit is 50)
                                  # This means we might miss data for very large systems.

    current_url_base = OQMD_BASE_URL # Define it once

    while page_num < max_pages_for_element_set:
        current_params = params.copy() # Use a copy to modify offset
        current_params['offset'] = page_num * current_params['limit']

        # Construct URL for logging before making the request
        query_string = requests.compat.urlencode(current_params)
        log_url = f"{current_url_base}?{query_string}"
        print(f"  Querying URL (page {page_num + 1}): {log_url}")

        try:
            response = requests.get(current_url_base, params=current_params, timeout=60) # Increased timeout
            response.raise_for_status()
            data = response.json()

            if data.get("data") and isinstance(data["data"], list):
                oqmd_data_for_set.extend(data["data"])
                print(f"  Received {len(data['data'])} new entries. Total for {element_set_str} so far: {len(oqmd_data_for_set)}")
            else:
                print(f"  No 'data' list found or not a list for {element_set_str} (page {page_num+1}).")
                # Log if "data" key is missing or if its value is unexpected
                if "data" not in data:
                    print(f"    'data' key missing in response. Response keys: {list(data.keys())}")
                elif not isinstance(data["data"], list):
                    print(f"    'data' field is not a list. Type: {type(data['data'])}. Value: {str(data['data'])[:100]}")
                # print(f"Full response for {element_set_str} (page {page_num+1}): {json.dumps(data, indent=2)[:500]}") # Log snippet of full response


            if data.get("meta", {}).get("more_data_available") and data.get("links", {}).get("next"):
                page_num += 1
                print(f"  Pagination: More data available, proceeding to next page for {element_set_str}.")
                time.sleep(REQUEST_DELAY)
            else:
                print(f"  Pagination: No more data or no next link for {element_set_str}.")
                break
        except requests.exceptions.HTTPError as http_err:
            print(f"  HTTP error occurred for {element_set_str}: {http_err} - Response: {response.text[:200] if response else 'No response text'}")
            break
        except requests.exceptions.RequestException as req_err:
            print(f"  Request exception occurred for {element_set_str}: {req_err}")
            break
        except json.JSONDecodeError as json_err:
            print(f"  JSON decode error for {element_set_str}: {json_err}. Response text: {response.text[:200]}")
            break
        except Exception as e:
            print(f"  An unexpected error occurred for {element_set_str}: {e}")
            break

    return oqmd_data_for_set if oqmd_data_for_set else None

def main():
    print("--- Starting OQMD Data Fetching (Initial Version) ---")
    compositions = get_unique_compositions(SUPERCON_PROCESSED_FILE)

    if not compositions:
        print("No compositions to process. Exiting.")
        return

    queried_count = 0
    all_fetched_data = {}

    for i, comp_str in enumerate(compositions):
        if LIMIT_COMPOSITIONS_TO_QUERY is not None and queried_count >= LIMIT_COMPOSITIONS_TO_QUERY:
            print(f"Reached query limit of {LIMIT_COMPOSITIONS_TO_QUERY}. Stopping.")
            break

        print(f"\nProcessing SuperCon composition {i+1}/{len(compositions)}: {comp_str}")
        elements = get_elements_from_composition(comp_str) # New call

        if elements:
            data_list = fetch_oqmd_data_for_elements(comp_str, elements) # New call

            if data_list:
                # Store data keyed by the original SuperCon composition string for later association
                all_fetched_data[comp_str] = data_list
                print(f"Successfully fetched {len(data_list)} OQMD entries for element set derived from {comp_str}.")
            else:
                print(f"No OQMD data fetched or error for element set from {comp_str}.")
                all_fetched_data[comp_str] = None
        else:
            print(f"Could not derive valid element set from SuperCon composition: {comp_str}. Skipping OQMD query.")
            all_fetched_data[comp_str] = None # Mark as error/no query

        queried_count += 1
        if queried_count < LIMIT_COMPOSITIONS_TO_QUERY and i < len(compositions) - 1: # Check against None for LIMIT
             print(f"Sleeping for {REQUEST_DELAY} second(s)...")
             time.sleep(REQUEST_DELAY)

    print("\n--- OQMD Data Fetching Summary (Initial Run) ---")
    if all_fetched_data:
        successful_queries = {k:v for k,v in all_fetched_data.items() if v is not None}
        print(f"Data fetched for {len(successful_queries)} compositions (out of {queried_count} attempted):")
        for comp, data in successful_queries.items():
            print(f"  Composition: {comp}, Records found: {len(data)}")
            if data: # Print id of first record for brevity
                 print(f"    First record entry_id: {data[0].get('entry_id', 'N/A')}, name: {data[0].get('name', 'N/A')}")

    # Add saving logic here
    if all_fetched_data:
        print(f"\nSaving all fetched OQMD data to {OQMD_OUTPUT_FILE}...")
        try:
            with open(OQMD_OUTPUT_FILE, "w") as f:
                json.dump(all_fetched_data, f, indent=2)
            print(f"Successfully saved data to {OQMD_OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving data to {OQMD_OUTPUT_FILE}: {e}")
    else:
        print("No data was fetched in this run, so nothing to save.") # Added else for clarity

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
