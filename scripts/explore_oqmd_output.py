import json
import os # Added os for path check

OQMD_RAW_DATA_FILE = "data/oqmd_data_raw.json"

print(f"Exploring structure of {OQMD_RAW_DATA_FILE}...")

try:
    if not os.path.exists(OQMD_RAW_DATA_FILE):
        # Added a check for the file and an attempt to generate it
        print(f"WARNING: {OQMD_RAW_DATA_FILE} not found. Attempting to generate it by running scripts/fetch_oqmd_data.py...")
        if os.path.exists("scripts/fetch_oqmd_data.py"):
            try:
                # Ensure fetch_oqmd_data.py is executable
                os.chmod("scripts/fetch_oqmd_data.py", 0o755)
                run_prereq_script_command = "python scripts/fetch_oqmd_data.py"
                # Before running, temporarily reduce LIMIT_COMPOSITIONS_TO_QUERY in fetch_oqmd_data.py for this regeneration
                # This is complex to do robustly from here. For now, assume it runs with its current settings.
                # Or, if the file is critical, the user should ensure it's generated with appropriate limits first.
                # Given the context, fetch_oqmd_data.py was just run with LIMIT=7, so the file should be small.
                exit_code = os.system(run_prereq_script_command)
                if exit_code != 0:
                    print(f"fetch_oqmd_data.py failed with exit code {exit_code}. Cannot proceed with exploration.")
                    exit() # Exit the exploration script
                if not os.path.exists(OQMD_RAW_DATA_FILE): # Check again
                    print(f"{OQMD_RAW_DATA_FILE} still not found after running prerequisite script. Cannot proceed.")
                    exit() # Exit the exploration script
                print(f"{OQMD_RAW_DATA_FILE} generated successfully by prerequisite script.")
            except Exception as e_prereq:
                print(f"Error running scripts/fetch_oqmd_data.py: {e_prereq}. Cannot proceed.")
                exit() # Exit the exploration script
        else:
            print("scripts/fetch_oqmd_data.py not found. Cannot generate missing input file {OQMD_RAW_DATA_FILE}. Please ensure it exists.")
            exit() # Exit the exploration script


    with open(OQMD_RAW_DATA_FILE, "r") as f:
        data = json.load(f)

    if not data:
        print(f"{OQMD_RAW_DATA_FILE} is empty or does not contain valid JSON.")
    else:
        print(f"Successfully loaded {OQMD_RAW_DATA_FILE}.")
        print(f"Number of SuperCon compositions with OQMD data (or attempt): {len(data)}")

        # Print data for the first 1-2 SuperCon compositions that have OQMD entries
        count = 0
        processed_valid_entry_count = 0 # To ensure we show up to 2 *valid* entries
        for supercon_comp, oqmd_records in data.items():
            if processed_valid_entry_count >= 2:
                break
            print(f"\n--- Data for SuperCon composition: {supercon_comp} ---")
            if oqmd_records is None:
                print("  No OQMD data was fetched (e.g., error or no matching element set).")
            elif not oqmd_records: # Empty list
                print("  OQMD query returned no matching entries for this element set.")
            else:
                print(f"  Number of OQMD records found for this element set: {len(oqmd_records)}")
                # Print the first OQMD record for this composition to see its structure
                if len(oqmd_records) > 0:
                    print("  Structure of the first OQMD record:")
                    print(json.dumps(oqmd_records[0], indent=2))
                processed_valid_entry_count+=1 # Increment only if we found and printed a valid record
            count += 1 # This outer count is less important now

except FileNotFoundError:
    # This specific block might be redundant now due to the os.path.exists check above,
    # but kept for general robustness if the generation logic fails silently or is removed.
    print(f"ERROR: {OQMD_RAW_DATA_FILE} not found. Please ensure 'fetch_oqmd_data.py' was run successfully.")
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {OQMD_RAW_DATA_FILE}. The file might be corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
