import pandas as pd
import os

RAW_FILE_PATH = "supercon_data/raw.tsv"
PROCESSED_FILE_PATH = "data/supercon_processed.csv"

def process_supercon_data():
    print(f"Starting processing of {RAW_FILE_PATH}...")

    if not os.path.exists(RAW_FILE_PATH):
        print(f"ERROR: Raw data file not found at {RAW_FILE_PATH}")
        # As per instruction, download and unzip if not found (robustness)
        print("Attempting to download and unzip SuperCon dataset...")
        download_command = 'curl -L -o supercon-dataset.zip "https://www.kaggle.com/api/v1/datasets/download/chaozhuang/supercon-dataset" && unzip -o supercon-dataset.zip -d supercon_data'
        download_result = os.system(download_command)
        if download_result != 0 or not os.path.exists(RAW_FILE_PATH):
            print(f"Failed to download or find {RAW_FILE_PATH} after download attempt. Exiting.")
            return
        print("Dataset downloaded and unzipped successfully.")


    try:
        # Read the TSV file, skipping the first metadata line and using the second line as header
        df = pd.read_csv(RAW_FILE_PATH, sep='\t', skiprows=[0], header=0)
        print(f"Successfully loaded {RAW_FILE_PATH}. Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading {RAW_FILE_PATH}: {e}")
        return

    # Identify the correct column names based on previous exploration
    composition_col = 'chemical formula'
    tc_col = 'Tc (of this sample) recommended'

    if composition_col not in df.columns:
        print(f"ERROR: Composition column '{composition_col}' not found in the dataframe.")
        return
    if tc_col not in df.columns:
        print(f"ERROR: Tc column '{tc_col}' not found in the dataframe.")
        return

    # Select relevant columns
    df_processed = df[[composition_col, tc_col]].copy()

    # Rename columns for clarity
    df_processed.rename(columns={
        composition_col: 'composition',
        tc_col: 'critical_temperature_tc'
    }, inplace=True)

    # Convert Tc to numeric, coercing errors to NaN
    df_processed['critical_temperature_tc'] = pd.to_numeric(df_processed['critical_temperature_tc'], errors='coerce')

    # Drop rows where Tc could not be parsed to a number or is missing
    original_count = len(df_processed)
    df_processed.dropna(subset=['critical_temperature_tc', 'composition'], inplace=True)
    dropped_count = original_count - len(df_processed)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows due to missing or non-numeric Tc or missing composition.")

    # Remove entries with Tc <= 0, as these are not typically considered superconductors in this context
    positive_tc_original_count = len(df_processed)
    df_processed = df_processed[df_processed['critical_temperature_tc'] > 0]
    dropped_non_positive_tc = positive_tc_original_count - len(df_processed)
    if dropped_non_positive_tc > 0:
        print(f"Dropped {dropped_non_positive_tc} rows with Tc <= 0.")


    # Save the processed data
    try:
        df_processed.to_csv(PROCESSED_FILE_PATH, index=False)
        print(f"Successfully saved processed data to {PROCESSED_FILE_PATH}")
    except Exception as e:
        print(f"Error saving processed data to {PROCESSED_FILE_PATH}: {e}")
        return

    num_entries = len(df_processed)
    num_unique_compositions = df_processed['composition'].nunique()

    print(f"Processing complete.")
    print(f"Total entries processed and saved: {num_entries}")
    print(f"Number of unique compositions: {num_unique_compositions}")

if __name__ == "__main__":
    process_supercon_data()
