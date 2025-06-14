import pandas as pd
import argparse
import os
import warnings

def generate_top_tc_entries(input_csv_path, output_csv_path, tc_column_name, top_n=20):
    """
    Reads a CSV file, selects the top N entries based on the Tc column,
    and saves them to a new CSV file.
    """
    print(f"Starting processing for Top {top_n} Tc entries...")
    print(f"Input CSV: {input_csv_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Tc Column: '{tc_column_name}'")

    # 1. Read the input CSV
    if not os.path.exists(input_csv_path):
        warnings.warn(f"Error: Input file not found: {input_csv_path}", UserWarning)
        print("Aborting.")
        return

    try:
        df = pd.read_csv(input_csv_path)
    except pd.errors.EmptyDataError:
        warnings.warn(f"Warning: The input file is empty: {input_csv_path}", UserWarning)
        print("Aborting.")
        return
    except Exception as e:
        warnings.warn(f"Error reading CSV file {input_csv_path}: {e}", UserWarning)
        print("Aborting.")
        return

    if df.empty:
        print(f"The input CSV file {input_csv_path} is empty after loading. No data to process.")
        # Optionally, write an empty CSV if that's desired behavior
        # For now, just return.
        return

    print(f"Successfully loaded {len(df)} rows from {input_csv_path}.")

    # 2. Validate Tc Column
    if tc_column_name not in df.columns:
        warnings.warn(f"Error: Tc column '{tc_column_name}' not found in the CSV file.", UserWarning)
        print(f"Available columns are: {df.columns.tolist()}")
        print("Aborting.")
        return

    print(f"Tc column '{tc_column_name}' found.")

    # 3. Validate and Clean Tc Data
    # Convert Tc column to numeric, coercing errors to NaN
    original_row_count = len(df)
    df[tc_column_name] = pd.to_numeric(df[tc_column_name], errors='coerce')

    # Check how many values became NaN after coercion
    nan_tc_count = df[tc_column_name].isna().sum()
    if nan_tc_count > 0:
        warnings.warn(f"Warning: {nan_tc_count} rows had non-numeric values in the '{tc_column_name}' column "
                      f"and were converted to NaN.", UserWarning)

    # Drop rows where Tc is NaN
    df.dropna(subset=[tc_column_name], inplace=True)

    rows_dropped_count = original_row_count - len(df)
    if rows_dropped_count > 0: # This will include the nan_tc_count if they were the only reason for drop
        print(f"{rows_dropped_count} rows were dropped due to missing or non-numeric Tc values.")

    if df.empty:
        print(f"No valid numeric Tc data found in '{tc_column_name}' column after cleaning. Output file will be empty or not created.")
        # Optionally write an empty CSV
        # df_top_n = pd.DataFrame(columns=df.columns if original_row_count > 0 else [tc_column_name]) # Create empty df with original cols
        # df_top_n.to_csv(output_csv_path, index=False)
        # print(f"Empty CSV (with headers) saved to {output_csv_path} as no valid Tc data was found.")
        return

    print(f"After cleaning, {len(df)} rows with valid numeric Tc values remain.")

    # 4. Sort by Tc and Select Top N
    if df.empty: # Should have been caught earlier, but double-check
        print("No data available to sort and select. Exiting generation of top N file.")
        # Consider writing an empty file with headers if desired, as handled in cleaning step
        # For now, just return if df is empty at this stage.
        return

    df_sorted = df.sort_values(by=tc_column_name, ascending=False)

    actual_top_n = min(top_n, len(df_sorted)) # Handle cases where fewer than top_n rows are available

    df_top_n = df_sorted.head(actual_top_n)

    if len(df_sorted) < top_n:
        warnings.warn(
            f"Warning: Requested top {top_n} entries, but only {len(df_sorted)} "
            f"valid entries were available after cleaning. Output will contain {len(df_sorted)} entries.",
            UserWarning
        )
    else:
        print(f"Selected top {actual_top_n} entries based on '{tc_column_name}'.")

    # Placeholder for next step (writing to CSV)
    print(f"Data sorting and selection complete. Found {len(df_top_n)} entries for output.")
    # For debugging or inspection, you could print df_top_n.head() here
    # print(df_top_n.head())

    # 5. Write to New CSV File
    if not df_top_n.empty:
        try:
            # Ensure output directory exists (main() already does this, but good for function robustness if called directly)
            # output_dir_for_func = os.path.dirname(output_csv_path)
            # if output_dir_for_func and not os.path.exists(output_dir_for_func):
            #     os.makedirs(output_dir_for_func)

            df_top_n.to_csv(output_csv_path, index=False)
            print(f"Successfully saved {len(df_top_n)} top Tc entries to: {output_csv_path}")
        except Exception as e:
            warnings.warn(f"Error writing output CSV to {output_csv_path}: {e}", UserWarning)
            print(f"Failed to save the output CSV file.")
    elif os.path.exists(input_csv_path) and pd.read_csv(input_csv_path).empty: # Input was empty
        print(f"Input CSV ({input_csv_path}) was empty. No output file created.")
    elif not df.empty and df_top_n.empty and top_n > 0 : # Valid Tc data existed, but maybe top_n was 0 or became 0
        print(f"No entries selected for top {top_n} (e.g., if top_n was 0 or data filtered out). Output file not created.")
    else: # This case implies df was empty after cleaning, already handled by returning earlier in the function.
          # Or if df_top_n is empty for other reasons not caught by specific messages.
        print("No data to write to output CSV. File not created.")

def main():
    parser = argparse.ArgumentParser(description="Generate a CSV file with top N entries by critical temperature (Tc).")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/supercon_processed.csv",
        help="Path to the input CSV file (e.g., supercon_processed.csv)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/top_20_superconductors_by_tc.csv",
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--tc_column",
        type=str,
        default="critical_temperature_tc",
        help="Name of the column containing critical temperature (Tc) values."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top entries to select."
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    generate_top_tc_entries(args.input_csv, args.output_csv, args.tc_column, args.top_n)

if __name__ == "__main__":
    main()
