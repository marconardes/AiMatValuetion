import json
import os
import warnings
import pandas as pd
import math

import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice # Added for Lattice creation

from utils.config_loader import load_config
from utils.graph_utils import structure_to_graph # Assuming this will be used or adapted

def create_graph_dataset(config_path='config.yml'):
    '''
    Processes raw material data into graph objects, saves them,
    and then splits them into training, validation, and test sets.

    Args:
        config_path (str, optional): Path to the configuration file.
                                     Defaults to 'config.yml'.
    '''
    print("Starting graph dataset creation and splitting...")

    # 1. Load configuration
    config = load_config(config_path=config_path)
    if not config:
        warnings.warn(f"Failed to load configuration from {config_path}. Aborting.", UserWarning)
        return

    prepare_config = config.get('prepare_gnn_data', {})
    raw_data_path = prepare_config.get('processed_oqmd_csv_filename', 'data/oqmd_processed.csv') # Updated key and default path
    processed_graphs_path = prepare_config.get('processed_graphs_filename', 'data/processed_graphs.pt')
    train_graphs_path = prepare_config.get('train_graphs_filename', 'data/train_graphs.pt')
    val_graphs_path = prepare_config.get('val_graphs_filename', 'data/val_graphs.pt')
    test_graphs_path = prepare_config.get('test_graphs_filename', 'data/test_graphs.pt')
    random_seed = prepare_config.get('random_seed', 42)
    train_ratio = prepare_config.get('train_ratio', 0.7)
    val_ratio = prepare_config.get('val_ratio', 0.2)
    test_ratio = prepare_config.get('test_ratio', 0.1)

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        warnings.warn(f"Train ({train_ratio}), validation ({val_ratio}), and test ({test_ratio}) ratios do not sum closely to 1.0. Please check config.yml.", UserWarning)
        # For now, we'll proceed, but in a robust script, you might normalize or exit.

    print("Configuration loaded.")
    print(f"Using raw data file: {raw_data_path}")
    print(f"Processed graphs will be saved to: {processed_graphs_path}")
    print(f"Train data: {train_graphs_path}, Val data: {val_graphs_path}, Test data: {test_graphs_path}")
    print(f"Splitting with Train: {train_ratio*100}%, Validation: {val_ratio*100}%, Test: {test_ratio*100}%, Seed: {random_seed}")

    # 2. Load raw data
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file '{raw_data_path}' not found. Please check config.yml and file path.")
        return

    try:
        # Load data using pandas for CSV files
        raw_materials_data = pd.read_csv(raw_data_path)
        print(f"Loaded {len(raw_materials_data)} raw material entries from {raw_data_path}.")
    except FileNotFoundError:
        print(f"Error: The file {raw_data_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {raw_data_path} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Could not parse {raw_data_path}. Check CSV format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading {raw_data_path} with pandas: {e}")
        return

    all_graph_data = []
    processed_count = 0
    # Old CIF related counters - will be replaced or updated
    # skipped_missing_cif = 0
    # skipped_cif_parse_error = 0
    skipped_missing_structure_json = 0
    skipped_structure_json_parse_error = 0
    skipped_structure_creation_error = 0
    skipped_missing_targets = 0
    skipped_processing_error = 0

    # 3. Process each material into a torch_geometric.data.Data object
    # Iterate over DataFrame rows if using pandas
    for i, material_doc in raw_materials_data.iterrows(): # Changed for pandas DataFrame
        material_id = material_doc.get('material_id', f"Unknown_ID_{i+1}") # .get still works for Series
        print(f"Processing material {i+1}/{len(raw_materials_data)}: {material_id}")

        # --- Structure Reconstruction from JSON ---
        oqmd_unit_cell_json = material_doc.get('oqmd_unit_cell_json')
        oqmd_sites_json = material_doc.get('oqmd_sites_json')

        if pd.isna(oqmd_unit_cell_json) or not oqmd_unit_cell_json or \
           pd.isna(oqmd_sites_json) or not oqmd_sites_json:
            warnings.warn(f"Skipping material {material_id} due to missing oqmd_unit_cell_json or oqmd_sites_json.")
            skipped_missing_structure_json += 1
            continue

        struct = None # Initialize struct to None
        try:
            # Parse unit cell parameters
            lattice_params = json.loads(oqmd_unit_cell_json)
            # Ensure correct number of parameters for Lattice.from_parameters
            if len(lattice_params) != 6:
                raise ValueError(f"Lattice parameters for {material_id} are not of length 6 (a,b,c,alpha,beta,gamma). Found: {lattice_params}")
            lattice = Lattice.from_parameters(*lattice_params)

            # Parse sites
            sites_data = json.loads(oqmd_sites_json)
            species_list = []
            coords_list = []
            for site_info in sites_data:
                # Example site_info: {"species": [{"element": "Li", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0]}
                # Adapt based on actual structure of 'species' and 'xyz' in your JSON
                if not isinstance(site_info.get('species'), list) or not site_info['species']:
                     raise ValueError(f"Invalid or missing species data for a site in {material_id}.")
                # Assuming simple case: one element per species entry, full occupancy
                # For more complex cases (e.g. disordered structures), this needs adjustment
                species_str = site_info['species'][0]['element'] # Take the element string
                species_list.append(species_str)
                coords_list.append(site_info['xyz'])

            if not species_list or not coords_list:
                raise ValueError(f"No species or coordinates extracted for {material_id}.")

            struct = Structure(lattice, species_list, coords_list, coords_are_cartesian=False)

        except json.JSONDecodeError as e:
            warnings.warn(f"Error decoding JSON for structure (cell or sites) for {material_id}: {e}. Skipping.")
            skipped_structure_json_parse_error += 1
            continue
        except ValueError as ve: # Catch ValueErrors from lattice/site processing
            warnings.warn(f"Error processing structure JSON data for {material_id}: {ve}. Skipping.")
            skipped_structure_json_parse_error += 1 # Or a more specific counter if needed
            continue
        except Exception as e: # Catch-all for other errors during Structure creation
            warnings.warn(f"Error creating Pymatgen Structure for {material_id} from JSON: {e}. Skipping.")
            skipped_structure_creation_error += 1
            continue

        if struct is None: # Should be caught by earlier continues, but as a safeguard
            warnings.warn(f"Structure object not created for {material_id}, reason unknown (should have been caught). Skipping.")
            # This might indicate a logic flaw if reached.
            skipped_structure_creation_error +=1 # Or a generic error counter
            continue

        # --- End Structure Reconstruction ---

        try:
            graph_dict = structure_to_graph(struct) # This function needs to exist in utils.graph_utils

            # --- Define Node Features (x) ---
            node_features = []
            if not graph_dict or 'nodes' not in graph_dict or not graph_dict['nodes']:
                warnings.warn(f"Skipping material {material_id} due to missing or empty node data from structure_to_graph.")
                skipped_processing_error +=1
                continue

            for node_data in graph_dict['nodes']:
                # Assuming 'atomic_number' and 'pauling_electronegativity' are present
                # Replace 'electronegativity' with 'pauling_electronegativity' if that's what your util provides
                an = node_data.get('atomic_number')
                en = node_data.get('electronegativity', node_data.get('pauling_electronegativity')) # Check for both common names
                if an is None or en is None:
                    warnings.warn(f"Node in {material_id} is missing atomic_number or electronegativity. Skipping material.")
                    # This skip should be outside the loop if one node feature is bad for the whole graph
                    raise ValueError("Missing node features")
                node_features.append([an, en])
            x = torch.tensor(node_features, dtype=torch.float)

            # --- Define Edge Index (edge_index) ---
            if 'edges' not in graph_dict or not graph_dict['edges']: # Handle no edges case
                # If no edges, edge_index should be empty tensor of shape [2,0]
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float) # edge_attr also empty but with correct feature dim
            else:
                source_nodes = [e['source_node_index'] for e in graph_dict['edges']]
                target_nodes = [e['target_node_index'] for e in graph_dict['edges']]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

                # --- Define Edge Features (edge_attr) ---
                edge_attributes = []
                # Check if the first edge has 'distance' to infer if others do.
                if graph_dict['edges'] and 'distance' in graph_dict['edges'][0]:
                    for edge_data in graph_dict['edges']:
                        dist = edge_data.get('distance')
                        if dist is None:
                             warnings.warn(f"Edge in {material_id} is missing distance. Skipping material.")
                             raise ValueError("Missing edge distance")
                        edge_attributes.append([dist])
                    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
                else: # No distance or no edges with distance
                    edge_attr = torch.empty((edge_index.shape[1], 0), dtype=torch.float) # Assuming 0 edge features if not present

            # --- Define Target Properties (y) ---
            # Get raw target values using new keys
            target_band_gap = material_doc.get('oqmd_band_gap')
            target_formation_energy = material_doc.get('oqmd_delta_e') # This is total energy

            # Normalize formation energy (oqmd_delta_e) by number of sites
            if target_formation_energy is not None: # Only proceed if it's not already None
                try:
                    # Attempt to convert to float here to do arithmetic
                    formation_energy_float = float(target_formation_energy)
                    num_sites = struct.num_sites # struct must be successfully created at this point
                    if num_sites > 0:
                        target_formation_energy = formation_energy_float / num_sites
                    elif num_sites == 0 and formation_energy_float != 0: # Avoid division by zero if num_sites is 0
                        warnings.warn(f"Material {material_id} has 0 sites but non-zero formation energy {formation_energy_float}. Skipping target normalization. Check data integrity.")
                        # Optionally, skip this material by setting targets to None or continue with unnormalized energy
                        # For now, we'll let it pass to the None check below, or fail at float conversion if it was a non-numeric string
                    # If num_sites is 0 and formation_energy_float is 0, it can proceed, will be [bg, 0.0]
                except ValueError:
                    # If float conversion fails here, it will be caught again below.
                    # This can happen if target_formation_energy is a string like "N/A"
                    pass # Let the None check and subsequent float conversion handle this comprehensively
                except Exception as e: # Catch any other unexpected error during normalization
                    warnings.warn(f"Error normalizing formation energy for {material_id}: {e}. Proceeding with potentially unnormalized value.")
                    # This will likely cause an issue later if target_formation_energy is not a number.

            # Check if either target is missing after attempting normalization for formation energy
            if target_band_gap is None or target_formation_energy is None:
                warnings.warn(f"Skipping material {material_id} due to missing target values (oqmd_band_gap or oqmd_delta_e after potential normalization).")
                skipped_missing_targets += 1
                continue

            # Ensure targets are numerical and not strings like "N/A"
            try:
                # target_formation_energy might already be float if normalization occurred
                targets = [float(target_band_gap), float(target_formation_energy)]
                y = torch.tensor([targets], dtype=torch.float)
            except ValueError:
                # This catches if band_gap was non-numeric, or if formation_energy was non-numeric and normalization was skipped/failed
                warnings.warn(f"Skipping material {material_id} due to non-numerical target values after processing.")
                skipped_missing_targets += 1
                continue

            data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_obj.material_id = material_id
            all_graph_data.append(data_obj)
            processed_count += 1

        except ValueError as ve: # Catch specific value errors from feature extraction
             warnings.warn(f"Value error while processing features for material {material_id}: {ve}. Skipping.")
             skipped_processing_error += 1
             continue
        except Exception as e:
            warnings.warn(f"Failed to process material {material_id} into graph object: {e}")
            skipped_processing_error += 1
            continue

    print("\n--- Processing Summary ---")
    print(f"Total raw material entries: {len(raw_materials_data)}")
    print(f"Successfully processed into graph objects: {processed_count}")
    print(f"Skipped due to missing structure JSON (cell or sites): {skipped_missing_structure_json}")
    print(f"Skipped due to error parsing structure JSON (cell or sites): {skipped_structure_json_parse_error}")
    print(f"Skipped due to error creating Pymatgen Structure object: {skipped_structure_creation_error}")
    print(f"Skipped due to missing target properties: {skipped_missing_targets}")
    print(f"Skipped due to other processing errors (graph conversion, features, etc.): {skipped_processing_error}")

    if not all_graph_data:
        print("No graph data was successfully processed. Exiting.")
        return

    print(f"\nSuccessfully processed {len(all_graph_data)} materials into graph objects.")

    # 4. Save the full list of graph objects
    if all_graph_data:
        output_dir = os.path.dirname(processed_graphs_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        torch.save(all_graph_data, processed_graphs_path)
        print(f"Saved all {len(all_graph_data)} processed graph objects to {processed_graphs_path}")
    else:
        print("No graph data was processed, so nothing to save for the full dataset.")
        print("Aborting splitting and saving of splits as there is no data.")
        return # Exit if no data to split

    # 5. Split the data
    if all_graph_data:
        num_total = len(all_graph_data)

        # Ensure we have enough data to split, at least 1 sample per set desired
        # if num_total < (1/min(train_ratio, val_ratio, test_ratio) if min(train_ratio, val_ratio, test_ratio) > 0 else float('inf')): # Avoid division by zero
        #     warnings.warn(f"Not enough data ({num_total} samples) to perform the desired split. Minimum samples required based on smallest ratio. Skipping split.", UserWarning)
        # The else: was here, it's now removed. The following block is unindented.
        # First split: Train vs. (Validation + Test)
        # train_test_split takes the size of the TEST set as parameter. So (1 - train_ratio) is val_ratio + test_ratio
        train_data, temp_data = train_test_split(
            all_graph_data,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )

        # Second split: Validation vs. Test from temp_data
        # The test_size for this split must be relative to temp_data's size
        # test_ratio_for_temp_split = test_ratio / (val_ratio + test_ratio)
        # Handle potential division by zero if val_ratio + test_ratio is 0 (though checked by sum to 1 earlier)
        if (val_ratio + test_ratio) == 0:
             if test_ratio > 0: # Trying to get a test set when val+test is 0, impossible
                 val_data = temp_data
                 test_data = []
                 warnings.warn("Validation and Test ratios are zero, but test ratio is > 0. Cannot split for test set from empty remainder.", UserWarning)
             else: # Both val and test are 0
                 val_data = temp_data # Or [] depending on desired behavior
                 test_data = []
        elif not temp_data: # If temp_data is empty (e.g. train_ratio was 1.0)
            val_data = []
            test_data = []
        else:
            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_seed, # Use same seed for reproducibility of this part of split
                shuffle=True
            )

        print(f"\nDataset split results:")
        print(f"  Training set: {len(train_data)} samples ({len(train_data)/num_total*100:.1f}%)")
        print(f"  Validation set: {len(val_data)} samples ({len(val_data)/num_total*100:.1f}%)")
        print(f"  Test set: {len(test_data)} samples ({len(test_data)/num_total*100:.1f}%)")

        # 6. Save the split datasets
        datasets_to_save = {
            "train": (train_data, train_graphs_path),
            "validation": (val_data, val_graphs_path),
            "test": (test_data, test_graphs_path),
        }

        for name, (data_list, path) in datasets_to_save.items():
            if data_list: # Only save if list is not empty
                output_dir = os.path.dirname(path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created directory for {name} data: {output_dir}")

                torch.save(data_list, path)
                print(f"Saved {name} data ({len(data_list)} samples) to {path}")
            else:
                print(f"{name.capitalize()} dataset is empty. Nothing to save to {path}.")
    else:
        # This case should ideally be caught earlier, after the processing summary.
        print("No graph data available to split. Skipping splitting and saving of splits.")


    print("\nGraph dataset creation and splitting process finished.")

if __name__ == "__main__":
    create_graph_dataset()
