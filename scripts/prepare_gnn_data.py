import json
import os
import warnings

import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure

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
    raw_data_path = prepare_config.get('raw_data_filename', 'data/mp_raw_data.json')
    processed_graphs_path = prepare_config.get('processed_graphs_filename', 'data/processed_graphs.pt')
    train_graphs_path = prepare_config.get('train_graphs_filename', 'data/train_graphs.pt')
    val_graphs_path = prepare_config.get('val_graphs_filename', 'data/val_graphs.pt')
    test_graphs_path = prepare_config.get('test_graphs_filename', 'data/test_graphs.pt')
    random_seed = prepare_config.get('random_seed', 42)
    train_ratio = prepare_config.get('train_ratio', 0.7)
    val_ratio = prepare_config.get('val_ratio', 0.2)
    test_ratio = prepare_config.get('test_ratio', 0.1)

    if not (train_ratio + val_ratio + test_ratio == 1.0): # Basic check
        warnings.warn(f"Train ({train_ratio}), validation ({val_ratio}), and test ({test_ratio}) ratios do not sum to 1.0. Please check config.yml.", UserWarning)
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
        with open(raw_data_path, 'r') as f:
            raw_materials_data = json.load(f)
        print(f"Loaded {len(raw_materials_data)} raw material entries from {raw_data_path}.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {raw_data_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading {raw_data_path}: {e}")
        return

    all_graph_data = []
    processed_count = 0
    skipped_missing_cif = 0
    skipped_cif_parse_error = 0
    skipped_missing_targets = 0
    skipped_processing_error = 0

    # 3. Process each material into a torch_geometric.data.Data object
    for i, material_doc in enumerate(raw_materials_data):
        material_id = material_doc.get('material_id', f"Unknown_ID_{i+1}")
        print(f"Processing material {i+1}/{len(raw_materials_data)}: {material_id}")

        cif_string = material_doc.get('cif') # Corrected key from 'cif_string' to 'cif' based on common usage
        if not cif_string:
            warnings.warn(f"Skipping material {material_id} due to missing CIF string.")
            skipped_missing_cif += 1
            continue

        try:
            struct = Structure.from_str(cif_string, fmt="cif")
        except Exception as e:
            warnings.warn(f"Error parsing CIF for {material_id}: {e}. Skipping.")
            skipped_cif_parse_error += 1
            continue

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
            target_band_gap = material_doc.get('band_gap_mp')
            target_formation_energy = material_doc.get('formation_energy_per_atom_mp')

            if target_band_gap is None or target_formation_energy is None:
                warnings.warn(f"Skipping material {material_id} due to missing target values (band_gap_mp or formation_energy_per_atom_mp).")
                skipped_missing_targets += 1
                continue

            # Ensure targets are numerical and not strings like "N/A"
            try:
                targets = [float(target_band_gap), float(target_formation_energy)]
                y = torch.tensor([targets], dtype=torch.float)
            except ValueError:
                warnings.warn(f"Skipping material {material_id} due to non-numerical target values.")
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
    print(f"Skipped due to missing CIF string: {skipped_missing_cif}")
    print(f"Skipped due to CIF parsing error: {skipped_cif_parse_error}")
    print(f"Skipped due to missing target properties: {skipped_missing_targets}")
    print(f"Skipped due to other processing errors (features, etc.): {skipped_processing_error}")

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
        if num_total < (1/min(train_ratio, val_ratio, test_ratio) if min(train_ratio, val_ratio, test_ratio) > 0 else float('inf')): # Avoid division by zero
            warnings.warn(f"Not enough data ({num_total} samples) to perform the desired split. Minimum samples required based on smallest ratio. Skipping split.", UserWarning)
        else:
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
