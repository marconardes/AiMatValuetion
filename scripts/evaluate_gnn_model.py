import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data # Added for dummy data creation
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
import math # For sqrt

# Attempt to import local modules
try:
    from models.gnn_oracle_net import OracleNetGNN
    from utils.config_loader import load_config
except ImportError as e:
    print(f"Error importing local modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    if "OracleNetGNN" not in globals():
        class OracleNetGNN(torch.nn.Module): # type: ignore
            def __init__(self, num_node_features, hidden_channels, num_graph_features_output=1):
                super().__init__()
                self.fc = torch.nn.Linear(num_node_features, num_graph_features_output)
                print("Warning: Using DUMMY OracleNetGNN model implementation for evaluation.")
            def forward(self, x, edge_index, batch, edge_attr=None):
                if x is None or batch is None: return torch.zeros( (int(batch.max()) + 1 if batch is not None else 1),1, device=x.device if x is not None else "cpu")
                out_list = []
                for i in torch.unique(batch):
                    out_list.append(torch.sum(x[batch == i], dim=0))
                if not out_list: return torch.zeros((0, self.fc.out_features), device=x.device if x is not None else "cpu")
                return self.fc(torch.stack(out_list))

    if "load_config" not in globals():
        def load_config(config_path): # type: ignore
            print(f"Warning: Using DUMMY load_config for {config_path} in evaluation.")
            return {}

CONFIG_PATH = 'config.yml'

# Re-using/adapting dummy data creation from training script
def create_dummy_data_if_not_exists(filepath, num_samples=10, num_node_features=2, num_targets=2, gnn_target_index=0, file_description="data"):
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(filepath):
        print(f"Creating dummy {file_description} for {filepath}...")
        data_list = []
        for i in range(num_samples): # Changed _ to i for dummy_id
            num_nodes = torch.randint(3, 10, (1,)).item()
            x = torch.randn(num_nodes, num_node_features)
            if num_nodes == 0:
                 edge_index = torch.empty((2,0), dtype=torch.long)
                 edge_attr = torch.empty((0,1))
            else:
                num_edges = torch.randint(num_nodes -1 if num_nodes > 0 else 0, num_nodes * 2 if num_nodes > 0 else 1, (1,)).item()
                edge_source = torch.randint(0, num_nodes, (num_edges,))
                edge_target = torch.randint(0, num_nodes, (num_edges,))
                edge_index = torch.stack([edge_source, edge_target], dim=0)
                edge_attr = torch.randn(num_edges, 1)

            if gnn_target_index >= num_targets:
                actual_num_targets = gnn_target_index + 1
            else:
                actual_num_targets = num_targets
            y_val = torch.randn(1, actual_num_targets)

            data = Data(x=x, edge_index=edge_index, y=y_val, edge_attr=edge_attr)
            data.material_id = f"dummy_id_{i}" # Add dummy material_id
            data.num_node_features = num_node_features
            data_list.append(data)
        torch.save(data_list, filepath)
        print(f"Dummy {file_description} with {num_samples} samples saved to {filepath}")


def evaluate_gnn():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.deprecation")

    # --- Configuration Loading ---
    config_data = load_config(CONFIG_PATH)
    if config_data is None: config_data = {}

    gnn_config = config_data.get('gnn_params', {})
    gnn_test_graphs_path = gnn_config.get('test_graphs_path', 'data/test_graphs.pt')
    gnn_train_graphs_path = gnn_config.get('train_graphs_path', 'data/train_graphs.pt') # For random baseline
    gnn_model_load_path = gnn_config.get('model_save_path', 'data/oracle_net_gnn.pth') # Path to trained model

    hidden_channels = int(gnn_config.get('hidden_channels', 16)) # Must match saved model
    gnn_target_index = int(gnn_config.get('target_index', 0))
    batch_size = int(gnn_config.get('batch_size', 4))
    num_top_errors_to_show = int(gnn_config.get('gnn_num_top_errors_to_show', 5)) # For error analysis

    num_node_features_dummy = int(gnn_config.get('num_node_features', 2))
    num_targets_dummy = int(gnn_config.get('num_targets_in_file', 2))

    print("--- GNN Evaluation Configuration ---")
    print(f"Test graphs path: {gnn_test_graphs_path}")
    print(f"Train graphs path (for baseline): {gnn_train_graphs_path}")
    print(f"Trained model path: {gnn_model_load_path}")
    print(f"Hidden channels (for model init): {hidden_channels}")
    print(f"Target index: {gnn_target_index}")
    print(f"Batch size: {batch_size}")
    print(f"Number of top errors to show: {num_top_errors_to_show}")
    print("------------------------------------")

    # --- Dummy Data Generation (if needed) ---
    if gnn_target_index >= num_targets_dummy:
        print(f"Error: gnn_target_index ({gnn_target_index}) is invalid for num_targets_dummy ({num_targets_dummy}).")
        return

    create_dummy_data_if_not_exists(gnn_test_graphs_path, num_samples=batch_size, num_node_features=num_node_features_dummy, num_targets=num_targets_dummy, gnn_target_index=gnn_target_index, file_description="test data")
    create_dummy_data_if_not_exists(gnn_train_graphs_path, num_samples=batch_size, num_node_features=num_node_features_dummy, num_targets=num_targets_dummy, gnn_target_index=gnn_target_index, file_description="train data (for baseline)")

    # --- Device Selection ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Test Data ---
    try:
        print(f"Loading test data from {gnn_test_graphs_path}...")
        test_dataset = torch.load(gnn_test_graphs_path)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {gnn_test_graphs_path}. Exiting.")
        return
    except Exception as e:
        print(f"An error occurred while loading test data: {e}. Exiting.")
        return

    if not test_dataset or not isinstance(test_dataset, list) or not all(isinstance(item, Data) for item in test_dataset):
        print(f"Error: Test dataset is empty or not in the expected format. Path: {gnn_test_graphs_path}")
        return

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine num_node_features from the test data for model instantiation
    actual_num_node_features = None
    if test_dataset and hasattr(test_dataset[0], 'num_node_features'):
        actual_num_node_features = test_dataset[0].num_node_features
    elif test_dataset and hasattr(test_dataset[0], 'x') and test_dataset[0].x is not None:
        actual_num_node_features = test_dataset[0].x.shape[1]

    if actual_num_node_features is None:
        print(f"CRITICAL ERROR: Could not determine num_node_features from loaded test data in {gnn_test_graphs_path}.")
        return
    print(f"Inferred num_node_features from test data: {actual_num_node_features}")

    # --- Load Trained Model ---
    model = OracleNetGNN(num_node_features=actual_num_node_features,
                         hidden_channels=hidden_channels,
                         num_graph_features_output=1) # Output is 1 value
    try:
        if not os.path.exists(gnn_model_load_path):
            print(f"Error: Trained model file not found at {gnn_model_load_path}.")
            print("Please ensure the model is trained and path is correct in config.yml or run training script.")
            print("Proceeding with uninitialized model for baseline calculation only.")
            trained_model_loaded = False
        else:
            model.load_state_dict(torch.load(gnn_model_load_path, map_location=device))
            print(f"Trained model loaded from {gnn_model_load_path}")
            trained_model_loaded = True
    except Exception as e:
        print(f"Error loading model state_dict: {e}. Ensure model architecture matches saved weights.")
        print("Proceeding with uninitialized model for baseline calculation only.")
        trained_model_loaded = False

    model.to(device)
    model.eval()

    # --- GNN Predictions on Test Set ---
    all_predictions_list_gnn = []
    all_targets_list_gnn = []
    all_material_ids_gnn = []
    processed_samples_count = 0


    if trained_model_loaded:
        print("Evaluating GNN model on the test set...")
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if not hasattr(data, 'y') or data.y is None or data.y.shape[1] <= gnn_target_index:
                    print(f"Warning: Skipping a batch in test_loader due to missing or malformed 'y' (target_index: {gnn_target_index}, y_shape: {data.y.shape if hasattr(data, 'y') else 'N/A'}).")
                    continue
                data = data.to(device)
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                pred = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)

                target = data.y[:, gnn_target_index].unsqueeze(1)

                all_predictions_list_gnn.append(pred.cpu().numpy())
                all_targets_list_gnn.append(target.cpu().numpy())

                # Collect material IDs
                batch_material_ids = data.material_id if hasattr(data, 'material_id') else [f"N/A_batch{i}_idx{j}" for j in range(data.num_graphs)]
                if isinstance(batch_material_ids, list): # DataLoader might batch them as a list of lists/strings
                    all_material_ids_gnn.extend(batch_material_ids)
                else: # If it's a tensor or other structure, handle appropriately or log warning
                    try:
                        all_material_ids_gnn.extend(batch_material_ids.tolist()) # Example for tensor
                    except:
                        all_material_ids_gnn.extend([f"ErrorID_batch{i}_idx{j}" for j in range(data.num_graphs)])
                processed_samples_count += data.num_graphs


        if all_predictions_list_gnn and all_targets_list_gnn:
            all_predictions_gnn = np.concatenate(all_predictions_list_gnn).flatten()
            all_targets_gnn = np.concatenate(all_targets_list_gnn).flatten()

            # Ensure material_ids list matches the length of predictions/targets
            # This might need adjustment if some graphs in a batch are skipped.
            # For now, assuming all_material_ids_gnn is correctly populated per graph.
            if len(all_material_ids_gnn) != len(all_predictions_gnn):
                 print(f"Warning: Length of material_ids ({len(all_material_ids_gnn)}) does not match predictions ({len(all_predictions_gnn)}). Error analysis might be misaligned.")
                 # Pad material_ids if necessary, though ideally this shouldn't happen with per-graph ID collection
                 all_material_ids_gnn.extend(["ErrPad_ID"] * (len(all_predictions_gnn) - len(all_material_ids_gnn)))


            # --- Calculate GNN Metrics ---
            mae_gnn = mean_absolute_error(all_targets_gnn, all_predictions_gnn)
            mse_gnn = mean_squared_error(all_targets_gnn, all_predictions_gnn)
            rmse_gnn = math.sqrt(mse_gnn)

            print("\n--- GNN Model Performance ---")
            print(f"MAE: {mae_gnn:.4f}")
            print(f"RMSE: {rmse_gnn:.4f}")
            print("-----------------------------")

            # --- Basic Error Analysis ---
            print("\n--- Basic Error Analysis ---")
            abs_errors = np.abs(all_predictions_gnn - all_targets_gnn)

            error_data = []
            for i in range(len(abs_errors)):
                mat_id = all_material_ids_gnn[i] if i < len(all_material_ids_gnn) else f"Unknown_ID_idx_{i}"
                error_data.append({
                    "material_id": mat_id,
                    "true_value": all_targets_gnn[i],
                    "predicted_value": all_predictions_gnn[i],
                    "absolute_error": abs_errors[i]
                })

            sorted_error_data = sorted(error_data, key=lambda x: x['absolute_error'], reverse=True)

            print(f"Top {num_top_errors_to_show} predictions with highest absolute error:")
            for i in range(min(num_top_errors_to_show, len(sorted_error_data))):
                item = sorted_error_data[i]
                print(f"  Material ID: {item['material_id']}, True: {item['true_value']:.4f}, Predicted: {item['predicted_value']:.4f}, Abs Error: {item['absolute_error']:.4f}")
            print("-----------------------------")

            # --- Placeholder for XAI Integration (Priority 10) ---
            # Future work: Integrate GNN explainability techniques here to understand model predictions.
            # Examples:
            # - GNNExplainer: Identify important subgraphs or node features for specific predictions.
            # - Attention analysis: If using GAT or similar attention-based models, visualize attention weights.
            # - Saliency maps: Highlight which parts of the input graph most influence the output.
            #
            # This could involve:
            # 1. Iterating through specific (e.g., high-error or correctly predicted) test samples.
            # 2. For each sample, applying an XAI method to get an explanation.
            # 3. Visualizing or logging these explanations.
            # For instance, after identifying top errors or interesting cases:
            # # Assuming 'test_dataset' is available and 'sorted_error_data' contains indices or can be mapped back
            # interesting_sample_indices = [test_dataset.index_select(entry['material_id']) for entry in sorted_error_data[:num_top_errors_to_show]] # This needs careful implementation based on how IDs map to dataset indices
            # for sample_idx in interesting_sample_indices: # This is conceptual
            #     if sample_idx is None: continue # If material_id couldn't be mapped to an index
            #     data_sample = test_dataset[sample_idx]
            #     # Ensure data_sample is a single Data object and on the correct device for the model
            #     # explanation = compute_gnn_explainer(model, data_sample.to(device), device) # Example
            #     # log_or_visualize_explanation(explanation, data_sample.material_id) # Example
            # --- End of XAI Placeholder ---

        else:
            print("No valid GNN predictions were made, possibly due to issues with test data targets.")
            mae_gnn, rmse_gnn = float('nan'), float('nan')
            all_predictions_gnn, all_targets_gnn = None, None # Ensure these are None for baseline
    else:
        print("Skipping GNN model evaluation as the trained model was not loaded.")
        mae_gnn, rmse_gnn = float('nan'), float('nan')
        all_predictions_gnn, all_targets_gnn = None, None # Ensure these are None for baseline


    # --- Random Baseline ---
    print("\nCalculating Random Baseline performance...")
    try:
        train_dataset_for_baseline = torch.load(gnn_train_graphs_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {gnn_train_graphs_path} for baseline calculation. Exiting baseline.")
        return # Cannot proceed with baseline if this fails
    except Exception as e:
        print(f"An error occurred loading training data for baseline: {e}. Exiting baseline.")
        return

    if not train_dataset_for_baseline or not isinstance(train_dataset_for_baseline, list) or not all(isinstance(item, Data) for item in train_dataset_for_baseline):
        print(f"Error: Training dataset for baseline is empty or not in expected format. Path: {gnn_train_graphs_path}")
        return

    train_targets_for_baseline = []
    for data in train_dataset_for_baseline:
        if hasattr(data, 'y') and data.y is not None and data.y.shape[1] > gnn_target_index:
            train_targets_for_baseline.append(data.y[:, gnn_target_index].item()) # Assuming y is [1, num_targets]
        # else:
            # print(f"Warning: A graph in train_dataset_for_baseline has missing or malformed 'y' (target_index: {gnn_target_index}). Skipping for baseline mean calculation.")

    if not train_targets_for_baseline:
        print("Error: No valid targets found in the training data to calculate baseline mean.")
        return

    mean_train_target = np.mean(train_targets_for_baseline)
    print(f"Mean target value from training set (used for baseline): {mean_train_target:.4f}")

    # Use actual targets from the GNN evaluation if available, otherwise re-extract from test_loader
    # This ensures consistency if some test batches were skipped for GNN due to target issues
    # but might be usable for baseline (though unlikely if target structure is the problem).

    # Re-extract all_targets from test_loader for baseline to ensure it's complete
    # irrespective of GNN prediction loop issues.
    all_targets_for_baseline_eval = []
    for data in test_loader: # Iterate through test_loader again
        if hasattr(data, 'y') and data.y is not None and data.y.shape[1] > gnn_target_index:
            target = data.y[:, gnn_target_index].unsqueeze(1)
            all_targets_for_baseline_eval.append(target.cpu().numpy())

    if not all_targets_for_baseline_eval:
        print("Error: No valid targets found in test data for baseline evaluation.")
        return

    all_targets_for_baseline_eval = np.concatenate(all_targets_for_baseline_eval).flatten()

    baseline_predictions = np.full_like(all_targets_for_baseline_eval, fill_value=mean_train_target)

    mae_baseline = mean_absolute_error(all_targets_for_baseline_eval, baseline_predictions)
    mse_baseline = mean_squared_error(all_targets_for_baseline_eval, baseline_predictions)
    rmse_baseline = math.sqrt(mse_baseline)

    print("\n--- Random Baseline Performance (predicting mean of train targets) ---")
    print(f"MAE: {mae_baseline:.4f}")
    print(f"RMSE: {rmse_baseline:.4f}")
    print("--------------------------------------------------------------------")

    # --- Comparison ---
    print("\n--- Comparison Summary ---")
    if trained_model_loaded:
        print(f"GNN MAE: {mae_gnn:.4f}  |  Random Baseline MAE: {mae_baseline:.4f}")
        print(f"GNN RMSE: {rmse_gnn:.4f} |  Random Baseline RMSE: {rmse_baseline:.4f}")
    else:
        print("GNN Model not evaluated. Cannot compare.")
        print(f"Random Baseline MAE: {mae_baseline:.4f}")
        print(f"Random Baseline RMSE: {rmse_baseline:.4f}")
    print("--------------------------\n")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in os.sys.path:
        os.sys.path.insert(0, PROJECT_ROOT)

    try:
        from models.gnn_oracle_net import OracleNetGNN
        from utils.config_loader import load_config
    except ImportError:
        print("Warning (evaluate_gnn): Still unable to import local modules. Dummy versions might be used.")
        pass

    using_dummy_gnn = 'OracleNetGNN' in globals() and OracleNetGNN.__module__ == __name__
    using_dummy_config = 'load_config' in globals() and (hasattr(load_config, '__module__') and load_config.__module__ == __name__) # type: ignore

    if using_dummy_gnn or using_dummy_config:
        print("\nINFO (evaluate_gnn): Running with DUMMY/FALLBACK versions for:")
        if using_dummy_gnn: print("  - OracleNetGNN model")
        if using_dummy_config: print("  - load_config function")
        print("This is expected if 'models' or 'utils' are not found or if run as a standalone script.\n")

    evaluate_gnn()
