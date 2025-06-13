import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data # Added for dummy data creation
import os
import warnings

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
                print("Warning: Using DUMMY OracleNetGNN model implementation.")
            def forward(self, x, edge_index, batch, edge_attr=None):
                # A very simple aggregation for dummy purposes
                if x is None or batch is None: return torch.zeros( (int(batch.max()) + 1 if batch is not None else 1),1, device=x.device if x is not None else "cpu")

                # Sum features for each graph in the batch, assuming output is [batch_size, num_output_features]
                # This is a simplified placeholder. A real model would have GCN layers.
                out_list = []
                for i in torch.unique(batch):
                    out_list.append(torch.sum(x[batch == i], dim=0)) # Sum features for nodes in graph i

                if not out_list: # Handle empty batch case
                     return torch.zeros((0, self.fc.out_features), device=x.device if x is not None else "cpu")

                return self.fc(torch.stack(out_list))


    if "load_config" not in globals():
        def load_config(config_path): # type: ignore
            print(f"Warning: Using DUMMY load_config for {config_path}. Create utils/config_loader.py and config.yml.")
            return {}

CONFIG_PATH = 'config.yml' # Define at module level

def create_dummy_data_if_not_exists(filepath, num_samples=10, num_node_features=2, num_targets=2, gnn_target_index=0):
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name): # Ensure directory exists
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(filepath):
        print(f"Creating dummy data for {filepath}...")
        data_list = []
        for _ in range(num_samples):
            num_nodes = torch.randint(3, 10, (1,)).item()
            x = torch.randn(num_nodes, num_node_features)

            # Ensure edge_index is valid: max index < num_nodes and it's L-shaped (2, num_edges)
            if num_nodes == 0: # Handle case with no nodes if that's possible for the model
                 edge_index = torch.empty((2,0), dtype=torch.long)
                 edge_attr = torch.empty((0,1))
            else:
                num_edges = torch.randint(num_nodes -1 if num_nodes > 0 else 0, num_nodes * 2 if num_nodes > 0 else 1, (1,)).item()
                edge_source = torch.randint(0, num_nodes, (num_edges,))
                edge_target = torch.randint(0, num_nodes, (num_edges,))
                edge_index = torch.stack([edge_source, edge_target], dim=0)
                edge_attr = torch.randn(num_edges, 1) # Dummy edge features (1 per edge)

            # Ensure y matches the structure expected (1 row, num_targets columns)
            # and that the target_index is valid for these num_targets
            if gnn_target_index >= num_targets:
                actual_num_targets = gnn_target_index + 1
                # print(f"Warning: gnn_target_index {gnn_target_index} >= num_targets {num_targets}. Adjusting dummy y to have {actual_num_targets} targets for {filepath}.")
            else:
                actual_num_targets = num_targets

            y_val = torch.randn(1, actual_num_targets) # Shape [1, num_targets] as per prepare_gnn_data.py

            data = Data(x=x, edge_index=edge_index, y=y_val, edge_attr=edge_attr)
            # Add num_node_features attribute, similar to how real data might have it
            data.num_node_features = num_node_features
            data_list.append(data)
        torch.save(data_list, filepath)
        print(f"Dummy data with {num_samples} samples saved to {filepath}")


def train_gnn():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.deprecation")

    # --- Configuration Loading ---
    config_data = load_config(CONFIG_PATH)
    if config_data is None:
        config_data = {}

    gnn_config = config_data.get('gnn_params', {})
    gnn_train_graphs_path = gnn_config.get('train_graphs_path', 'data/train_graphs.pt')
    gnn_val_graphs_path = gnn_config.get('val_graphs_path', 'data/val_graphs.pt')
    gnn_model_save_path = gnn_config.get('model_save_path', 'data/oracle_net_gnn.pth')
    learning_rate = float(gnn_config.get('learning_rate', 0.001))
    batch_size = int(gnn_config.get('batch_size', 4)) # Smaller batch for dummy data
    epochs = int(gnn_config.get('epochs', 5)) # Reduced for faster dummy runs
    hidden_channels = int(gnn_config.get('hidden_channels', 16)) # Smaller for dummy data
    gnn_target_index = int(gnn_config.get('target_index', 0))

    # Parameters for dummy data generation if files are not found
    num_node_features_dummy = int(gnn_config.get('num_node_features', 2))
    num_targets_dummy = int(gnn_config.get('num_targets_in_file', 2)) # How many targets are in y in the .pt file


    print("--- GNN Training Configuration ---")
    print(f"Train graphs path: {gnn_train_graphs_path}")
    print(f"Validation graphs path: {gnn_val_graphs_path}")
    print(f"Model save path: {gnn_model_save_path}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"Target index to train on (from y): {gnn_target_index}")
    print(f"Num node features (for dummy data if created): {num_node_features_dummy}")
    print(f"Num targets in y file (for dummy data if created, y shape [1, N]): {num_targets_dummy}")
    print("---------------------------------")

    # --- Create Dummy Data if necessary ---
    if gnn_target_index >= num_targets_dummy:
        print(f"Error: gnn_target_index ({gnn_target_index}) from config is invalid for the number of targets "
              f"({num_targets_dummy}) expected in the data files (for dummy data generation). "
              f"Please ensure gnn_target_index < num_targets_in_file.")
        # Adjust num_targets_dummy to be safe for dummy generation if this check is not considered fatal
        # For now, this is a configuration error.
        return

    create_dummy_data_if_not_exists(gnn_train_graphs_path, num_samples=2 * batch_size, num_node_features=num_node_features_dummy, num_targets=num_targets_dummy, gnn_target_index=gnn_target_index)
    create_dummy_data_if_not_exists(gnn_val_graphs_path, num_samples=1 * batch_size, num_node_features=num_node_features_dummy, num_targets=num_targets_dummy, gnn_target_index=gnn_target_index)


    # --- Device Selection ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    try:
        print(f"Loading training data from {gnn_train_graphs_path}...")
        # Load full Data objects, not just weights. PyTorch 2.6+ defaults weights_only=True.
        # Set to False as these .pt files contain complex torch_geometric Data objects.
        train_dataset = torch.load(gnn_train_graphs_path, map_location=device, weights_only=False)
        print(f"Loading validation data from {gnn_val_graphs_path}...")
        # Load full Data objects, not just weights.
        val_dataset = torch.load(gnn_val_graphs_path, map_location=device, weights_only=False)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}. Although dummy data creation was attempted, something went wrong or path is incorrect.")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    if not train_dataset or not val_dataset: # Check if lists are empty
        print("Error: Training or validation dataset is empty (no data objects loaded). Cannot proceed.")
        return
    if not isinstance(train_dataset, list) or not isinstance(val_dataset, list) or not all(isinstance(item, Data) for item in train_dataset+val_dataset):
        print(f"Error: Loaded data is not in the expected format (list of Data objects). Type train: {type(train_dataset)}, Type val: {type(val_dataset)}")
        if isinstance(train_dataset, list) and train_dataset: print(f" Type of first train element: {type(train_dataset[0])}")
        return


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Initialization ---
    # Determine num_node_features from the actual loaded data
    # This overrides num_node_features_dummy for model instantiation
    actual_num_node_features = None
    if train_dataset and hasattr(train_dataset[0], 'num_node_features'):
        actual_num_node_features = train_dataset[0].num_node_features
    elif train_dataset and hasattr(train_dataset[0], 'x') and train_dataset[0].x is not None:
         actual_num_node_features = train_dataset[0].x.shape[1]

    if actual_num_node_features is None:
        print(f"CRITICAL ERROR: Could not determine num_node_features from loaded data in {gnn_train_graphs_path}.")
        print("Ensure your data objects have 'num_node_features' attribute or 'x' attribute with feature data.")
        return

    print(f"Using actual num_node_features from loaded data: {actual_num_node_features}")

    model = OracleNetGNN(num_node_features=actual_num_node_features,
                         hidden_channels=hidden_channels,
                         num_graph_features_output=1).to(device) # Output is 1 value
    print("\n--- Model Architecture ---")
    print(model)
    print("------------------------\n")

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # --- Main Training Loop ---
    best_val_loss = float('inf')
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        epoch_train_loss = 0
        processed_train_batches = 0
        for batch_idx, data in enumerate(train_loader):
            if not hasattr(data, 'x') or data.x is None or \
               not hasattr(data, 'edge_index') or data.edge_index is None or \
               not hasattr(data, 'batch') or data.batch is None:
                print(f"Warning: Train Batch {batch_idx} is missing x, edge_index, or batch. Skipping.")
                continue
            if not hasattr(data, 'y') or data.y is None:
                print(f"Warning: Train Batch {batch_idx} has no attribute 'y' or y is None. Skipping batch.")
                continue
            if data.y.shape[1] <= gnn_target_index:
                print(f"Error: gnn_target_index ({gnn_target_index}) is out of bounds for train data.y with shape {data.y.shape}. Skipping batch.")
                continue

            data = data.to(device)
            optimizer.zero_grad()

            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

            out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)

            target = data.y[:, gnn_target_index].unsqueeze(1) # Shape [batch_size, 1]
            target = target.to(device)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            processed_train_batches += 1

        avg_epoch_train_loss = epoch_train_loss / processed_train_batches if processed_train_batches > 0 else 0

        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        processed_val_batches = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                if not hasattr(data, 'x') or data.x is None or \
                   not hasattr(data, 'edge_index') or data.edge_index is None or \
                   not hasattr(data, 'batch') or data.batch is None:
                    print(f"Warning: Validation Batch {batch_idx} is missing x, edge_index, or batch. Skipping.")
                    continue
                if not hasattr(data, 'y') or data.y is None:
                    print(f"Warning: Val Batch {batch_idx} has no attribute 'y' or y is None. Skipping batch.")
                    continue
                if data.y.shape[1] <= gnn_target_index:
                    print(f"Error: gnn_target_index ({gnn_target_index}) is out of bounds for val data.y with shape {data.y.shape}. Skipping batch.")
                    continue

                data = data.to(device)
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)

                target = data.y[:, gnn_target_index].unsqueeze(1) # Shape [batch_size, 1]
                target = target.to(device)

                loss = criterion(out, target)
                epoch_val_loss += loss.item()
                processed_val_batches +=1

        avg_epoch_val_loss = epoch_val_loss / processed_val_batches if processed_val_batches > 0 else 0

        print(f"Epoch: {epoch:03d}, Avg Train Loss: {avg_epoch_train_loss:.4f}, Avg Val Loss: {avg_epoch_val_loss:.4f}")

        # Model Saving
        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            # Ensure directory for model saving exists
            model_save_dir = os.path.dirname(gnn_model_save_path)
            if model_save_dir and not os.path.exists(model_save_dir): # Check if model_save_dir is not empty
                os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), gnn_model_save_path)
            print(f"Model saved to {gnn_model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if os.path.exists(gnn_model_save_path) and best_val_loss != float('inf'):
        print(f"Saved model path: {gnn_model_save_path}")
    else:
        print(f"Model not saved (or not improved). Check save path or validation loss: {gnn_model_save_path}")
    print("-------------------------\n")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in os.sys.path:
        os.sys.path.insert(0, PROJECT_ROOT)

    try:
        # Re-import to ensure the non-dummy versions are used if available after path adjustment
        from models.gnn_oracle_net import OracleNetGNN
        from utils.config_loader import load_config
    except ImportError:
        print("Warning: Still unable to import local modules after path adjustment. Previously defined dummy versions might be used.")
        pass # Fallbacks are defined at the top


    # Check if the actual modules were loaded, or if we are using the dummy ones defined in this file.
    # This relies on __module__ being set to this file's name for the dummy classes/functions.
    using_dummy_gnn = 'OracleNetGNN' in globals() and OracleNetGNN.__module__ == __name__
    using_dummy_config = 'load_config' in globals() and (hasattr(load_config, '__module__') and load_config.__module__ == __name__) # type: ignore

    if using_dummy_gnn or using_dummy_config:
        print("\nINFO: Running train_gnn_model.py with DUMMY/FALLBACK versions for:")
        if using_dummy_gnn:
            print("  - OracleNetGNN model")
        if using_dummy_config:
            print("  - load_config function")
        print("This is expected if 'models' or 'utils' are not found or if run as a standalone script without these modules.")
        print("The script will attempt to use these dummy versions. For full functionality, ensure modules are available.\n")

    train_gnn()
