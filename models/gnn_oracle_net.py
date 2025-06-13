import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class OracleNetGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_graph_features_output=1):
        super(OracleNetGNN, self).__init__()
        torch.manual_seed(42) # For reproducibility
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_graph_features_output)

    def forward(self, x, edge_index, batch, edge_attr=None): # Added edge_attr to match Data object
        # 1. Obtain node embeddings
        edge_weight = None
        if edge_attr is not None:
            if edge_attr.ndim > 1 and edge_attr.shape[1] == 1:
                edge_weight = edge_attr.squeeze()
            elif edge_attr.ndim == 1: # Already squeezed
                edge_weight = edge_attr
            else:
                # Handle other cases or raise an error if edge_attr is not as expected
                # For GCNConv, edge_weight should be 1D or None.
                # If edge_attr has more than one feature, GCNConv cannot use it directly as edge_weight.
                # In such cases, one might need to process edge_attr differently or use a different conv layer.
                pass # Defaults to None

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier/regressor
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

if __name__ == '__main__':
    # Example usage:
    # Assuming 'data/train_graphs.pt' exists and contains Data objects
    # This is just for basic testing of the model structure.
    # In a real scenario, this would be part of the training script.
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader # Corrected import for DataLoader

        # Create a dummy Data object
        # Node features: atomic_number, electronegativity
        dummy_x = torch.tensor([[8, 2.0], [1, 2.2], [1, 2.2]], dtype=torch.float)
        # Edge index: connections
        dummy_edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long).t().contiguous()
        # Edge attributes: distance (optional for GCNConv, but good to show)
        dummy_edge_attr = torch.tensor([[1.5], [1.6]], dtype=torch.float)
        dummy_y = torch.tensor([10.0], dtype=torch.float) # Example target value
        dummy_batch = torch.tensor([0, 0, 0], dtype=torch.long) # Single graph in batch

        dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, y=dummy_y)
        # Note: batch attribute is typically added by DataLoader

        # For multiple graphs in a batch, use DataLoader
        data_list = [dummy_data, dummy_data] # Create a list of Data objects
        loader = DataLoader(data_list, batch_size=2)
        batch_data = next(iter(loader))

        print("Testing OracleNetGNN model...")
        # num_node_features should match the input data (e.g., 2 for atomic_number, electronegativity)
        model = OracleNetGNN(num_node_features=2, hidden_channels=64, num_graph_features_output=1)
        print(model)

        # Test with single Data object (manually add batch vector for single graph)
        # In practice, you'd likely use DataLoader even for a single graph if it's part of a dataset.
        # For a truly single, isolated graph not part of a batch:
        single_data_for_model = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, batch=dummy_batch)
        output_single = model(single_data_for_model.x, single_data_for_model.edge_index, single_data_for_model.batch, single_data_for_model.edge_attr)
        print(f"Output shape for single Data object: {output_single.shape}") # Expected: [1, 1] (1 graph, 1 output feature)
        print(f"Output for single Data object: {output_single}")

        # Test with batched Data object from DataLoader
        output_batch = model(batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.edge_attr)
        print(f"Output shape for batched Data object: {output_batch.shape}") # Expected: [2, 1] (2 graphs, 1 output feature each)
        print(f"Output for batched Data object: {output_batch}")

    except ImportError as e:
        print(f"PyTorch or PyTorch Geometric not installed, skipping GNN model example. Error: {e}")
    except Exception as e:
        print(f"An error occurred during GNN model example: {e}")
