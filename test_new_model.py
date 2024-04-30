from new_model import Pool, Unpool
import torch
import torch.nn as nn

def test_pool():
    # Create an instance of the Pool class
    pool = Pool(k=0.5, in_dim=16, p=0.2)
    unpool = Unpool()

    # Create dummy input tensors
    x = torch.randn(4, 8, 16)  # Node features in batches
    edge_attr = torch.randn(4, 16, 16)  # Edge attributes for each graph in the batch

    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 7, 0, 1]])  # Edge indices
    # Call the forward method of the Pool class
    new_x, new_edge_index, new_edge_index_ordered, new_edge_attr, pool_indices, pool_indices_ordered = pool(x, edge_index, edge_attr)

    # Assert the shapes of the output tensors
    print("new_x shape:", new_x.shape)  # Expected shape: (2, 3, 10)  # Expected shape: (batch_size, num_selected_nodes, node_features)
    print("new_edge_index shape:", new_edge_index.shape)  # Expected shape: (2, num_edges)
    print("new_edge_attr shape:", new_edge_attr.shape)  # Expected shape: (2, 3, 10)  # Expected shape: (batch_size, num_edges, edge_attributes)
    print("pool_indices shape:", pool_indices.shape)  # Expected shape: (2, 3)  # Expected shape: (batch_size, num_selected_nodes)
    print("pool_indices_ordered shape:", pool_indices_ordered.shape)  # Expected shape: (2, 3)  # Expected shape: (batch_size, num_selected_nodes)

    # Call the backward method of the Pool class
    unpooled_x, unpooled_edge_attr = unpool(new_x, edge_index, new_edge_attr, pool_indices, 8) 
    # Assert the shape of the unpooled tensor
    print("unpooled_x shape:", unpooled_x.shape)  # Expected shape: (2, 5, 10)  # Expected shape: (batch_size, num_nodes, node_features)
    print("unpooled_edge_attr shape:", unpooled_edge_attr.shape)  # Expected shape: (2, 5, 10)  # Expected shape: (batch_size, num_edges, edge_attributes)
    # Assert that the selected nodes are within the range of the original node indices
    assert torch.all(pool_indices >= 0) and torch.all(pool_indices < x.size(1))

    print("All test cases passed!")

# Run the test function
test_pool()