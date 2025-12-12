import torch
import pickle
import numpy as np
import os
import hashlib
import time
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def Dataloader(x, edge_index, edge_attr, y):
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def load_pyg_data_from_pickle(pkl_path):
    """
    Function to load PyTorch Geometric data directly from a pickle file
    
    Args:
        pkl_path (str): Pickle file path
        
    Returns:
        Data: PyTorch Geometric Data object
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

class GraphDataset(Dataset):
    """Graph Dataset Class"""
    def __init__(self, data_list):
        super(GraphDataset, self).__init__()
        self.data_list = data_list
        
    def len(self):
        return len(self.data_list)
        
    def get(self, idx):
        return self.data_list[idx]

def get_cache_path(pkl_path, train_ratio, val_ratio, cache_dir='./data_convert/original_data/cache_data',
                   balance_classes=False, max_samples_per_class=None, class_balance_ratio=1.0,
                   samples_per_class_dict=None, normalize_coords=True, coord_scale_factor=0.001):
    """
    Generates cache file paths.
    
    Args:
        pkl_path (str): Original pickle file path
        train_ratio (float): Training data ratio
        val_ratio (float): Validation data ratio
        cache_dir (str): Cache directory path
        balance_classes (bool): Whether to balance classes
        max_samples_per_class (int): Maximum samples per class
        class_balance_ratio (float): Sample ratio for class 2
        samples_per_class_dict (dict): Dictionary of sample counts per class
        normalize_coords (bool): Whether to normalize coordinates
        coord_scale_factor (float): Coordinate scale factor
        
    Returns:
        tuple: Train, validation, test cache file paths
    """
    # Extract base filename
    base_filename = os.path.basename(pkl_path).split('.')[0]
    
    # Convert samples_per_class_dict to string (for hash generation)
    samples_dict_str = "None"
    if samples_per_class_dict is not None:
        # Convert dictionary to sorted string (e.g., "0:1000,1:2000,2:1500")
        samples_dict_str = ",".join([f"{k}:{v}" for k, v in sorted(samples_per_class_dict.items())])
    
    # Generate parameter hash (include all parameters, including samples_per_class_dict)
    params_str = f"{pkl_path}_{train_ratio}_{val_ratio}_{balance_classes}_{max_samples_per_class}_{class_balance_ratio}_{samples_dict_str}_{normalize_coords}_{coord_scale_factor}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:10]
    
    # Debug: Print hash info if samples_per_class_dict is set
    if samples_per_class_dict is not None:
        print(f"Cache hash generation info: samples_per_class_dict={samples_per_class_dict}, hash={params_hash}")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache file paths
    train_cache = os.path.join(cache_dir, f"{base_filename}_train_{params_hash}.pkl")
    val_cache = os.path.join(cache_dir, f"{base_filename}_val_{params_hash}.pkl")
    test_cache = os.path.join(cache_dir, f"{base_filename}_test_{params_hash}.pkl")
    
    return train_cache, val_cache, test_cache

def normalize_coordinates(data, normalize_coords=True, coord_scale_factor=0.001):
    """
    Function to normalize coordinate information
    
    Args:
        data: PyTorch Geometric Data object
        normalize_coords: Whether to normalize coordinates
        coord_scale_factor: Coordinate scale factor (convert 4K coordinates to normal coordinates)
    
    Returns:
        Data: Data object with normalized coordinates
    """
    if not normalize_coords:
        return data
    
    print(f"Applying coordinate normalization... (Scale factor: {coord_scale_factor})")
    
    # Handle infinite and NaN values
    data.x[torch.isinf(data.x)] = 0
    data.x[torch.isnan(data.x)] = 0
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr[torch.isinf(data.edge_attr)] = 0
        data.edge_attr[torch.isnan(data.edge_attr)] = 0
    
    # Normalize coordinate features (usually features 0 and 1)
    if data.x.shape[1] >= 2:
        # Normalize X, Y coordinates by dividing by scale factor
        data.x[:, 0] = data.x[:, 0] * coord_scale_factor  # X coordinate
        data.x[:, 1] = data.x[:, 1] * coord_scale_factor  # Y coordinate
        
        # Normalize edge attributes as well (if coordinate-related attributes exist)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[1] >= 2:
            data.edge_attr[:, 0] = data.edge_attr[:, 0] * coord_scale_factor
            data.edge_attr[:, 1] = data.edge_attr[:, 1] * coord_scale_factor
    
    # Apply StandardScaler to all features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Normalize node features
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
    
    # Normalize edge features
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = torch.tensor(scaler.fit_transform(data.edge_attr.numpy()), dtype=torch.float32)
    
    print(f"Coordinate normalization completed")
    print(f"  Node feature range: min={data.x.min().item():.4f}, max={data.x.max().item():.4f}")
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"  Edge feature range: min={data.edge_attr.min().item():.4f}, max={data.edge_attr.max().item():.4f}")
    
    return data

def balance_class_samples(data, balance_classes=False, max_samples_per_class=None, class_balance_ratio=1.0, samples_per_class_dict=None):
    """
    Function to handle class imbalance
    
    Args:
        data: PyTorch Geometric Data object
        balance_classes: Whether to balance classes
        max_samples_per_class: Maximum samples per class (applied equally to all classes)
        class_balance_ratio: Sample ratio for class 2 (1.0 means original, 0.5 means half)
        samples_per_class_dict: Dictionary of sample counts per class (e.g., {0: 1000, 1: 2000, 2: 1500})
                              If this parameter is set, it takes precedence over max_samples_per_class
    
    Returns:
        Data: Data object with balanced classes
    """
    if not balance_classes:
        return data
    
    # Collect indices per class
    unique_labels = torch.unique(data.y)
    class_indices = {}
    
    for label in unique_labels:
        class_indices[label.item()] = torch.where(data.y == label)[0]
    
    # Calculate sample counts per class
    class_counts = {label: len(indices) for label, indices in class_indices.items()}
    print(f"Original class distribution: {class_counts}")
    
    # Adjust sample count for class 2 (only if samples_per_class_dict is not provided)
    if samples_per_class_dict is None and class_balance_ratio < 1.0 and 2 in class_indices:
        target_count_class2 = int(len(class_indices[2]) * class_balance_ratio)
        if target_count_class2 < len(class_indices[2]):
            # Random sampling from class 2
            selected_indices_class2 = torch.randperm(len(class_indices[2]))[:target_count_class2]
            class_indices[2] = class_indices[2][selected_indices_class2]
            print(f"Adjusting class 2 sample count: {class_counts[2]} -> {len(class_indices[2])}")
    
    # Apply different sample counts per class (Priority 1)
    if samples_per_class_dict is not None:
        for label in class_indices:
            if label in samples_per_class_dict:
                target_count = samples_per_class_dict[label]
                if target_count < len(class_indices[label]):
                    # Random sampling with specified sample count
                    selected_indices = torch.randperm(len(class_indices[label]))[:target_count]
                    class_indices[label] = class_indices[label][selected_indices]
                    print(f"Adjusting class {label} sample count: {class_counts[label]} -> {len(class_indices[label])} (Target: {target_count})")
                elif target_count > len(class_indices[label]):
                    print(f"Class {label}: Requested sample count ({target_count}) is larger than original sample count ({len(class_indices[label])}). Keeping original.")
    
    # Limit maximum sample count (applied only if samples_per_class_dict is not provided)
    elif max_samples_per_class is not None:
        for label in class_indices:
            if len(class_indices[label]) > max_samples_per_class:
                selected_indices = torch.randperm(len(class_indices[label]))[:max_samples_per_class]
                class_indices[label] = class_indices[label][selected_indices]
                print(f"Limiting class {label} sample count: {class_counts[label]} -> {len(class_indices[label])}")
    
    # Combine selected indices
    selected_indices = torch.cat([indices for indices in class_indices.values()])
    selected_indices = torch.sort(selected_indices)[0]  # Sort
    
    # Print new class distribution
    new_class_counts = {}
    for label in unique_labels:
        new_class_counts[label.item()] = (data.y[selected_indices] == label).sum().item()
    print(f"Adjusted class distribution: {new_class_counts}")
    
    # Create subgraph
    return create_subgraph_from_indices(data, selected_indices)

def create_subgraph_from_indices(data, node_indices):
    """
    Function to create a subgraph with selected node indices
    """
    # Convert to node index set
    node_set = set(node_indices.tolist())
    
    # Edge filtering (keep edges only if both nodes are in the selected node set)
    edge_mask = []
    
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if src in node_set and dst in node_set:
            edge_mask.append(i)
            
    # Create new edge index and attributes
    if len(edge_mask) > 0:
        edge_mask = torch.tensor(edge_mask, dtype=torch.long)
        filtered_edge_index = data.edge_index[:, edge_mask]
        
        # Remap node indices
        node_idx_map = {idx.item(): i for i, idx in enumerate(node_indices)}
        remapped_edge_index = torch.zeros_like(filtered_edge_index)
        for i in range(filtered_edge_index.size(1)):
            remapped_edge_index[0, i] = node_idx_map[filtered_edge_index[0, i].item()]
            remapped_edge_index[1, i] = node_idx_map[filtered_edge_index[1, i].item()]
        
        # Filter edge attributes if they exist
        filtered_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    else:
        # Create empty tensor if no edges exist
        remapped_edge_index = torch.zeros((2, 0), dtype=torch.long)
        filtered_edge_attr = torch.zeros((0, data.edge_attr.size(1))) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    
    # Extract node features and labels
    node_x = data.x[node_indices]
    node_y = data.y[node_indices] if hasattr(data, 'y') and data.y is not None else None
    
    # Create new Data object
    subgraph = Data(
        x=node_x,
        edge_index=remapped_edge_index,
        edge_attr=filtered_edge_attr,
        y=node_y
    )
    
    return subgraph

def create_data_loaders(pkl_path, batch_size=32, train_ratio=0.7, val_ratio=0.15, 
                       use_cache=True, cache_dir='./data_convert/original_data/cache_data',
                       num_workers=4, balance_classes=False, max_samples_per_class=None, 
                       class_balance_ratio=1.0, samples_per_class_dict=None,
                       normalize_coords=True, coord_scale_factor=0.001):
    """
    Function to create data loaders
    
    Args:
        pkl_path: Pickle file path
        batch_size: Batch size
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        use_cache: Whether to use cache
        cache_dir: Cache directory
        num_workers: Number of workers
        balance_classes: Whether to balance classes
        max_samples_per_class: Maximum samples per class
        class_balance_ratio: Sample ratio for class 2
        normalize_coords: Whether to normalize coordinates
        coord_scale_factor: Coordinate scale factor
    """
    # Get cache file paths (including normalization parameters)
    train_cache, val_cache, test_cache = get_cache_path(
        pkl_path, train_ratio, val_ratio, cache_dir,
        balance_classes, max_samples_per_class, class_balance_ratio,
        samples_per_class_dict, normalize_coords, coord_scale_factor
    )
    
    # Load from cache if all cache files exist and use_cache is True
    if use_cache and os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
        print(f"Loading data from cache files...")
        print(f"  Cache file path: {train_cache}")
        if samples_per_class_dict is not None:
            print(f"  ⚠️  Warning: samples_per_class_dict is set but using previous cache.")
            print(f"  ⚠️  To apply new sampling settings, delete cache files or set use_cache=False.")
        
        start_time = time.time()
        
        # Load subgraphs from cache
        with open(train_cache, 'rb') as f:
            train_subgraph = pickle.load(f)
        with open(val_cache, 'rb') as f:
            val_subgraph = pickle.load(f)
        with open(test_cache, 'rb') as f:
            test_subgraph = pickle.load(f)
            
        print(f"Data loaded from cache (Time elapsed: {time.time() - start_time:.2f}s)")
    else:
        print(f"Loading from original data... (Will be saved to cache later)")
        start_time = time.time()
        
        # Load data from Pickle file
        pyg_data = load_pyg_data_from_pickle(pkl_path)
        
        # Apply coordinate normalization
        if normalize_coords:
            pyg_data = normalize_coordinates(pyg_data, normalize_coords, coord_scale_factor)
        
        # Apply class imbalance handling
        if balance_classes:
            print("Applying class imbalance handling to original data...")
            pyg_data = balance_class_samples(
                pyg_data, 
                balance_classes=balance_classes,
                max_samples_per_class=max_samples_per_class,
                class_balance_ratio=class_balance_ratio,
                samples_per_class_dict=samples_per_class_dict
            )
        
        # Generate data indices and split
        indices = list(range(pyg_data.num_nodes))
        
        # Split train, temp (train_ratio)
        train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
        
        # Split validation, test from temp
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio_adjusted, random_state=42)
        
        # Subgraph creation function
        def create_subgraph(data, node_indices):
            # Convert to node index set
            node_set = set(node_indices)
            
            # Edge filtering (keep edges only if both nodes are in the selected node set)
            edge_mask = []
            
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
                if src in node_set and dst in node_set:
                    edge_mask.append(i)
                    
            # Create new edge index and attributes
            if len(edge_mask) > 0:
                edge_mask = torch.tensor(edge_mask, dtype=torch.long)
                filtered_edge_index = data.edge_index[:, edge_mask]
                
                # Remap node indices
                node_idx_map = {idx: i for i, idx in enumerate(node_indices)}
                remapped_edge_index = torch.zeros_like(filtered_edge_index)
                for i in range(filtered_edge_index.size(1)):
                    remapped_edge_index[0, i] = node_idx_map[filtered_edge_index[0, i].item()]
                    remapped_edge_index[1, i] = node_idx_map[filtered_edge_index[1, i].item()]
                
                # Filter edge attributes if they exist
                filtered_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            else:
                # Create empty tensor if no edges exist
                remapped_edge_index = torch.zeros((2, 0), dtype=torch.long)
                filtered_edge_attr = torch.zeros((0, data.edge_attr.size(1))) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            
            # Extract node features and labels
            node_x = data.x[node_indices]
            node_y = data.y[node_indices] if hasattr(data, 'y') and data.y is not None else None
            
            # Create new Data object
            subgraph = Data(
                x=node_x,
                edge_index=remapped_edge_index,
                edge_attr=filtered_edge_attr,
                y=node_y
            )
            
            return subgraph
        
        # Create subgraphs for each split
        print("Creating subgraphs...")
        train_subgraph = create_subgraph(pyg_data, train_idx)
        val_subgraph = create_subgraph(pyg_data, val_idx)
        test_subgraph = create_subgraph(pyg_data, test_idx)
        
        # Save cache
        if use_cache:
            print("Saving data to cache...")
            os.makedirs(os.path.dirname(train_cache), exist_ok=True)
            with open(train_cache, 'wb') as f:
                pickle.dump(train_subgraph, f)
            with open(val_cache, 'wb') as f:
                pickle.dump(val_subgraph, f)
            with open(test_cache, 'wb') as f:
                pickle.dump(test_subgraph, f)
            print(f"Data saved to cache (Cache path: {cache_dir})")
            
        print(f"Data processing completed (Time elapsed: {time.time() - start_time:.2f}s)")
    
    # Print subgraph info
    print(f"Train Subgraph: {train_subgraph.num_nodes} nodes, {train_subgraph.num_edges} edges")
    print(f"Validation Subgraph: {val_subgraph.num_nodes} nodes, {val_subgraph.num_edges} edges")
    print(f"Test Subgraph: {test_subgraph.num_nodes} nodes, {test_subgraph.num_edges} edges")
    
    # Create GraphDataset
    train_dataset = GraphDataset([train_subgraph])
    val_dataset = GraphDataset([val_subgraph])
    test_dataset = GraphDataset([test_subgraph])
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# Example usage
# json_path = "/path/to/your/json_file.json"
# create_data_loaders(json_path, batch_size=32)