import torch
from torch import Tensor
import warnings
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch.nn as nn
        
# 4. Concrete application example: Assigning different weights to distance and speed
# As a more concrete example, I will show you how to assign different weights to edge attributes such as distance, speed, and bounding box position:
class NeighborAwareSAGEConv(MessagePassing):
    """
    GraphSAGE layer that applies different weights for each type of edge attribute
    """
    def __init__(self, in_channels, out_channels, edge_dim=9, aggr='mean', graph_data_type='image'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.graph_data_type = graph_data_type
        # Linear layer for message transformation
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
        # Root node (self) weight
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        
        # Weights by edge attribute category (e.g., distance, speed, direction, acceleration, position/size)
        if self.graph_data_type == 'image':
            self.distance_proj = torch.nn.Linear(1, 8)  # Distance
            self.speed_proj = torch.nn.Linear(1, 8)     # Speed
            self.dir_proj = torch.nn.Linear(3, 8)       # Direction
            self.accel_proj = torch.nn.Linear(1, 8)     # Acceleration
            self.bbox_proj = torch.nn.Linear(4, 8)      # Bounding box
        elif self.graph_data_type == 'world':
            self.distance_proj = torch.nn.Linear(1, 8)  # Distance
            self.speed_proj = torch.nn.Linear(1, 8)     # Speed
            self.dir_proj = torch.nn.Linear(3, 8)       # Direction
            self.accel_proj = torch.nn.Linear(1, 8)     # Acceleration
            self.position_proj = torch.nn.Linear(3, 8)      # Position
        
        # Calculate final weights
        
        # Calculate input_size based on edge_dim
        if self.graph_data_type == 'image':
            if edge_dim == 5:  # bbox(4), speed(1)
                input_size = 16  # 8 * 2 (8 dimensions per feature)
            elif edge_dim == 6:  # bbox(4), speed(1), acceleration(1)
                input_size = 24  # 8 * 3 (bbox, speed, acceleration)
            elif edge_dim == 7:  # bbox(4), speed(1), acceleration(1), distance(1)
                input_size = 32  # 8 * 4 (bbox, speed, acceleration, distance)
            elif edge_dim == 10:  # bbox(4), speed(1), direction(3), acceleration(1), distance(1)
                input_size = 40  # 8 * 5 (bbox, speed, acceleration, distance, direction)
            else:
                raise ValueError(f"Unsupported edge_dim: {edge_dim}")
        elif self.graph_data_type == 'world':
            if edge_dim == 4:  # position(3), speed(1)
                input_size = 16  # 8 * 2 (8 dimensions per feature)
            elif edge_dim == 5:  # position(3), speed(1)
                input_size = 24  # 8 * 3 (bbox, speed, acceleration)
            elif edge_dim == 6:  # position(3), speed(1), acceleration(1)
                input_size = 32  # 8 * 4 (bbox, speed, acceleration, distance)
            elif edge_dim == 9:  # position(3), speed(1), direction(3), acceleration(1), distance(1)
                input_size = 40  # 8 * 5 (bbox, speed, acceleration, distance, direction)
            else:
                raise ValueError(f"Unsupported edge_dim: {edge_dim}")
        self.weight_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),  # Sum of values projected to 8 dimensions per feature
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_self.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None):
        # Transformation for self
        x_self = self.lin_self(x)
        # edge_index debugging
        # print("--------------------------------")
        # print("edge_index shape:", edge_index.shape)
        # # Calculate occurrence count per target node
        # unique_targets, target_counts = torch.unique(edge_index[1], return_counts=True)
        # print("Occurrence count per target node:")
        # for target, count in zip(unique_targets, target_counts):  # Print only first 5
        #     if count == 5:
        #         print(f"Node {target}: {count} occurrences")
        # print("...")  # Omit the rest
        # print("Number of unique target nodes:", edge_index[1].unique().size(0))       # Number of unique target nodes
        # print("Total number of nodes:", x.size(0))                                   # Total number of nodes
        # print("Total number of edges:", edge_index.size(1))                          # Total number of edges
        # print("--------------------------------")

        # print("--------------------------------")
        # print("x_self", x_self)
        # print("x_self.shape", x_self.shape) #449092,128: train loader data size / 96233,128: val, test loader data size
        # print("--------------------------------")
        
        # Message propagation from neighbor nodes
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # print("--------------------------------")
        # print("out", out)
        # print("out.shape", out.shape) #[449092,128]: train loader data size / [96,233,128] val, test loader data size
        # print("--------------------------------")
        # Combine
        return x_self + out
    
    def message(self, x_j, edge_attr):
        # Check if edge attributes are valid and process each feature
        edge_features = []
        
        if edge_attr is not None:
            if self.graph_data_type == 'image':
                if self.edge_dim == 5:  # bbox(4), speed(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)

                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    
                elif self.edge_dim == 6:  # bbox(4), speed(1), acceleration(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 7:  # bbox(4), speed(1), acceleration(1), distance(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. Distance (1 dimension)
                    distance = edge_attr[:, 6:7]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 10:  # bbox(4), speed(1), direction(3), acceleration(1), distance(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Direction (3 dimensions)
                    direction = edge_attr[:, 5:8]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)

                    # 4. Acceleration (1 dimension)
                    accel = edge_attr[:, 8:9]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. Distance (1 dimension)
                    distance = edge_attr[:, 9:10]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
            elif self.graph_data_type == 'world':
                if self.edge_dim == 4:  # position(3), speed(1)
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                elif self.edge_dim == 5:  # position(3), speed(1), acceleration(1)
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                elif self.edge_dim == 6:  # position(3), speed(1), acceleration(1), distance(1) 
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    # 4. Distance (1 dimension)
                    distance = edge_attr[:, 5:6]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                elif self.edge_dim == 9:  # position(3), speed(1), direction(3), acceleration(1), distance(1)
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. Direction (3 dimensions)
                    direction = edge_attr[:, 4:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)
                    # 4. Acceleration (1 dimension)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    # 5. Distance (1 dimension)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
        # Combine all transformed features
        # print("--------------------------------")
        # print("edge_features", edge_features)
        # print("--------------------------------")
        if edge_features:
            h_combined = torch.cat(edge_features, dim=1)
            
            # Calculate final weights
            weight = self.weight_net(h_combined)
            
            # Save weights (for visualization or analysis)
            self.last_attention_weights = weight
            
            # Basic node transformation
            x_j = self.lin(x_j)
            
            return x_j * weight
        else:
            # Perform only basic node transformation if no edge attributes
            self.last_edge_weights = None
            return self.lin(x_j)
    
    def get_attention_weights(self):
        return self.last_attention_weights

class NeighborAware_SAGEConv_Attention_NodeLayer(MessagePassing):
    """
    GraphSAGE layer that applies different weights for each type of edge attribute
    """
    def __init__(self, in_channels, out_channels, edge_dim=9, aggr='mean', attention_type='mlp', 
                 batch_size=10000, scale=4, graph_data_type='image', type_node_layer='traditional'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.graph_data_type = graph_data_type
        self.type_node_layer = type_node_layer
        self.attention_type = attention_type  # 'mlp', 'inner_product', 'cosine'
        self.batch_size = batch_size  # Batch processing size
        self.scale = scale  # Scaling factor
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        # Linear layer for message transformation
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
        # Root node (self) weight
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        
        # Weights by edge attribute category (e.g., distance, speed, direction, acceleration, position/size)
        if self.graph_data_type == 'image':
            self.distance_proj = torch.nn.Linear(1, 8)  # Distance
            self.speed_proj = torch.nn.Linear(1, 8)     # Speed
            self.dir_proj = torch.nn.Linear(3, 8)       # Direction
            self.accel_proj = torch.nn.Linear(1, 8)     # Acceleration
            self.bbox_proj = torch.nn.Linear(4, 8)      # Bounding box
        elif self.graph_data_type == 'world':
            self.distance_proj = torch.nn.Linear(1, 8)  # Distance
            self.speed_proj = torch.nn.Linear(1, 8)     # Speed
            self.dir_proj = torch.nn.Linear(3, 8)       # Direction
            self.accel_proj = torch.nn.Linear(1, 8)     # Acceleration
            self.position_proj = torch.nn.Linear(3, 8)      # Position
        
        # Calculate final weights
        # Calculate input_size based on edge_dim
        if self.graph_data_type == 'image':
            if self.edge_dim == 5:  # bbox(4), speed(1)
                input_size = 16  # 8 * 2 (8 dimensions per feature)
            elif self.edge_dim == 6:  # bbox(4), speed(1), acceleration(1)
                input_size = 24  # 8 * 3 (bbox, speed, acceleration)
            elif self.edge_dim == 7:  # bbox(4), speed(1), acceleration(1), distance(1)
                input_size = 32  # 8 * 4 (bbox, speed, acceleration, distance)
            elif self.edge_dim == 10:  # bbox(4), speed(1), direction(3), acceleration(1), distance(1)
                input_size = 40  # 8 * 5 (bbox, speed, acceleration, distance, direction)
            else:
                raise ValueError(f"Unsupported edge_dim: {self.edge_dim}")
        elif self.graph_data_type == 'world':
            if self.edge_dim == 4:  # position(3), speed(1)
                input_size = 16  # 8 * 2 (8 dimensions per feature)
            elif self.edge_dim == 5:  # position(3), speed(1), acceleration(1)
                input_size = 24  # 8 * 3 (bbox, speed, acceleration)
            elif self.edge_dim == 6:  # position(3), speed(1), acceleration(1), distance(1)
                input_size = 32  # 8 * 4 (bbox, speed, acceleration, distance)
            elif self.edge_dim == 9:  # position(3), speed(1), direction(3), acceleration(1), distance(1)
                input_size = 40  # 8 * 5 (bbox, speed, acceleration, distance, direction)
            else:
                raise ValueError(f"Unsupported edge_dim: {self.edge_dim}")

        self._last_attention_weights = None  # For saving attention weights
        
        # Initialize layers based on attention type
        if attention_type == 'mlp':
            self.attention_layer = torch.nn.Sequential(
                torch.nn.Linear(input_size, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid()
            )
        
        # Common value vector projection (v)
        self.value_proj = torch.nn.Linear(input_size, 16)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_self.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None):


        if self.type_node_layer == 'traditional':
            x_self = self.lin_self(x)
        else:
            raise ValueError(f"Unsupported type_node_layer: {self.type_node_layer}")
        
        # Message propagation from neighbor nodes
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Combine
        return x_self + out
    
    def message(self, x_j, edge_attr):
        # Check if edge attributes are valid and process each feature
        edge_features = []
        
        # Process edge features
        if edge_attr is not None and edge_attr.size(1) > 0:  # Check if edge_attr is not empty
            if self.graph_data_type == 'image':
                if self.edge_dim == 5:  # bbox(4), speed(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)

                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    
                elif self.edge_dim == 6:  # bbox(4), speed(1), acceleration(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 7:  # bbox(4), speed(1), acceleration(1), distance(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. Distance (1 dimension)
                    distance = edge_attr[:, 6:7]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 9:  # bbox(4), speed(1), direction(2), acceleration(1), distance(1)
                    # 1. bbox (4 dimensions)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Direction (2 dimensions)
                    direction = edge_attr[:, 5:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)

                    # 4. Acceleration (1 dimension)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. Distance (1 dimension)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
            elif self.graph_data_type == 'world':
                if self.edge_dim == 4:  # position(3), speed(1)
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                elif self.edge_dim == 5:  # position(3), speed(1), acceleration(1)
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 6:  # position(3), 속도(1), 가속도(1), 거리(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Acceleration (1 dimension)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. Distance (1 dimension)
                    distance = edge_attr[:, 5:6]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 9:  # position(3), speed(1), direction(3), acceleration(1), distance(1)
                    # 1. position (3 dimensions)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. Speed (1 dimension)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. Direction (3 dimensions)
                    direction = edge_attr[:, 4:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)
                    
                    # 4. Acceleration (1 dimension)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. Distance (1 dimension)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                
        
        # Combine all transformed features
        if edge_features:
            h_combined = torch.cat(edge_features, dim=1)
            
            x_j = self.lin(x_j)
            
            # Apply different weight calculation methods based on attention type
            if self.attention_type == 'mlp':
                attention_weights = self.attention_layer(h_combined)
            
            elif self.attention_type == 'inner_product':
                # Preparation for batch processing
                num_edges = h_combined.size(0)
                
                query = self.inner_product_query(h_combined) # [num_edges, hidden_size]
                key = self.inner_product_key(h_combined) # [num_edges, hidden_size]

                # Reduce memory usage with batch processing
                scores_chunks = []
                for start in range(0, num_edges, self.batch_size):
                    end = min(start + self.batch_size, num_edges)
                    
                    # Use element-wise multiplication and sum instead of bmm (memory efficient)
                    batch_q = query[start:end]
                    batch_k = key[start:end]
                    batch_scores = torch.sum(batch_q * batch_k, dim=1) / self.scale
                    scores_chunks.append(batch_scores)

                # Combine all batch results
                scores = torch.cat(scores_chunks, dim=0)
                attention_weights = F.softmax(scores, dim=0).unsqueeze(-1)
            
            elif self.attention_type == 'cosine':
                # 3. Cosine similarity based attention
                query = self.cosine_key(h_combined)
                key = self.cosine_query(h_combined)
                
                # Calculate cosine similarity
                scores_chunks = []
                num_edges = h_combined.size(0)
                for start in range(0, num_edges, self.batch_size):
                    end = min(start + self.batch_size, num_edges)
                    
                    # Use element-wise multiplication and sum instead of bmm (memory efficient)
                    batch_q = query[start:end]
                    batch_k = key[start:end]
                    batch_scores = self.cosine(batch_q, batch_k)
                    # print("batch_scores.shape: ", batch_scores.shape)
                    scores_chunks.append(batch_scores)
                # print("scores_chunks[0].shape: ", scores_chunks[0].shape)
                attention_weights = torch.cat(scores_chunks, dim=0).unsqueeze(-1)
            
            
            # Save weights (for visualization or analysis)
            self.last_attention_weights = attention_weights
            
            # Apply attention weights to transformed node features
            return x_j * attention_weights
        else:
            # Perform only basic node transformation if no edge attributes
            return self.lin(x_j)
        
        
    def get_attention_weights(self):
        return self._last_attention_weights