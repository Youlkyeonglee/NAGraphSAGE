import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, SAGEConv
from torch.nn import Linear
from models.convlayer import NeighborAwareSAGEConv, NeighborAware_SAGEConv_Attention_NodeLayer

class NeighborAwareGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=9, num_layers=3, aggr='mean', graph_data_type='image'):
        super(NeighborAwareGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer (pass aggr parameter)
        self.convs.append(NeighborAwareSAGEConv(in_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, graph_data_type=graph_data_type))
        
        # Middle layers (pass aggr parameter)
        for _ in range(num_layers - 2):
            self.convs.append(NeighborAwareSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, graph_data_type=graph_data_type))
        
        # Last layer (pass aggr parameter)
        if num_layers > 1:
            self.convs.append(NeighborAwareSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, graph_data_type=graph_data_type))
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        
        # Batch normalization
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Dropout
        self.dropout = 0.3
    
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Create subgraph (include only nodes in batch)
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # Save layer output
        layer_outputs = []
        
        # Pass through graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            if i < self.num_layers - 1:  # Apply activation function only before the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # Save feature vector before classification
        final_features = x
        
        # Classification
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        # Save layer output
        self.layer_outputs = layer_outputs
        
        if return_features:
            # Return feature vector and final output together
            return out, final_features
        return out
    
    def get_all_edge_weights(self):
        """Collects and returns edge weights of all layers.
        
        Returns:
            list: List of edge weights for each layer. Each element is a tensor of shape [num_edges, 1].
        """
        edge_weights_list = []
        for conv in self.convs:
            if hasattr(conv, 'get_edge_weights'):
                weights = conv.get_edge_weights()
                if weights is not None:
                    edge_weights_list.append(weights)
        return edge_weights_list

    
class NAGraphSAGE_Attention_NodeLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=9, num_layers=3, aggr='mean', 
                 attention_type='mlp', batch_size=100, scale=4, graph_data_type='image', type_node_layer='traditional'):
        super(NAGraphSAGE_Attention_NodeLayer, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer (pass aggr parameter)
        self.convs.append(NeighborAware_SAGEConv_Attention_NodeLayer(in_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, 
                                                                      attention_type=attention_type, batch_size=batch_size, scale=scale, 
                                                                      graph_data_type=graph_data_type, type_node_layer=type_node_layer))
        
        # Middle layers (pass aggr parameter)
        for _ in range(num_layers - 2):
            self.convs.append(NeighborAware_SAGEConv_Attention_NodeLayer(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, attention_type=attention_type, batch_size=batch_size, scale=scale, graph_data_type=graph_data_type, type_node_layer=type_node_layer))
        
        # Last layer (pass aggr parameter)
        if num_layers > 1:
            self.convs.append(NeighborAware_SAGEConv_Attention_NodeLayer(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, attention_type=attention_type, batch_size=batch_size, scale=scale, graph_data_type=graph_data_type, type_node_layer=type_node_layer))
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        
        # Batch normalization
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Dropout
        self.dropout = 0.3
    
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Create subgraph (include only nodes in batch)
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # Save layer output
        layer_outputs = []
        
        # Pass through graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            if i < self.num_layers - 1:  # Apply activation function only before the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # Save feature vector before classification
        final_features = x
        
        # Classification
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        # Save layer output
        self.layer_outputs = layer_outputs
        
        if return_features:
            # Return feature vector and final output together
            return out, final_features
        return out