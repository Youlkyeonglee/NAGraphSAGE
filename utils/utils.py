import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2
import time
import numpy as np
import traceback
import csv
import json
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from torch_geometric.data import Data
from shapely.geometry import Polygon
from datetime import datetime
import collections

# Function to calculate parameter count and model size
def print_model_summary(model):
    total_params = 0
    total_size_bytes = 0
    byte_per_param = 4  # float32 is 4 bytes

    print("Model Parameter Summary:")
    print("-" * 80)
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_size_bytes = param_count * byte_per_param
        total_params += param_count
        total_size_bytes += param_size_bytes
        print(f"Layer: {name:<40} | Parameters: {param_count:<8} | Size: {param_size_bytes:<6} bytes | Shape: {param.size()}")

    # Print total count and size
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024
    print("-" * 80)
    print(f"Total Parameters: {total_params}")
    print(f"Total Model Size: {total_size_bytes} bytes ({total_size_kb:.2f} KB, {total_size_mb:.4f} MB)")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Signal to stop training
        return False
    
# Add loss function definitions
class FocalLoss(nn.Module):
    """
    Focal Loss implementation effective for class imbalance problems
    (alpha for class weights, gamma for focusing on hard samples)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss implementation for preventing overfitting
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        
        # Create one-hot encoding and apply smoothing
        one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        smoothed_targets = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(smoothed_targets * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    """
    Dice Loss implementation effective for imbalanced classes
    """
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply softmax
        inputs = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding
        n_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Product of predictions and targets
        intersection = (inputs * targets_one_hot).sum(dim=0)
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=0) + targets_one_hot.sum(dim=0) + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class CentroidSeparationLoss(nn.Module):
    """
    Centroid Separation Loss function to improve separation in inter-class t-SNE visualization
    
    Calculates the centroid of each class in the feature space, 
    and operates by maximizing the distance between class centroids and minimizing the sample variance within classes.
    
    This loss function is particularly effective for separating classes that overlap in the feature space, such as class_1 and class_2.
    """
    def __init__(self, num_classes, feat_dim, lambda_inter=1.0, lambda_intra=1.0, margin=2.0):
        super(CentroidSeparationLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_inter = lambda_inter  # Inter-class distance weight
        self.lambda_intra = lambda_intra  # Intra-class distance weight
        self.margin = margin  # Minimum inter-class distance
        
        # Initialize class centroids as learnable parameters
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features, targets, embeddings=None):
        """
        Args:
            features: Feature vector just before the last layer of the model (batch size x feature dimension)
            targets: Target class labels (batch size)
            embeddings: Additional embedding vectors (optional)
        """
        if embeddings is not None:
            features = embeddings
        
        batch_size = features.size(0)
        
        # Calculate centroids for each class
        centers = torch.zeros(self.num_classes, self.feat_dim).to(features.device)
        counts = torch.zeros(self.num_classes).to(features.device)
        
        for i in range(batch_size):
            c = targets[i].item()
            counts[c] += 1
            centers[c] += features[i]
        
        # Prevent division by zero
        counts = torch.clamp(counts, min=1)
        centers = centers / counts.view(-1, 1)
        
        # Intra-class loss: samples of the same class should be close to the centroid
        intra_loss = 0
        for i in range(batch_size):
            c = targets[i].item()
            dist = torch.sum((features[i] - centers[c]) ** 2)
            intra_loss += dist
        intra_loss = intra_loss / batch_size
        
        # Inter-class loss: centroids of different classes should be far apart
        inter_loss = 0
        n_pairs = 0
        
        # Specifically focus on separation between class_1 and class_2
        special_weight = 2.0  # Increase weight between class_1 and class_2
        
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                # Apply special weight to class_1 and class_2
                weight = special_weight if (i == 1 and j == 2) or (i == 2 and j == 1) else 1.0
                
                dist = torch.sum((centers[i] - centers[j]) ** 2)
                # Apply penalty if distance is less than margin
                inter_loss += weight * torch.clamp(self.margin - dist, min=0)
                n_pairs += 1
        
        if n_pairs > 0:
            inter_loss = inter_loss / n_pairs
        
        # Final loss: minimize intra-class distance, maximize inter-class distance
        loss = self.lambda_intra * intra_loss + self.lambda_inter * inter_loss
        
        return loss

class ContrastiveCenterLoss(nn.Module):
    """
    Contrastive Center Loss
    
    Learns to place samples of the same class close together and samples of different classes far apart in the feature space.
    Designed specifically to address the issue of class_1 and class_2 overlapping in t-SNE visualization.
    """
    def __init__(self, num_classes, feat_dim, temperature=0.07, lambda_val=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temperature = temperature  # Temperature parameter for contrastive learning
        self.lambda_val = lambda_val  # Loss weight
        
        # Initialize class prototype vectors (learnable)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, targets):
        """
        Args:
            features: Feature vectors (batch size x feature dimension)
            targets: Target class labels (batch size)
        """
        batch_size = features.size(0)
        
        # Normalize features (for cosine similarity calculation)
        features = F.normalize(features, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)
        
        # Calculate inter-class similarity matrix
        sim_matrix = torch.matmul(features, centers.T) / self.temperature
        
        # Calculate loss for each sample
        losses = []
        for i in range(batch_size):
            pos_idx = targets[i].item()  # Actual class of current sample
            
            # Similarity with positive class
            pos_sim = sim_matrix[i, pos_idx]
            
            # Similarity with all classes (in log-sum-exp form)
            neg_sim = torch.logsumexp(sim_matrix[i], dim=0)
            
            # InfoNCE loss
            curr_loss = -pos_sim + neg_sim
            
            # Apply additional weight for class_1 and class_2
            if pos_idx == 1 or pos_idx == 2:
                curr_loss = curr_loss * 2.0  # Apply more weight
                
            losses.append(curr_loss)
        
        # Calculate mean loss
        loss = torch.stack(losses).mean() * self.lambda_val
        
        return loss

# Loss function selection function
def get_loss_function(loss_name, class_weights=None, num_classes=None, **kwargs):
    """
    Returns a loss function object based on the loss function name
    
    Args:
        loss_name (str): Loss function name ('ce', 'focal', 'smoothing', 'dice', 'centroid', 'contrastive')
        class_weights (torch.Tensor): Class weights
        num_classes (int): Number of classes
        **kwargs: Additional parameters (focal_gamma, smoothing_alpha, feat_dim, etc.)
    
    Returns:
        nn.Module: Selected loss function
    """
    if loss_name.lower() == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name.lower() == 'weighted_ce':
        if class_weights is None and num_classes is not None:
            print("Warning: No class weights provided, using standard CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name.lower() == 'focal':
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_name.lower() == 'smoothing':
        smoothing = kwargs.get('smoothing_alpha', 0.1)
        return LabelSmoothingLoss(smoothing=smoothing)
    
    elif loss_name.lower() == 'dice':
        return DiceLoss()
    
    elif loss_name.lower() == 'centroid':
        if num_classes is None:
            raise ValueError("num_classes must be provided for CentroidSeparationLoss")
        feat_dim = kwargs.get('feat_dim', 64)
        lambda_inter = kwargs.get('lambda_inter', 1.0)
        lambda_intra = kwargs.get('lambda_intra', 1.0)
        margin = kwargs.get('margin', 2.0)
        return CentroidSeparationLoss(
            num_classes=num_classes, 
            feat_dim=feat_dim,
            lambda_inter=lambda_inter,
            lambda_intra=lambda_intra,
            margin=margin
        )
    
    elif loss_name.lower() == 'contrastive':
        if num_classes is None:
            raise ValueError("num_classes must be provided for ContrastiveCenterLoss")
        feat_dim = kwargs.get('feat_dim', 64)
        temperature = kwargs.get('temperature', 0.07)
        lambda_val = kwargs.get('lambda_val', 1.0)
        return ContrastiveCenterLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            temperature=temperature,
            lambda_val=lambda_val
        )
    
    else:
        print(f"Unknown loss function: {loss_name}, using CrossEntropyLoss as default")
        return nn.CrossEntropyLoss(weight=class_weights)

# Class weight calculation function
def calculate_class_weights(loader, num_classes, device):
    """
    Function to calculate class weights from data loader
    
    Args:
        loader: Data loader
        num_classes: Number of classes
        device: Device to use for calculation
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = torch.zeros(num_classes)
    
    # Calculate number of samples per class
    for batch in loader:
        for cls in range(num_classes):
            class_counts[cls] += (batch.y == cls).sum().item()
    
    # Calculate weights (higher weights for classes with fewer samples)
    weights = 1.0 / (class_counts + 1e-8)  # Prevent zero division
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return weights.to(device)


class VehicleDetectorWithGNN:
    def __init__(self, model_path, gnn_model_path=None, conf_thresh=0.25, iou_thresh=0.45, device=None, lane_json_path=None,
                 model_type='GraphSAGE', hidden_channels=128, num_layers=5, edge_dim=9, num_classes=3, 
                 node_features=8, aggr='mean', attention_type='mlp', num_relations=7, graph_data_type='image'):
        # Device setup
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        print(f"YOLO Model loaded from {model_path}")
        
        # Save GNN model settings
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_features = node_features
        self.aggr = aggr
        self.attention_type = attention_type
        self.num_relations = num_relations
        self.graph_data_type = graph_data_type  # Save graph_data_type
        
        # Initialize settings
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.current_frame = 0
        self.vehicle_history = {}
        self.history_length = 10
        
        # Set motion state names and colors
        self.motion_state_names = {
            0: "Stop",
            1: "Lane Change",
            2: "Normal Driving"
        }
        
        self.motion_colors = {
            0: (0, 0, 255),    # Stop: Red
            1: (0, 255, 255),  # Lane Change: Yellow
            2: (0, 255, 0)     # Normal Driving: Green
        }
        
        # Load GNN model
        self.gnn_model = None
        if gnn_model_path and os.path.exists(gnn_model_path):
            try:
                # Create model instance according to model type
                from models.model import GAT, GraphSAGE, AttributeAwareGraphSAGE, AttributeAwareRGCN_SAGE
                
                if model_type == 'GraphSAGE':
                    self.gnn_model = GraphSAGE(
                        in_channels=node_features,
                        hidden_channels=hidden_channels,
                        out_channels=num_classes,
                        num_layers=num_layers,
                        aggr=aggr
                    )
                elif model_type == 'GAT':
                    self.gnn_model = GAT(
                        in_channels=node_features,
                        hidden_channels=hidden_channels,
                        out_channels=num_classes,
                        num_layers=num_layers,
                        dropout=0.5,
                        edge_dim=edge_dim
                    )
                elif model_type == 'AttributeAwareGraphSAGE':
                    self.gnn_model = AttributeAwareGraphSAGE(
                        in_channels=node_features,
                        hidden_channels=hidden_channels,
                        out_channels=num_classes,
                        edge_dim=edge_dim,
                        num_layers=num_layers,
                        aggr=aggr,
                        graph_data_type=graph_data_type
                    )
                elif model_type == 'NAGraphSAGE_Traditional':
                    # Import NAGraphSAGE_Traditional optionally
                    try:
                        from models.model import NAGraphSAGE_Traditional
                        self.gnn_model = NAGraphSAGE_Traditional(
                            in_channels=node_features,
                            hidden_channels=hidden_channels,
                            out_channels=num_classes,
                            edge_dim=edge_dim,
                            num_layers=num_layers,
                            aggr=aggr,
                            attention_type=attention_type,
                            batch_size=10000,
                            scale=4,
                            top_k=None,
                            edge_sampling_ratio=1.0
                        )
                    except ImportError:
                        print(f"Warning: NAGraphSAGE_Traditional not found, skipping...")
                        self.gnn_model = None
                elif model_type == 'AttributeAwareRGCN_SAGE':
                    self.gnn_model = AttributeAwareRGCN_SAGE(
                        in_channels=node_features,
                        hidden_channels=hidden_channels,
                        out_channels=num_classes,
                        num_relations=num_relations,
                        batch_size=10000,
                        scale=4,
                        edge_dim=edge_dim,
                        num_layers=num_layers,
                        aggr=aggr,
                        attention_type=attention_type
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Move model to device
                if self.gnn_model is not None:
                    self.gnn_model = self.gnn_model.to(self.device)
                    
                    # Load saved weights
                    state_dict = torch.load(gnn_model_path, map_location=self.device)
                    
                    # If state_dict is OrderedDict, consider it as model state_dict and load
                    if isinstance(state_dict, collections.OrderedDict) or isinstance(state_dict, dict):
                        self.gnn_model.load_state_dict(state_dict)
                    # If it is a full model, use state_dict attribute
                    elif hasattr(state_dict, 'state_dict'):
                        self.gnn_model.load_state_dict(state_dict.state_dict())
                    # If saved in other format (e.g., checkpoint)
                    else:
                        if 'model' in state_dict:
                            self.gnn_model.load_state_dict(state_dict['model'])
                        elif 'state_dict' in state_dict:
                            self.gnn_model.load_state_dict(state_dict['state_dict'])
                        else:
                            raise ValueError("Unknown model format.")
                    
                    print(f"GNN model loaded successfully: {gnn_model_path}")
                    print(f"Model type: {model_type}")
                    self.gnn_model.eval()  # Set evaluation mode
                
            except Exception as e:
                print(f"Failed to load GNN model: {e}")
                traceback.print_exc()
                self.gnn_model = None
        
        # Initialize lane data related variables
        self.lane_areas = {}
        self.lane_centerlines = {}
        
        # Load lane data (if provided)
        if lane_json_path and os.path.exists(lane_json_path):
            self.load_lane_data(lane_json_path)
    
    def load_lane_data(self, lane_json_path):
        """Load lane data"""
        try:
            print(f"Attempting to load lane data: {lane_json_path}")
            with open(lane_json_path, 'r') as f:
                lane_data = json.load(f)
            
            # Extract lane areas and centerlines
            lane_areas = {}
            lane_centerlines = {}
            
            for lane in lane_data.get('lanes', []):
                lane_id = lane.get('id')
                points = lane.get('points', [])
                
                if points and len(points) >= 4:
                    # Save lane area as Polygon
                    polygon = Polygon(points)
                    lane_areas[lane_id] = polygon
                    
                    # Calculate lane centerline (midpoints of first and last points)
                    n = len(points)
                    half_n = n // 2
                    centerline = []
                    for i in range(half_n):
                        p1 = points[i]
                        p2 = points[n-i-1]
                        mid_x = (p1[0] + p2[0]) / 2
                        mid_y = (p1[1] + p2[1]) / 2
                        centerline.append([mid_x, mid_y])
                    
                    lane_centerlines[lane_id] = centerline
            
            self.lane_areas = lane_areas
            self.lane_centerlines = lane_centerlines
            print(f"Lane data loaded: {len(lane_areas)} lanes, {len(lane_centerlines)} centerlines")
        except Exception as e:
            print(f"Failed to load lane data: {e}")
            traceback.print_exc()
    
    def detect(self, frame):
        """
        Detect vehicles in frame
        """
        try:
            results = self.model.track(frame, persist=True, conf=self.conf, iou=self.iou, tracker="bytetrack.yaml")
            return results
        except Exception as e:
            print(f"Error during vehicle detection: {e}")
            traceback.print_exc()
            return None
    
    def create_vehicle_graph(self, detections, width, height):
        """
        Convert vehicle data to graph format
        """
        if len(detections) == 0:
            return None
        
        try:
            features = []
            edge_index = []
            edge_attr = []
            
            # Distance threshold for connecting nearby vehicles
            distance_threshold = 200  # In pixels
            
            # Create feature vector for each vehicle
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id, track_id = det
                
                # Calculate center point and size
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width_box = x2 - x1
                height_box = y2 - y1
                
                # Convert to normalized coordinates
                norm_x = center_x / width
                norm_y = center_y / height
                norm_width = width_box / width
                norm_height = height_box / height
                
                # Calculate speed and direction
                speed = 0.0
                direction_x, direction_y = 0.0, 0.0
                acceleration = 0.0
                
                if track_id in self.vehicle_history and len(self.vehicle_history[track_id]) > 0:
                    prev_info = self.vehicle_history[track_id][-1]
                    prev_x = prev_info.get('center', [0, 0])[0]
                    prev_y = prev_info.get('center', [0, 0])[1]
                    
                    dx = center_x - prev_x
                    dy = center_y - prev_y
                    displacement = math.sqrt(dx*dx + dy*dy)
                    
                    # Calculate speed and direction
                    speed = displacement
                    if displacement > 0:
                        direction_x = dx / displacement
                        direction_y = dy / displacement
                    
                    # Calculate acceleration
                    prev_speed = prev_info.get('speed', 0.0)
                    acceleration = speed - prev_speed
                
                # Feature vector: [x, y, w, h, speed, dir_x, dir_y, acceleration]
                feature = [
                    norm_x, norm_y, norm_width, norm_height,
                    speed, direction_x, direction_y, acceleration
                ]
                
                features.append(feature)
            
            # Create edges (connect nearby vehicles)
            for i in range(len(detections)):
                x1i, y1i, x2i, y2i, _, _, _ = detections[i]
                center_i = [(x1i + x2i) / 2, (y1i + y2i) / 2]
                
                for j in range(len(detections)):
                    if i == j:
                        continue
                    
                    x1j, y1j, x2j, y2j, _, _, _ = detections[j]
                    center_j = [(x1j + x2j) / 2, (y1j + y2j) / 2]
                    
                    # Calculate distance between two vehicles
                    distance = math.sqrt(
                        (center_i[0] - center_j[0])**2 + 
                        (center_i[1] - center_j[1])**2
                    )
                    
                    # Create edge if distance is less than threshold
                    if distance < distance_threshold:
                        edge_index.append([i, j])
                        edge_attr.append([distance / distance_threshold])  # Normalized distance
            
            # Create PyTorch Geometric data object
            if features and edge_index:
                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                return data
            else:
                # If there are no edges (graph with only nodes)
                if features:
                    x = torch.tensor(features, dtype=torch.float32).to(self.device)
                    data = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long).to(self.device), 
                               edge_attr=torch.zeros((0, 1), dtype=torch.float).to(self.device))
                    return data
        except Exception as e:
            print(f"Error creating graph: {e}")
            traceback.print_exc()
        
        return None
    
    def _serialize_for_json(self, data):
        """Convert NumPy types to JSON serializable Python native types"""
        if isinstance(data, dict):
            return {key: self._serialize_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        else:
            return data