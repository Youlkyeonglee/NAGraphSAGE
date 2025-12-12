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

# 파라미터 수 및 용량 계산 함수
def print_model_summary(model):
    total_params = 0
    total_size_bytes = 0
    byte_per_param = 4  # float32는 4바이트

    print("Model Parameter Summary:")
    print("-" * 80)
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_size_bytes = param_count * byte_per_param
        total_params += param_count
        total_size_bytes += param_size_bytes
        print(f"Layer: {name:<40} | Parameters: {param_count:<8} | Size: {param_size_bytes:<6} bytes | Shape: {param.size()}")

    # 총합 및 용량 출력
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
                return True  # 학습 중단 신호
        return False
    
# 손실 함수 정의 부분 추가
class FocalLoss(nn.Module):
    """
    클래스 불균형 문제에 효과적인 Focal Loss 구현
    (alpha로 클래스 가중치, gamma로 어려운 샘플에 집중)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스 가중치
        self.gamma = gamma  # 집중도 파라미터
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
    과적합 방지를 위한 Label Smoothing Loss 구현
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        
        # 원-핫 인코딩 생성 후 스무딩 적용
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
    불균형 클래스에 효과적인 Dice Loss 구현
    """
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 소프트맥스 적용
        inputs = F.softmax(inputs, dim=1)
        
        # 원-핫 인코딩 생성
        n_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # 예측값과 타겟 값의 곱
        intersection = (inputs * targets_one_hot).sum(dim=0)
        
        # Dice 계수 계산
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=0) + targets_one_hot.sum(dim=0) + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class CentroidSeparationLoss(nn.Module):
    """
    클래스 간 t-SNE 시각화에서 분리를 향상시키기 위한 중심점 분리 손실 함수
    
    특징 공간에서 각 클래스의 중심점(centroid)을 계산하고, 
    클래스 간의 중심점 거리는 최대화하고 클래스 내의 샘플 분산은 최소화하는 방식으로 동작합니다.
    
    이 손실 함수는 특히 class_1과 class_2처럼 특징 공간에서 겹치는 클래스들을 분리하는데 효과적입니다.
    """
    def __init__(self, num_classes, feat_dim, lambda_inter=1.0, lambda_intra=1.0, margin=2.0):
        super(CentroidSeparationLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_inter = lambda_inter  # 클래스 간 거리 가중치
        self.lambda_intra = lambda_intra  # 클래스 내 거리 가중치
        self.margin = margin  # 최소 클래스 간 거리
        
        # 클래스 중심점을 학습 가능한 파라미터로 초기화
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features, targets, embeddings=None):
        """
        Args:
            features: 모델의 마지막 레이어 직전의 특징 벡터 (배치 크기 x 특징 차원)
            targets: 타겟 클래스 레이블 (배치 크기)
            embeddings: 추가적인 임베딩 벡터 (선택적)
        """
        if embeddings is not None:
            features = embeddings
        
        batch_size = features.size(0)
        
        # 각 클래스별 중심점 계산
        centers = torch.zeros(self.num_classes, self.feat_dim).to(features.device)
        counts = torch.zeros(self.num_classes).to(features.device)
        
        for i in range(batch_size):
            c = targets[i].item()
            counts[c] += 1
            centers[c] += features[i]
        
        # 0으로 나누는 것 방지
        counts = torch.clamp(counts, min=1)
        centers = centers / counts.view(-1, 1)
        
        # 클래스 내 손실 (intra-class loss): 같은 클래스 샘플들이 중심점에 가깝게
        intra_loss = 0
        for i in range(batch_size):
            c = targets[i].item()
            dist = torch.sum((features[i] - centers[c]) ** 2)
            intra_loss += dist
        intra_loss = intra_loss / batch_size
        
        # 클래스 간 손실 (inter-class loss): 다른 클래스 중심점들이 서로 멀게
        inter_loss = 0
        n_pairs = 0
        
        # 특별히 class_1과 class_2 간의 분리에 더 집중
        special_weight = 2.0  # class_1과 class_2 사이의 가중치를 높임
        
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                # class_1과 class_2에 특별한 가중치 부여
                weight = special_weight if (i == 1 and j == 2) or (i == 2 and j == 1) else 1.0
                
                dist = torch.sum((centers[i] - centers[j]) ** 2)
                # 거리가 margin보다 작으면 페널티 부여
                inter_loss += weight * torch.clamp(self.margin - dist, min=0)
                n_pairs += 1
        
        if n_pairs > 0:
            inter_loss = inter_loss / n_pairs
        
        # 최종 손실: 클래스 내 거리는 최소화, 클래스 간 거리는 최대화
        loss = self.lambda_intra * intra_loss + self.lambda_inter * inter_loss
        
        return loss

class ContrastiveCenterLoss(nn.Module):
    """
    대조적 중심 손실 (Contrastive Center Loss)
    
    특징 공간에서 같은 클래스의 샘플들은 가깝게, 다른 클래스의 샘플들은 멀게 배치하도록 학습합니다.
    특히 t-SNE 시각화에서 class_1과 class_2가 겹치는 문제를 해결하기 위해 설계되었습니다.
    """
    def __init__(self, num_classes, feat_dim, temperature=0.07, lambda_val=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temperature = temperature  # 대조 학습의 온도 파라미터
        self.lambda_val = lambda_val  # 손실 가중치
        
        # 클래스별 프로토타입 벡터 초기화 (학습 가능)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, targets):
        """
        Args:
            features: 특징 벡터 (배치 크기 x 특징 차원)
            targets: 타겟 클래스 레이블 (배치 크기)
        """
        batch_size = features.size(0)
        
        # 특징을 정규화 (cosine similarity 계산을 위해)
        features = F.normalize(features, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)
        
        # 클래스 간 유사도 행렬 계산
        sim_matrix = torch.matmul(features, centers.T) / self.temperature
        
        # 각 샘플에 대한 손실 계산
        losses = []
        for i in range(batch_size):
            pos_idx = targets[i].item()  # 현재 샘플의 실제 클래스
            
            # 양성(positive) 클래스와의 유사도
            pos_sim = sim_matrix[i, pos_idx]
            
            # 모든 클래스와의 유사도 (로그-섬-지수 형태로)
            neg_sim = torch.logsumexp(sim_matrix[i], dim=0)
            
            # InfoNCE 손실
            curr_loss = -pos_sim + neg_sim
            
            # class_1과 class_2에 대해 추가 가중치 부여
            if pos_idx == 1 or pos_idx == 2:
                curr_loss = curr_loss * 2.0  # 더 많은 가중치 부여
                
            losses.append(curr_loss)
        
        # 평균 손실 계산
        loss = torch.stack(losses).mean() * self.lambda_val
        
        return loss

# 손실 함수 선택 함수
def get_loss_function(loss_name, class_weights=None, num_classes=None, **kwargs):
    """
    손실 함수 이름에 따라 손실 함수 객체를 반환
    
    Args:
        loss_name (str): 손실 함수 이름 ('ce', 'focal', 'smoothing', 'dice', 'centroid', 'contrastive')
        class_weights (torch.Tensor): 클래스 가중치
        num_classes (int): 클래스 수
        **kwargs: 추가 파라미터 (focal_gamma, smoothing_alpha, feat_dim 등)
    
    Returns:
        nn.Module: 선택된 손실 함수
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

# 클래스 가중치 계산 함수
def calculate_class_weights(loader, num_classes, device):
    """
    데이터 로더에서 클래스 가중치를 계산하는 함수
    
    Args:
        loader: 데이터 로더
        num_classes: 클래스 수
        device: 계산에 사용할 디바이스
        
    Returns:
        torch.Tensor: 클래스 가중치
    """
    class_counts = torch.zeros(num_classes)
    
    # 클래스별 샘플 수 계산
    for batch in loader:
        for cls in range(num_classes):
            class_counts[cls] += (batch.y == cls).sum().item()
    
    # 가중치 계산 (샘플 수가 적은 클래스에 더 높은 가중치)
    weights = 1.0 / (class_counts + 1e-8)  # zero division 방지
    weights = weights / weights.sum() * num_classes  # 정규화
    
    return weights.to(device)


class VehicleDetectorWithGNN:
    def __init__(self, model_path, gnn_model_path=None, conf_thresh=0.25, iou_thresh=0.45, device=None, lane_json_path=None,
                 model_type='GraphSAGE', hidden_channels=128, num_layers=5, edge_dim=9, num_classes=3, 
                 node_features=8, aggr='mean', attention_type='mlp', num_relations=7, graph_data_type='image'):
        # 디바이스 설정
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # YOLO 모델 로드
        self.model = YOLO(model_path)
        print(f"YOLO Model loaded from {model_path}")
        
        # GNN 모델 설정값 저장
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_features = node_features
        self.aggr = aggr
        self.attention_type = attention_type
        self.num_relations = num_relations
        self.graph_data_type = graph_data_type  # graph_data_type 저장
        
        # 설정값 초기화
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.current_frame = 0
        self.vehicle_history = {}
        self.history_length = 10
        
        # 모션 상태 이름 및 색상 설정
        self.motion_state_names = {
            0: "Stop",
            1: "Lane Change",
            2: "Normal Driving"
        }
        
        self.motion_colors = {
            0: (0, 0, 255),    # 정지: 빨간색
            1: (0, 255, 255),  # 차선변경: 노란색
            2: (0, 255, 0)     # 직진: 초록색
        }
        
        # GNN 모델 로드
        self.gnn_model = None
        if gnn_model_path and os.path.exists(gnn_model_path):
            try:
                # 모델 타입에 따른 모델 인스턴스 생성
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
                    # NAGraphSAGE_Traditional은 선택적으로 import
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
                
                # 모델을 디바이스로 이동
                if self.gnn_model is not None:
                    self.gnn_model = self.gnn_model.to(self.device)
                    
                    # 저장된 가중치 로드
                    state_dict = torch.load(gnn_model_path, map_location=self.device)
                    
                    # state_dict가 OrderedDict인 경우 모델 state_dict로 간주하고 로드
                    if isinstance(state_dict, collections.OrderedDict) or isinstance(state_dict, dict):
                        self.gnn_model.load_state_dict(state_dict)
                    # 전체 모델인 경우 state_dict 속성 사용
                    elif hasattr(state_dict, 'state_dict'):
                        self.gnn_model.load_state_dict(state_dict.state_dict())
                    # 다른 형식으로 저장된 경우 (예: checkpoint)
                    else:
                        if 'model' in state_dict:
                            self.gnn_model.load_state_dict(state_dict['model'])
                        elif 'state_dict' in state_dict:
                            self.gnn_model.load_state_dict(state_dict['state_dict'])
                        else:
                            raise ValueError("알 수 없는 모델 형식입니다.")
                    
                    print(f"GNN 모델 로드 성공: {gnn_model_path}")
                    print(f"모델 타입: {model_type}")
                    self.gnn_model.eval()  # 평가 모드 설정
                
            except Exception as e:
                print(f"GNN 모델 로드 실패: {e}")
                traceback.print_exc()
                self.gnn_model = None
        
        # 차선 데이터 관련 변수 초기화
        self.lane_areas = {}
        self.lane_centerlines = {}
        
        # 차선 데이터 로드 (제공된 경우)
        if lane_json_path and os.path.exists(lane_json_path):
            self.load_lane_data(lane_json_path)
    
    def load_lane_data(self, lane_json_path):
        """차선 데이터 로드"""
        try:
            print(f"차선 데이터 로드 시도: {lane_json_path}")
            with open(lane_json_path, 'r') as f:
                lane_data = json.load(f)
            
            # 차선 영역 및 중심선 추출
            lane_areas = {}
            lane_centerlines = {}
            
            for lane in lane_data.get('lanes', []):
                lane_id = lane.get('id')
                points = lane.get('points', [])
                
                if points and len(points) >= 4:
                    # 차선 영역을 Polygon으로 저장
                    polygon = Polygon(points)
                    lane_areas[lane_id] = polygon
                    
                    # 차선 중심선 계산 (첫 번째와 마지막 점의 중간 점들)
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
            print(f"차선 데이터 로드 완료: {len(lane_areas)}개 차선, {len(lane_centerlines)}개 중심선")
        except Exception as e:
            print(f"차선 데이터 로드 실패: {e}")
            traceback.print_exc()
    
    def detect(self, frame):
        """
        프레임에서 차량 검출
        """
        try:
            results = self.model.track(frame, persist=True, conf=self.conf, iou=self.iou, tracker="bytetrack.yaml")
            return results
        except Exception as e:
            print(f"차량 검출 중 오류 발생: {e}")
            traceback.print_exc()
            return None
    
    def create_vehicle_graph(self, detections, width, height):
        """
        차량 데이터를 그래프 형태로 변환
        """
        if len(detections) == 0:
            return None
        
        try:
            features = []
            edge_index = []
            edge_attr = []
            
            # 가까운 차량을 연결하는 거리 임계값
            distance_threshold = 200  # 픽셀 단위
            
            # 각 차량의 특성 벡터 생성
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id, track_id = det
                
                # 중심점 및 크기 계산
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width_box = x2 - x1
                height_box = y2 - y1
                
                # 정규화된 좌표로 변환
                norm_x = center_x / width
                norm_y = center_y / height
                norm_width = width_box / width
                norm_height = height_box / height
                
                # 속도 및 방향 계산
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
                    
                    # 속도와 방향 계산
                    speed = displacement
                    if displacement > 0:
                        direction_x = dx / displacement
                        direction_y = dy / displacement
                    
                    # 가속도 계산
                    prev_speed = prev_info.get('speed', 0.0)
                    acceleration = speed - prev_speed
                
                # 특성 벡터: [x, y, w, h, speed, dir_x, dir_y, acceleration]
                feature = [
                    norm_x, norm_y, norm_width, norm_height,
                    speed, direction_x, direction_y, acceleration
                ]
                
                features.append(feature)
            
            # 엣지 생성 (가까운 차량 연결)
            for i in range(len(detections)):
                x1i, y1i, x2i, y2i, _, _, _ = detections[i]
                center_i = [(x1i + x2i) / 2, (y1i + y2i) / 2]
                
                for j in range(len(detections)):
                    if i == j:
                        continue
                    
                    x1j, y1j, x2j, y2j, _, _, _ = detections[j]
                    center_j = [(x1j + x2j) / 2, (y1j + y2j) / 2]
                    
                    # 두 차량 간의 거리 계산
                    distance = math.sqrt(
                        (center_i[0] - center_j[0])**2 + 
                        (center_i[1] - center_j[1])**2
                    )
                    
                    # 거리가 임계값보다 작으면 엣지 생성
                    if distance < distance_threshold:
                        edge_index.append([i, j])
                        edge_attr.append([distance / distance_threshold])  # 정규화된 거리
            
            # PyTorch Geometric 데이터 객체 생성
            if features and edge_index:
                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                return data
            else:
                # 엣지가 없는 경우 (노드만 있는 그래프)
                if features:
                    x = torch.tensor(features, dtype=torch.float32).to(self.device)
                    data = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long).to(self.device), 
                               edge_attr=torch.zeros((0, 1), dtype=torch.float).to(self.device))
                    return data
        except Exception as e:
            print(f"그래프 생성 중 오류: {e}")
            traceback.print_exc()
        
        return None
    
    def _serialize_for_json(self, data):
        """NumPy 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환"""
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