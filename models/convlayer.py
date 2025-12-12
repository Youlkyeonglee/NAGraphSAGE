import torch
from torch import Tensor
import warnings
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.nn import GCN, GAT  # 추가
        
# 4. 구체적인 적용 예시: 거리와 속도에 다른 가중치 부여
# 더 구체적인 예로, 엣지 속성 중 거리와 속도, 바운딩 박스 위치 등에 다른 가중치를 부여하는 방법을 보여드리겠습니다:
class NeighborAwareSAGEConv(MessagePassing):
    """
    엣지 속성의 종류별로 다른 가중치를 적용하는 GraphSAGE 레이어
    """
    def __init__(self, in_channels, out_channels, edge_dim=9, aggr='mean', graph_data_type='image'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.graph_data_type = graph_data_type
        # 메시지 변환을 위한 선형 레이어
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
        # 루트 노드(자기 자신) 가중치
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        
        # 엣지 속성 카테고리별 가중치 (예: 거리, 속도, 방향, 가속도, 위치/크기)
        if self.graph_data_type == 'image':
            self.distance_proj = torch.nn.Linear(1, 8)  # 거리
            self.speed_proj = torch.nn.Linear(1, 8)     # 속도
            self.dir_proj = torch.nn.Linear(3, 8)       # 방향
            self.accel_proj = torch.nn.Linear(1, 8)     # 가속도
            self.bbox_proj = torch.nn.Linear(4, 8)      # 바운딩 박스
        elif self.graph_data_type == 'world':
            self.distance_proj = torch.nn.Linear(1, 8)  # 거리
            self.speed_proj = torch.nn.Linear(1, 8)     # 속도
            self.dir_proj = torch.nn.Linear(3, 8)       # 방향
            self.accel_proj = torch.nn.Linear(1, 8)     # 가속도
            self.position_proj = torch.nn.Linear(3, 8)      # 바운딩 박스
        
        # 최종 가중치 계산
        
        # edge_dim에 따른 input_size 계산
        if self.graph_data_type == 'image':
            if edge_dim == 5:  # bbox(4), 속도(1)
                input_size = 16  # 8 * 2 (각 특성당 8차원)
            elif edge_dim == 6:  # bbox(4), 속도(1), 가속도(1)
                input_size = 24  # 8 * 3 (bbox, 속도, 가속도)
            elif edge_dim == 7:  # bbox(4), 속도(1), 가속도(1), 거리(1)
                input_size = 32  # 8 * 4 (bbox, 속도, 가속도, 거리)
            elif edge_dim == 10:  # bbox(4), 속도(1), 방향(3), 가속도(1), 거리(1)
                input_size = 40  # 8 * 5 (bbox, 속도, 가속도, 거리, 방향)
            else:
                raise ValueError(f"Unsupported edge_dim: {edge_dim}")
        elif self.graph_data_type == 'world':
            if edge_dim == 4:  # position(3), 속도(1)
                input_size = 16  # 8 * 2 (각 특성당 8차원)
            elif edge_dim == 5:  # position(3), 속도(1)
                input_size = 24  # 8 * 3 (bbox, 속도, 가속도)
            elif edge_dim == 6:  # position(3), 속도(1), 가속도(1)
                input_size = 32  # 8 * 4 (bbox, 속도, 가속도, 거리)
            elif edge_dim == 9:  # position(3), 속도(1), 방향(3), 가속도(1), 거리(1)
                input_size = 40  # 8 * 5 (bbox, 속도, 가속도, 거리, 방향)
            else:
                raise ValueError(f"Unsupported edge_dim: {edge_dim}")
        self.weight_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),  # 각 특성당 8차원으로 투영된 값들의 총합
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_self.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None):
        # 자기 자신에 대한 변환
        x_self = self.lin_self(x)
        # edge_index 디버깅
        # print("--------------------------------")
        # print("edge_index 형태:", edge_index.shape)
        # # 타겟 노드별 등장 횟수 계산
        # unique_targets, target_counts = torch.unique(edge_index[1], return_counts=True)
        # print("타겟 노드별 등장 횟수:")
        # for target, count in zip(unique_targets, target_counts):  # 처음 5개만 출력
        #     if count == 5:
        #         print(f"노드 {target}: {count}회 등장")
        # print("...")  # 나머지는 생략
        # print("타겟 노드 고유 값 수:", edge_index[1].unique().size(0))       # 고유한 타겟 노드 수
        # print("전체 노드 수:", x.size(0))                                   # 전체 노드 수
        # print("전체 엣지 수:", edge_index.size(1))                          # 전체 엣지 수
        # print("--------------------------------")

        # print("--------------------------------")
        # print("x_self", x_self)
        # print("x_self.shape", x_self.shape) #449092,128: train loader 데이터 크기 / 96233,128: val, test loader 데이터 크기
        # print("--------------------------------")
        
        # 이웃 노드로부터의 메시지 전파
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # print("--------------------------------")
        # print("out", out)
        # print("out.shape", out.shape) #[449092,128]: train loader 데이터 크기 / [96,233,128] val, test loader 데이터 크기
        # print("--------------------------------")
        # 결합
        return x_self + out
    
    def message(self, x_j, edge_attr):
        # 엣지 속성이 유효한지 확인하고 각 특성을 처리합니다
        edge_features = []
        
        if edge_attr is not None:
            if self.graph_data_type == 'image':
                if self.edge_dim == 5:  # bbox(4), 속도(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)

                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    
                elif self.edge_dim == 6:  # bbox(4), 속도(1), 가속도(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 7:  # bbox(4), 속도(1), 가속도(1), 거리(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. 거리 (1차원)
                    distance = edge_attr[:, 6:7]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 10:  # bbox(4), 속도(1), 방향(3), 가속도(1), 거리(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 방향 (3차원)
                    direction = edge_attr[:, 5:8]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)

                    # 4. 가속도 (1차원)
                    accel = edge_attr[:, 8:9]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. 거리 (1차원)
                    distance = edge_attr[:, 9:10]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
            elif self.graph_data_type == 'world':
                if self.edge_dim == 4:  # position(3), 속도(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                elif self.edge_dim == 5:  # position(3), 속도(1), 가속도(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                elif self.edge_dim == 6:  # position(3), 속도(1), 가속도(1), 거리(1) 
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    # 4. 거리 (1차원)
                    distance = edge_attr[:, 5:6]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                elif self.edge_dim == 9:  # position(3), 속도(1), 방향(3), 가속도(1), 거리(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. 방향 (3차원)
                    direction = edge_attr[:, 4:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)
                    # 4. 가속도 (1차원)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    # 5. 거리 (1차원)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
        # 모든 변환된 특성 결합
        # print("--------------------------------")
        # print("edge_features", edge_features)
        # print("--------------------------------")
        if edge_features:
            h_combined = torch.cat(edge_features, dim=1)
            
            # 최종 가중치 계산
            weight = self.weight_net(h_combined)
            
            # 가중치를 저장 (시각화나 분석을 위해)
            self.last_attention_weights = weight
            
            # 기본 노드 변환
            x_j = self.lin(x_j)
            
            return x_j * weight
        else:
            # 엣지 속성이 없는 경우 기본 노드 변환만 수행
            self.last_edge_weights = None
            return self.lin(x_j)
    
    def get_attention_weights(self):
        return self.last_attention_weights

class NeighborAware_SAGEConv_Attention_NodeLayer(MessagePassing):
    """
    엣지 속성의 종류별로 다른 가중치를 적용하는 GraphSAGE 레이어
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
        self.batch_size = batch_size  # 배치 처리 크기
        self.scale = scale  # 스케일링 팩터
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        # 메시지 변환을 위한 선형 레이어
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
        # 루트 노드(자기 자신) 가중치
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        self.lin_self2 = torch.nn.Linear(out_channels, out_channels)
        self.gconv = GCN(in_channels, out_channels, num_layers=2, out_channels=out_channels, dropout=0.5)
        # self.gconv2 = GCN(out_channels, out_channels, cached=True, normalize=True)
        self.gat = GAT(in_channels=in_channels, hidden_channels=out_channels, out_channels=out_channels, num_layers=2, dropout=0.5)
        # 1) R-GCN 레이어 (노드 자체 특징 → 구조 인코딩 특징)

        
        
        # 엣지 속성 카테고리별 가중치 (예: 거리, 속도, 방향, 가속도, 위치/크기)
        if self.graph_data_type == 'image':
            self.distance_proj = torch.nn.Linear(1, 8)  # 거리
            self.speed_proj = torch.nn.Linear(1, 8)     # 속도
            self.dir_proj = torch.nn.Linear(3, 8)       # 방향
            self.accel_proj = torch.nn.Linear(1, 8)     # 가속도
            self.bbox_proj = torch.nn.Linear(4, 8)      # 바운딩 박스
        elif self.graph_data_type == 'world':
            self.distance_proj = torch.nn.Linear(1, 8)  # 거리
            self.speed_proj = torch.nn.Linear(1, 8)     # 속도
            self.dir_proj = torch.nn.Linear(3, 8)       # 방향
            self.accel_proj = torch.nn.Linear(1, 8)     # 가속도
            self.position_proj = torch.nn.Linear(3, 8)      # 바운딩 박스
        
        # 최종 가중치 계산
        # edge_dim에 따른 input_size 계산
        if self.graph_data_type == 'image':
            if self.edge_dim == 5:  # bbox(4), 속도(1)
                input_size = 16  # 8 * 2 (각 특성당 8차원)
            elif self.edge_dim == 6:  # bbox(4), 속도(1), 가속도(1)
                input_size = 24  # 8 * 3 (bbox, 속도, 가속도)
            elif self.edge_dim == 7:  # bbox(4), 속도(1), 가속도(1), 거리(1)
                input_size = 32  # 8 * 4 (bbox, 속도, 가속도, 거리)
            elif self.edge_dim == 10:  # bbox(4), 속도(1), 방향(3), 가속도(1), 거리(1)
                input_size = 40  # 8 * 5 (bbox, 속도, 가속도, 거리, 방향)
            else:
                raise ValueError(f"Unsupported edge_dim: {self.edge_dim}")
        elif self.graph_data_type == 'world':
            if self.edge_dim == 4:  # position(3), 속도(1)
                input_size = 16  # 8 * 2 (각 특성당 8차원)
            elif self.edge_dim == 5:  # position(3), 속도(1), 가속도(1)
                input_size = 24  # 8 * 3 (bbox, 속도, 가속도)
            elif self.edge_dim == 6:  # position(3), 속도(1), 가속도(1), 거리(1)
                input_size = 32  # 8 * 4 (bbox, 속도, 가속도, 거리)
            elif self.edge_dim == 9:  # position(3), 속도(1), 방향(3), 가속도(1), 거리(1)
                input_size = 40  # 8 * 5 (bbox, 속도, 가속도, 거리, 방향)
            else:
                raise ValueError(f"Unsupported edge_dim: {self.edge_dim}")

        self._last_attention_weights = None  # 어텐션 가중치 저장용
        
        # 어텐션 타입에 따른 레이어 초기화
        if attention_type == 'mlp':
            self.attention_layer = torch.nn.Sequential(
                torch.nn.Linear(input_size, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid()
            )
        elif attention_type == 'inner_product':
            self.inner_product_key = torch.nn.Linear(input_size, 16)
            self.inner_product_query = torch.nn.Linear(input_size, 16)
        elif attention_type == 'cosine':
            self.cosine_key = torch.nn.Linear(input_size, 16)
            self.cosine_query = torch.nn.Linear(input_size, 16)
            self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        # 공통 값 벡터 투영 (v)
        self.value_proj = torch.nn.Linear(input_size, 16)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_self.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None):


        if self.type_node_layer == 'traditional':
            x_self = self.lin_self(x)
        elif self.type_node_layer == 'gcn':
            x_self = self.gconv(x, edge_index)
        elif self.type_node_layer == 'gat':
            x_self = self.gat(x, edge_index)
        else:
            raise ValueError(f"Unsupported type_node_layer: {self.type_node_layer}")
        
        # 이웃 노드로부터의 메시지 전파
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 결합
        return x_self + out
    
    def message(self, x_j, edge_attr):
        # 엣지 속성이 유효한지 확인하고 각 특성을 처리합니다
        edge_features = []
        
        # 엣지 특성 처리
        if edge_attr is not None and edge_attr.size(1) > 0:  # edge_attr가 비어있지 않은지 확인
            if self.graph_data_type == 'image':
                if self.edge_dim == 5:  # bbox(4), 속도(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)

                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    
                elif self.edge_dim == 6:  # bbox(4), 속도(1), 가속도(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 7:  # bbox(4), 속도(1), 가속도(1), 거리(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. 거리 (1차원)
                    distance = edge_attr[:, 6:7]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 9:  # bbox(4), 속도(1), 방향(2), 가속도(1), 거리(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 방향 (2차원)
                    direction = edge_attr[:, 5:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)

                    # 4. 가속도 (1차원)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. 거리 (1차원)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
            elif self.graph_data_type == 'world':
                if self.edge_dim == 4:  # position(3), 속도(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                elif self.edge_dim == 5:  # position(3), 속도(1), 가속도(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 6:  # position(3), 속도(1), 가속도(1), 거리(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. 거리 (1차원)
                    distance = edge_attr[:, 5:6]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 9:  # position(3), 속도(1), 방향(3), 가속도(1), 거리(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 방향 (3차원)
                    direction = edge_attr[:, 4:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)
                    
                    # 4. 가속도 (1차원)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. 거리 (1차원)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                
        
        # 모든 변환된 특성 결합
        if edge_features:
            h_combined = torch.cat(edge_features, dim=1)
            
            x_j = self.lin(x_j)
            
            # 어텐션 타입에 따라 가중치 계산 방식 다르게 적용
            if self.attention_type == 'mlp':
                attention_weights = self.attention_layer(h_combined)
            
            elif self.attention_type == 'inner_product':
                # 배치 처리를 위한 준비
                num_edges = h_combined.size(0)
                
                query = self.inner_product_query(h_combined) # [num_edges, hidden_size]
                key = self.inner_product_key(h_combined) # [num_edges, hidden_size]

                # 배치 처리로 메모리 사용량 감소
                scores_chunks = []
                for start in range(0, num_edges, self.batch_size):
                    end = min(start + self.batch_size, num_edges)
                    
                    # bmm 대신 element-wise 곱과 sum 사용 (메모리 효율적)
                    batch_q = query[start:end]
                    batch_k = key[start:end]
                    batch_scores = torch.sum(batch_q * batch_k, dim=1) / self.scale
                    scores_chunks.append(batch_scores)

                # 모든 배치 결과 결합
                scores = torch.cat(scores_chunks, dim=0)
                attention_weights = F.softmax(scores, dim=0).unsqueeze(-1)
            
            elif self.attention_type == 'cosine':
                # 3. 코사인 유사도 기반 어텐션
                query = self.cosine_key(h_combined)
                key = self.cosine_query(h_combined)
                
                # 코사인 유사도 계산
                scores_chunks = []
                num_edges = h_combined.size(0)
                for start in range(0, num_edges, self.batch_size):
                    end = min(start + self.batch_size, num_edges)
                    
                    # bmm 대신 element-wise 곱과 sum 사용 (메모리 효율적)
                    batch_q = query[start:end]
                    batch_k = key[start:end]
                    batch_scores = self.cosine(batch_q, batch_k)
                    # print("batch_scores.shape: ", batch_scores.shape)
                    scores_chunks.append(batch_scores)
                # print("scores_chunks[0].shape: ", scores_chunks[0].shape)
                attention_weights = torch.cat(scores_chunks, dim=0).unsqueeze(-1)
            
            
            # 가중치를 저장 (시각화나 분석을 위해)
            self.last_attention_weights = attention_weights
            
            # 변환된 노드 특성에 어텐션 가중치 적용
            return x_j * attention_weights
        else:
            # 엣지 속성이 없는 경우 기본 노드 변환만 수행
            return self.lin(x_j)
        
        
    def get_attention_weights(self):
        return self._last_attention_weights

class AttributeAwareSAGEConv(MessagePassing):
    """
    엣지 속성의 종류별로 다른 가중치를 적용하는 GraphSAGE 레이어
    """
    def __init__(self, in_channels, out_channels, edge_dim=9, aggr='mean', graph_data_type='image'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.graph_data_type = graph_data_type
        # 메시지 변환을 위한 선형 레이어
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
        # 루트 노드(자기 자신) 가중치
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        
        # 엣지 속성 카테고리별 가중치 (예: 거리, 속도, 방향, 가속도, 위치/크기)
        if self.graph_data_type == 'image':
            self.distance_proj = torch.nn.Linear(1, 8)  # 거리
            self.speed_proj = torch.nn.Linear(1, 8)     # 속도
            self.dir_proj = torch.nn.Linear(3, 8)       # 방향
            self.accel_proj = torch.nn.Linear(1, 8)     # 가속도
            self.bbox_proj = torch.nn.Linear(4, 8)      # 바운딩 박스
        elif self.graph_data_type == 'world':
            self.distance_proj = torch.nn.Linear(1, 8)  # 거리
            self.speed_proj = torch.nn.Linear(1, 8)     # 속도
            self.dir_proj = torch.nn.Linear(3, 8)       # 방향
            self.accel_proj = torch.nn.Linear(1, 8)     # 가속도
            self.position_proj = torch.nn.Linear(3, 8)      # 바운딩 박스
        
        # 최종 가중치 계산
        
        # edge_dim에 따른 input_size 계산
        if self.graph_data_type == 'image':
            if edge_dim == 5:  # bbox(4), 속도(1)
                input_size = 16  # 8 * 2 (각 특성당 8차원)
            elif edge_dim == 6:  # bbox(4), 속도(1), 가속도(1)
                input_size = 24  # 8 * 3 (bbox, 속도, 가속도)
            elif edge_dim == 7:  # bbox(4), 속도(1), 가속도(1), 거리(1)
                input_size = 32  # 8 * 4 (bbox, 속도, 가속도, 거리)
            elif edge_dim == 10:  # bbox(4), 속도(1), 방향(3), 가속도(1), 거리(1)
                input_size = 40  # 8 * 5 (bbox, 속도, 가속도, 거리, 방향)
            else:
                raise ValueError(f"Unsupported edge_dim: {edge_dim}")
        elif self.graph_data_type == 'world':
            if edge_dim == 4:  # position(3), 속도(1)
                input_size = 16  # 8 * 2 (각 특성당 8차원)
            elif edge_dim == 5:  # position(3), 속도(1)
                input_size = 24  # 8 * 3 (bbox, 속도, 가속도)
            elif edge_dim == 6:  # position(3), 속도(1), 가속도(1)
                input_size = 32  # 8 * 4 (bbox, 속도, 가속도, 거리)
            elif edge_dim == 9:  # position(3), 속도(1), 방향(3), 가속도(1), 거리(1)
                input_size = 40  # 8 * 5 (bbox, 속도, 가속도, 거리, 방향)
            else:
                raise ValueError(f"Unsupported edge_dim: {edge_dim}")
        self.weight_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),  # 각 특성당 8차원으로 투영된 값들의 총합
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_self.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None):
        # 자기 자신에 대한 변환
        x_self = self.lin_self(x)
        # edge_index 디버깅
        # print("--------------------------------")
        # print("edge_index 형태:", edge_index.shape)
        # # 타겟 노드별 등장 횟수 계산
        # unique_targets, target_counts = torch.unique(edge_index[1], return_counts=True)
        # print("타겟 노드별 등장 횟수:")
        # for target, count in zip(unique_targets, target_counts):  # 처음 5개만 출력
        #     if count == 5:
        #         print(f"노드 {target}: {count}회 등장")
        # print("...")  # 나머지는 생략
        # print("타겟 노드 고유 값 수:", edge_index[1].unique().size(0))       # 고유한 타겟 노드 수
        # print("전체 노드 수:", x.size(0))                                   # 전체 노드 수
        # print("전체 엣지 수:", edge_index.size(1))                          # 전체 엣지 수
        # print("--------------------------------")

        # print("--------------------------------")
        # print("x_self", x_self)
        # print("x_self.shape", x_self.shape) #449092,128: train loader 데이터 크기 / 96233,128: val, test loader 데이터 크기
        # print("--------------------------------")
        
        # 이웃 노드로부터의 메시지 전파
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # print("--------------------------------")
        # print("out", out)
        # print("out.shape", out.shape) #[449092,128]: train loader 데이터 크기 / [96,233,128] val, test loader 데이터 크기
        # print("--------------------------------")
        # 결합
        return x_self + out
    
    def message(self, x_j, edge_attr):
        # 엣지 속성이 유효한지 확인하고 각 특성을 처리합니다
        edge_features = []
        
        if edge_attr is not None:
            if self.graph_data_type == 'image':
                if self.edge_dim == 5:  # bbox(4), 속도(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)

                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    
                elif self.edge_dim == 6:  # bbox(4), 속도(1), 가속도(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                elif self.edge_dim == 7:  # bbox(4), 속도(1), 가속도(1), 거리(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 5:6]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 4. 거리 (1차원)
                    distance = edge_attr[:, 6:7]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
                elif self.edge_dim == 10:  # bbox(4), 속도(1), 방향(3), 가속도(1), 거리(1)
                    # 1. bbox (4차원)
                    bbox = edge_attr[:, 0:4]
                    h_bbox = self.bbox_proj(bbox)
                    edge_features.append(h_bbox)
                    
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 4:5]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    
                    # 3. 방향 (3차원)
                    direction = edge_attr[:, 5:8]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)

                    # 4. 가속도 (1차원)
                    accel = edge_attr[:, 8:9]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    
                    # 5. 거리 (1차원)
                    distance = edge_attr[:, 9:10]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
            elif self.graph_data_type == 'world':
                if self.edge_dim == 4:  # position(3), 속도(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                elif self.edge_dim == 5:  # position(3), 속도(1), 가속도(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                elif self.edge_dim == 6:  # position(3), 속도(1), 가속도(1), 거리(1) 
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. 가속도 (1차원)
                    accel = edge_attr[:, 4:5]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    # 4. 거리 (1차원)
                    distance = edge_attr[:, 5:6]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                elif self.edge_dim == 9:  # position(3), 속도(1), 방향(3), 가속도(1), 거리(1)
                    # 1. position (3차원)
                    position = edge_attr[:, 0:3]
                    h_position = self.position_proj(position)
                    edge_features.append(h_position)
                    # 2. 속도 (1차원)
                    speed = edge_attr[:, 3:4]
                    h_speed = self.speed_proj(speed)
                    edge_features.append(h_speed)
                    # 3. 방향 (3차원)
                    direction = edge_attr[:, 4:7]
                    h_dir = self.dir_proj(direction)
                    edge_features.append(h_dir)
                    # 4. 가속도 (1차원)
                    accel = edge_attr[:, 7:8]
                    h_accel = self.accel_proj(accel)
                    edge_features.append(h_accel)
                    # 5. 거리 (1차원)
                    distance = edge_attr[:, 8:9]
                    h_distance = self.distance_proj(distance)
                    edge_features.append(h_distance)
                    
        # 모든 변환된 특성 결합
        # print("--------------------------------")
        # print("edge_features", edge_features)
        # print("--------------------------------")
        if edge_features:
            h_combined = torch.cat(edge_features, dim=1)
            
            # 최종 가중치 계산
            weight = self.weight_net(h_combined)
            
            # 가중치를 저장 (시각화나 분석을 위해)
            self.last_edge_weights = weight.detach()
            
            # 기본 노드 변환
            x_j = self.lin(x_j)
            
            return x_j * weight
        else:
            # 엣지 속성이 없는 경우 기본 노드 변환만 수행
            self.last_edge_weights = None
            return self.lin(x_j)
    
    def get_edge_weights(self):
        """저장된 엣지 가중치를 반환합니다."""
        return self.last_edge_weights if hasattr(self, 'last_edge_weights') else None