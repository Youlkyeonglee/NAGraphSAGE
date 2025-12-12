import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, SAGEConv
from torch.nn import Linear
from models.convlayer import NeighborAwareSAGEConv, NeighborAware_SAGEConv_Attention_NodeLayer


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCN, self).__init__()
        
        # 레이어 수 저장
        self.num_layers = num_layers
        
        # 컨볼루션 레이어 리스트
        self.convs = torch.nn.ModuleList()
        
        # 첫 번째 레이어
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        # 마지막 레이어
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # 배치 정규화
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # 드롭아웃 비율
        self.dropout = 0.3
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 서브그래프 생성 (배치 내 노드만 포함)
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        
        # 컨볼루션 레이어 통과
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # 마지막 레이어
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

# 기존 GAT 모델 유지 (backwards compatibility)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.6, num_layers=2, edge_dim=5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        # OrderedDict 대신 직접 모듈을 정의
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim))
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, edge_dim=edge_dim)
    
    def forward(self, data, return_features=False):
        # Modify this line to extract node features and edge information safely
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Make sure edge_index only references nodes in the current batch
        # This creates a subgraph containing only the nodes present in the batch
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # 첫 번째 층: multi-head attention 적용 후 concatenate
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # 중간 층들: convs에 있는 레이어들 적용
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
        
        # 마지막 층 전의 특징 저장
        final_features = x
        
        # 마지막 층: multi-head attention 적용
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 최종 출력에 log_softmax 적용
        out = F.log_softmax(x, dim=1)
        
        if return_features:
            return out, final_features
        return out


# EGAT (Edge-enhanced Graph Attention Network)
class EGAT(torch.nn.Module):
    """
    Edge-enhanced Graph Attention Network
    엣지 특성을 활용하여 그래프의 구조적 정보를 더욱 효과적으로 학습하는 GAT 모델
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6, num_layers=2, edge_dim=5):
        super(EGAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        
        # 첫 번째 레이어: 엣지 특성 포함
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=dropout)
        
        # 중간 레이어들
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=dropout))
        
        # 마지막 레이어: 단일 헤드로 출력 차원 맞춤
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, edge_dim=edge_dim, dropout=dropout)
        
        # 배치 정규화
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads if _ < num_layers - 1 else out_channels))
    
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 서브그래프 생성 (배치 내 노드만 포함)
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # 첫 번째 레이어
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bns[0](x)
        x = F.elu(x)
        
        # 중간 레이어들
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i + 1](x)
            x = F.elu(x)
        
        # 마지막 레이어 전의 특징 저장
        final_features = x
        
        # 마지막 레이어
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bns[-1](x)
        
        # 최종 출력에 log_softmax 적용
        out = F.log_softmax(x, dim=1)
        
        if return_features:
            return out, final_features
        return out


    
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4, 
                 dropout=0.3, edge_dim=None, graph_data_type='image'):
        super(GraphTransformer, self).__init__()
        self.graph_data_type = graph_data_type
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 입력 임베딩 레이어
        self.input_embedding = Linear(in_channels, hidden_channels)
        
        # 트랜스포머 레이어 스택
        self.transformer_layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            # 멀티헤드 어텐션 레이어
            self.transformer_layers.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True
                )
            )
            # 레이어 정규화
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
            
        # 출력 레이어
        self.output_layer = Linear(hidden_channels, out_channels)
        
        # 각 레이어의 출력을 저장할 속성
        self.layer_outputs = None
        
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 입력 임베딩
        x = self.input_embedding(x)
        
        # 각 레이어의 출력을 저장할 리스트
        layer_outputs = []
        
        # 트랜스포머 레이어 통과
        for transformer, norm in zip(self.transformer_layers, self.layer_norms):
            # 잔차 연결
            residual = x
            
            # 트랜스포머 레이어
            x = transformer(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 잔차 연결 더하기
            x = x + residual
            
            # 레이어 정규화
            x = norm(x)
            
            # 활성화 함수
            x = F.gelu(x)
            
            layer_outputs.append(x)
        final_features = x
        # 출력 레이어
        x = self.output_layer(x)
        
        # 소프트맥스 적용
        x = F.log_softmax(x, dim=-1)
        
        if return_features:
            return x, final_features
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, aggr='mean'):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # 첫 번째 레이어
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # 마지막 레이어
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # 최종 분류기
        self.classifier = Linear(hidden_channels, out_channels)
        
        # 배치 정규화
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Dropout
        self.dropout = 0.3
        
        # 각 레이어의 출력을 저장할 속성
        self.layer_outputs = None
    
    def forward(self, data, return_features=False):
        x, edge_index = data.x, data.edge_index
        
        # 각 레이어의 출력을 저장할 리스트
        layer_outputs = []
        
        # 각 그래프 합성곱 레이어 통과
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)  # 레이어 출력 저장
        
        # 분류 전 마지막 특징 벡터 저장
        final_features = x
        
        # 분류
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        # 레이어 출력 저장
        self.layer_outputs = layer_outputs
        
        
        if return_features:
            # 특징 벡터와 최종 출력을 함께 반환
            return out, final_features
        return out
    
    

class NeighborAwareGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=9, num_layers=3, aggr='mean', graph_data_type='image'):
        super(NeighborAwareGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # 첫 번째 레이어 (aggr 파라미터 전달)
        self.convs.append(NeighborAwareSAGEConv(in_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, graph_data_type=graph_data_type))
        
        # 중간 레이어들 (aggr 파라미터 전달)
        for _ in range(num_layers - 2):
            self.convs.append(NeighborAwareSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, graph_data_type=graph_data_type))
        
        # 마지막 레이어 (aggr 파라미터 전달)
        if num_layers > 1:
            self.convs.append(NeighborAwareSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, graph_data_type=graph_data_type))
        
        # 최종 분류기
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        
        # 배치 정규화
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Dropout
        self.dropout = 0.3
    
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 서브그래프 생성 (배치 내 노드만 포함)
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # 레이어 출력 저장
        layer_outputs = []
        
        # 그래프 합성곱 레이어 통과
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            if i < self.num_layers - 1:  # 마지막 레이어 전까지만 활성화 함수 적용
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # 분류 전 마지막 특징 벡터 저장
        final_features = x
        
        # 분류
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        # 레이어 출력 저장
        self.layer_outputs = layer_outputs
        
        if return_features:
            # 특징 벡터와 최종 출력을 함께 반환
            return out, final_features
        return out
    
    def get_all_edge_weights(self):
        """모든 레이어의 엣지 가중치를 수집하여 반환합니다.
        
        Returns:
            list: 각 레이어의 엣지 가중치 리스트. 각 요소는 [num_edges, 1] 형태의 텐서입니다.
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
        
        # 첫 번째 레이어 (aggr 파라미터 전달)
        self.convs.append(NeighborAware_SAGEConv_Attention_NodeLayer(in_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, 
                                                                      attention_type=attention_type, batch_size=batch_size, scale=scale, 
                                                                      graph_data_type=graph_data_type, type_node_layer=type_node_layer))
        
        # 중간 레이어들 (aggr 파라미터 전달)
        for _ in range(num_layers - 2):
            self.convs.append(NeighborAware_SAGEConv_Attention_NodeLayer(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, attention_type=attention_type, batch_size=batch_size, scale=scale, graph_data_type=graph_data_type, type_node_layer=type_node_layer))
        
        # 마지막 레이어 (aggr 파라미터 전달)
        if num_layers > 1:
            self.convs.append(NeighborAware_SAGEConv_Attention_NodeLayer(hidden_channels, hidden_channels, edge_dim=edge_dim, aggr=aggr, attention_type=attention_type, batch_size=batch_size, scale=scale, graph_data_type=graph_data_type, type_node_layer=type_node_layer))
        
        # 최종 분류기
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        
        # 배치 정규화
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Dropout
        self.dropout = 0.3
    
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 서브그래프 생성 (배치 내 노드만 포함)
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # 레이어 출력 저장
        layer_outputs = []
        
        # 그래프 합성곱 레이어 통과
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            if i < self.num_layers - 1:  # 마지막 레이어 전까지만 활성화 함수 적용
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # 분류 전 마지막 특징 벡터 저장
        final_features = x
        
        # 분류
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        # 레이어 출력 저장
        self.layer_outputs = layer_outputs
        
        if return_features:
            # 특징 벡터와 최종 출력을 함께 반환
            return out, final_features
        return out


# ST-GCN (Spatial-Temporal Graph Convolutional Network)
class STGCN(torch.nn.Module):
    """Spatial-Temporal Graph Convolutional Network for time-series graph data"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 temporal_kernel_size=9, spatial_kernel_size=3, dropout=0.3):
        super(STGCN, self).__init__()
        self.num_layers = num_layers
        self.temporal_kernel_size = temporal_kernel_size
        self.dropout = dropout
        
        # 시간적 컨볼루션 레이어들
        self.temporal_convs = torch.nn.ModuleList()
        # 공간적 그래프 컨볼루션 레이어들
        self.spatial_convs = torch.nn.ModuleList()
        # 배치 정규화
        self.bns = torch.nn.ModuleList()
        
        # 첫 번째 레이어
        self.temporal_convs.append(
            torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=temporal_kernel_size, 
                          padding=(temporal_kernel_size - 1) // 2)
        )
        self.spatial_convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            self.temporal_convs.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=temporal_kernel_size,
                              padding=(temporal_kernel_size - 1) // 2)
            )
            self.spatial_convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 마지막 레이어
        if num_layers > 1:
            self.temporal_convs.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=temporal_kernel_size,
                              padding=(temporal_kernel_size - 1) // 2)
            )
            self.spatial_convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 최종 분류기
        self.classifier = Linear(hidden_channels, out_channels)
    
    def forward(self, data, return_features=False):
        x, edge_index = data.x, data.edge_index
        
        # 서브그래프 생성
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        
        # 노드 수와 특성 수
        num_nodes = x.size(0)
        num_features = x.size(1)
        
        # 시간적 차원을 추가하기 위해 reshape (batch_size=1로 가정)
        # 실제로는 시계열 데이터가 있어야 하지만, 여기서는 단일 타임스텝으로 처리
        x = x.unsqueeze(0)  # [1, num_nodes, num_features]
        x = x.transpose(1, 2)  # [1, num_features, num_nodes] - Conv1d를 위한 형태
        
        # ST-GCN 레이어 통과
        for i, (temporal_conv, spatial_conv, bn) in enumerate(zip(self.temporal_convs, self.spatial_convs, self.bns)):
            # 시간적 컨볼루션
            x = temporal_conv(x)  # [1, hidden_channels, num_nodes]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 공간적 그래프 컨볼루션을 위해 transpose
            x = x.transpose(1, 2).squeeze(0)  # [num_nodes, hidden_channels]
            
            # 공간적 그래프 컨볼루션
            x = spatial_conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 다음 레이어를 위해 다시 transpose
            x = x.unsqueeze(0).transpose(1, 2)  # [1, hidden_channels, num_nodes]
        
        # 최종 형태로 변환
        x = x.transpose(1, 2).squeeze(0)  # [num_nodes, hidden_channels]
        final_features = x
        
        # 분류
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        if return_features:
            return out, final_features
        return out


# ST-GraphTransformer (Spatial-Temporal Graph Transformer)
class STGraphTransformer(torch.nn.Module):
    """Spatial-Temporal Graph Transformer for time-series graph data"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 heads=4, dropout=0.3, edge_dim=None, temporal_kernel_size=3):
        super(STGraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # 시간적 컨볼루션 레이어
        self.temporal_convs = torch.nn.ModuleList()
        # 공간적 트랜스포머 레이어
        self.spatial_transformers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        # 입력 임베딩
        self.input_embedding = Linear(in_channels, hidden_channels)
        
        # 첫 번째 레이어
        self.temporal_convs.append(
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=temporal_kernel_size,
                          padding=(temporal_kernel_size - 1) // 2)
        )
        self.spatial_transformers.append(
            TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                          dropout=dropout, edge_dim=edge_dim, concat=True)
        )
        self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            self.temporal_convs.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=temporal_kernel_size,
                              padding=(temporal_kernel_size - 1) // 2)
            )
            self.spatial_transformers.append(
                TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                              dropout=dropout, edge_dim=edge_dim, concat=True)
            )
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
        
        # 마지막 레이어
        if num_layers > 1:
            self.temporal_convs.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=temporal_kernel_size,
                              padding=(temporal_kernel_size - 1) // 2)
            )
            self.spatial_transformers.append(
                TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                              dropout=dropout, edge_dim=edge_dim, concat=True)
            )
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
        
        # 출력 레이어
        self.output_layer = Linear(hidden_channels, out_channels)
    
    def forward(self, data, return_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 서브그래프 생성
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # 입력 임베딩
        x = self.input_embedding(x)
        
        # 시간적 차원 추가
        x = x.unsqueeze(0).transpose(1, 2)  # [1, hidden_channels, num_nodes]
        
        # ST-GraphTransformer 레이어 통과
        for temporal_conv, spatial_transformer, norm in zip(self.temporal_convs, self.spatial_transformers, self.layer_norms):
            # 시간적 컨볼루션
            x = temporal_conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 공간적 트랜스포머를 위해 transpose
            x = x.transpose(1, 2).squeeze(0)  # [num_nodes, hidden_channels]
            residual = x
            
            # 공간적 트랜스포머
            x = spatial_transformer(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
            x = norm(x)
            x = F.gelu(x)
            
            # 다음 레이어를 위해 다시 transpose
            x = x.unsqueeze(0).transpose(1, 2)  # [1, hidden_channels, num_nodes]
        
        # 최종 형태로 변환
        x = x.transpose(1, 2).squeeze(0)  # [num_nodes, hidden_channels]
        final_features = x
        
        # 출력 레이어
        x = self.output_layer(x)
        out = F.log_softmax(x, dim=1)
        
        if return_features:
            return out, final_features
        return out


# LSTM 모델 (시계열 데이터 처리)
class LSTMClassifier(torch.nn.Module):
    """LSTM-based classifier for sequential node features"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.3, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.bidirectional = bidirectional
        
        # LSTM 레이어
        self.lstm = torch.nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # LSTM 출력 차원 계산
        lstm_output_dim = hidden_channels * 2 if bidirectional else hidden_channels
        
        # 배치 정규화
        self.bn = torch.nn.BatchNorm1d(lstm_output_dim)
        
        # 최종 분류기
        self.classifier = Linear(lstm_output_dim, out_channels)
        
        self.dropout = dropout
    
    def forward(self, data, return_features=False):
        x = data.x  # [num_nodes, in_channels]
        
        # LSTM은 시계열 입력을 기대하므로, 노드 특성을 시퀀스로 변환
        # 각 노드의 특성 벡터를 시퀀스로 처리 (각 특성 차원을 타임스텝으로)
        num_nodes = x.size(0)
        num_features = x.size(1)
        
        # 각 노드를 독립적인 시퀀스로 처리
        # [num_nodes, num_features] -> [num_nodes, num_features, 1] -> [num_nodes, 1, num_features]
        # 또는 각 특성 차원을 타임스텝으로: [num_nodes, num_features] -> [num_nodes, num_features, 1]
        # 여기서는 각 노드의 특성을 시퀀스로 변환
        x = x.unsqueeze(-1)  # [num_nodes, num_features, 1]
        x = x.transpose(1, 2)  # [num_nodes, 1, num_features]
        
        # LSTM 통과
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [num_nodes, 1, hidden_channels * (2 if bidirectional)]
        
        # 마지막 타임스텝의 출력 사용
        x = lstm_out[:, -1, :]  # [num_nodes, hidden_channels * (2 if bidirectional)]
        
        # 배치 정규화
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        final_features = x
        
        # 분류
        logits = self.classifier(x)
        out = F.log_softmax(logits, dim=1)
        
        if return_features:
            return out, final_features
        return out