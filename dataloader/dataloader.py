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
    Pickle 파일에서 PyTorch Geometric 데이터를 직접 로드하는 함수
    
    Args:
        pkl_path (str): Pickle 파일 경로
        
    Returns:
        Data: PyTorch Geometric Data 객체
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

class GraphDataset(Dataset):
    """그래프 데이터셋 클래스"""
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
    캐시 파일 경로를 생성합니다.
    
    Args:
        pkl_path (str): 원본 피클 파일 경로
        train_ratio (float): 훈련 데이터 비율
        val_ratio (float): 검증 데이터 비율
        cache_dir (str): 캐시 디렉토리 경로
        balance_classes (bool): 클래스 불균형 조절 여부
        max_samples_per_class (int): 클래스당 최대 샘플 수
        class_balance_ratio (float): 클래스 2의 샘플 비율
        samples_per_class_dict (dict): 각 클래스별 샘플 수 딕셔너리
        normalize_coords (bool): 좌표 정규화 여부
        coord_scale_factor (float): 좌표 스케일 팩터
        
    Returns:
        tuple: 훈련, 검증, 테스트 캐시 파일 경로
    """
    # 원본 파일명 추출
    base_filename = os.path.basename(pkl_path).split('.')[0]
    
    # samples_per_class_dict를 문자열로 변환 (해시 생성용)
    samples_dict_str = "None"
    if samples_per_class_dict is not None:
        # 딕셔너리를 정렬된 문자열로 변환 (예: "0:1000,1:2000,2:1500")
        samples_dict_str = ",".join([f"{k}:{v}" for k, v in sorted(samples_per_class_dict.items())])
    
    # 파라미터 해시 생성 (모든 파라미터 포함, samples_per_class_dict 포함)
    params_str = f"{pkl_path}_{train_ratio}_{val_ratio}_{balance_classes}_{max_samples_per_class}_{class_balance_ratio}_{samples_dict_str}_{normalize_coords}_{coord_scale_factor}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:10]
    
    # 디버그: samples_per_class_dict가 설정된 경우 해시 정보 출력
    if samples_per_class_dict is not None:
        print(f"캐시 해시 생성 정보: samples_per_class_dict={samples_per_class_dict}, 해시={params_hash}")
    
    # 캐시 디렉토리 생성
    os.makedirs(cache_dir, exist_ok=True)
    
    # 캐시 파일 경로 생성
    train_cache = os.path.join(cache_dir, f"{base_filename}_train_{params_hash}.pkl")
    val_cache = os.path.join(cache_dir, f"{base_filename}_val_{params_hash}.pkl")
    test_cache = os.path.join(cache_dir, f"{base_filename}_test_{params_hash}.pkl")
    
    return train_cache, val_cache, test_cache

def normalize_coordinates(data, normalize_coords=True, coord_scale_factor=0.001):
    """
    좌표 정보를 정규화하는 함수
    
    Args:
        data: PyTorch Geometric Data 객체
        normalize_coords: 좌표 정규화 여부
        coord_scale_factor: 좌표 스케일 팩터 (4K 좌표를 일반 좌표로 변환)
    
    Returns:
        Data: 좌표가 정규화된 Data 객체
    """
    if not normalize_coords:
        return data
    
    print(f"좌표 정규화 적용 중... (스케일 팩터: {coord_scale_factor})")
    
    # 무한대 값과 NaN 값 처리
    data.x[torch.isinf(data.x)] = 0
    data.x[torch.isnan(data.x)] = 0
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr[torch.isinf(data.edge_attr)] = 0
        data.edge_attr[torch.isnan(data.edge_attr)] = 0
    
    # 좌표 특성 (보통 0, 1번 특성) 정규화
    if data.x.shape[1] >= 2:
        # X, Y 좌표를 스케일 팩터로 나누어 정규화
        data.x[:, 0] = data.x[:, 0] * coord_scale_factor  # X 좌표
        data.x[:, 1] = data.x[:, 1] * coord_scale_factor  # Y 좌표
        
        # 엣지 속성도 정규화 (좌표 관련 속성이 있다면)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[1] >= 2:
            data.edge_attr[:, 0] = data.edge_attr[:, 0] * coord_scale_factor
            data.edge_attr[:, 1] = data.edge_attr[:, 1] * coord_scale_factor
    
    # 전체 특성에 대한 StandardScaler 적용
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # 노드 특성 정규화
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
    
    # 엣지 특성 정규화
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = torch.tensor(scaler.fit_transform(data.edge_attr.numpy()), dtype=torch.float32)
    
    print(f"좌표 정규화 완료")
    print(f"  노드 특성 범위: min={data.x.min().item():.4f}, max={data.x.max().item():.4f}")
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"  엣지 특성 범위: min={data.edge_attr.min().item():.4f}, max={data.edge_attr.max().item():.4f}")
    
    return data

def balance_class_samples(data, balance_classes=False, max_samples_per_class=None, class_balance_ratio=1.0, samples_per_class_dict=None):
    """
    클래스 불균형을 조절하는 함수
    
    Args:
        data: PyTorch Geometric Data 객체
        balance_classes: 클래스 불균형 조절 여부
        max_samples_per_class: 클래스당 최대 샘플 수 (모든 클래스에 동일하게 적용)
        class_balance_ratio: 클래스 2의 샘플 비율 (1.0이면 원본, 0.5면 절반)
        samples_per_class_dict: 각 클래스별 샘플 수 딕셔너리 (예: {0: 1000, 1: 2000, 2: 1500})
                              이 파라미터가 설정되면 max_samples_per_class보다 우선 적용됨
    
    Returns:
        Data: 클래스가 조절된 Data 객체
    """
    if not balance_classes:
        return data
    
    # 클래스별 인덱스 수집
    unique_labels = torch.unique(data.y)
    class_indices = {}
    
    for label in unique_labels:
        class_indices[label.item()] = torch.where(data.y == label)[0]
    
    # 클래스별 샘플 수 계산
    class_counts = {label: len(indices) for label, indices in class_indices.items()}
    print(f"원본 클래스 분포: {class_counts}")
    
    # 클래스 2의 샘플 수 조절 (samples_per_class_dict가 없을 때만)
    if samples_per_class_dict is None and class_balance_ratio < 1.0 and 2 in class_indices:
        target_count_class2 = int(len(class_indices[2]) * class_balance_ratio)
        if target_count_class2 < len(class_indices[2]):
            # 클래스 2에서 랜덤 샘플링
            selected_indices_class2 = torch.randperm(len(class_indices[2]))[:target_count_class2]
            class_indices[2] = class_indices[2][selected_indices_class2]
            print(f"클래스 2 샘플 수 조절: {class_counts[2]} → {len(class_indices[2])}")
    
    # 각 클래스별로 다른 샘플 수 적용 (우선순위 1)
    if samples_per_class_dict is not None:
        for label in class_indices:
            if label in samples_per_class_dict:
                target_count = samples_per_class_dict[label]
                if target_count < len(class_indices[label]):
                    # 지정된 샘플 수로 랜덤 샘플링
                    selected_indices = torch.randperm(len(class_indices[label]))[:target_count]
                    class_indices[label] = class_indices[label][selected_indices]
                    print(f"클래스 {label} 샘플 수 조절: {class_counts[label]} → {len(class_indices[label])} (지정값: {target_count})")
                elif target_count > len(class_indices[label]):
                    print(f"클래스 {label}: 요청된 샘플 수({target_count})가 원본 샘플 수({len(class_indices[label])})보다 큽니다. 원본 유지.")
    
    # 최대 샘플 수 제한 (samples_per_class_dict가 없을 때만 적용)
    elif max_samples_per_class is not None:
        for label in class_indices:
            if len(class_indices[label]) > max_samples_per_class:
                selected_indices = torch.randperm(len(class_indices[label]))[:max_samples_per_class]
                class_indices[label] = class_indices[label][selected_indices]
                print(f"클래스 {label} 샘플 수 제한: {class_counts[label]} → {len(class_indices[label])}")
    
    # 선택된 인덱스 결합
    selected_indices = torch.cat([indices for indices in class_indices.values()])
    selected_indices = torch.sort(selected_indices)[0]  # 정렬
    
    # 새로운 클래스 분포 출력
    new_class_counts = {}
    for label in unique_labels:
        new_class_counts[label.item()] = (data.y[selected_indices] == label).sum().item()
    print(f"조절된 클래스 분포: {new_class_counts}")
    
    # 서브그래프 생성
    return create_subgraph_from_indices(data, selected_indices)

def create_subgraph_from_indices(data, node_indices):
    """
    선택된 노드 인덱스로 서브그래프를 생성하는 함수
    """
    # 노드 인덱스 집합으로 변환
    node_set = set(node_indices.tolist())
    
    # 엣지 필터링 (양쪽 노드가 모두 선택된 노드 집합에 있는 엣지만 유지)
    edge_mask = []
    
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if src in node_set and dst in node_set:
            edge_mask.append(i)
            
    # 새로운 엣지 인덱스와 특성 생성
    if len(edge_mask) > 0:
        edge_mask = torch.tensor(edge_mask, dtype=torch.long)
        filtered_edge_index = data.edge_index[:, edge_mask]
        
        # 노드 인덱스 재매핑
        node_idx_map = {idx.item(): i for i, idx in enumerate(node_indices)}
        remapped_edge_index = torch.zeros_like(filtered_edge_index)
        for i in range(filtered_edge_index.size(1)):
            remapped_edge_index[0, i] = node_idx_map[filtered_edge_index[0, i].item()]
            remapped_edge_index[1, i] = node_idx_map[filtered_edge_index[1, i].item()]
        
        # 엣지 특성이 있다면 필터링
        filtered_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    else:
        # 엣지가 없는 경우 빈 텐서 생성
        remapped_edge_index = torch.zeros((2, 0), dtype=torch.long)
        filtered_edge_attr = torch.zeros((0, data.edge_attr.size(1))) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    
    # 노드 특성과 레이블 추출
    node_x = data.x[node_indices]
    node_y = data.y[node_indices] if hasattr(data, 'y') and data.y is not None else None
    
    # 새로운 Data 객체 생성
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
    데이터 로더를 생성하는 함수
    
    Args:
        pkl_path: Pickle 파일 경로
        batch_size: 배치 크기
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        use_cache: 캐시 사용 여부
        cache_dir: 캐시 디렉토리
        num_workers: 워커 수
        balance_classes: 클래스 불균형 조절 여부
        max_samples_per_class: 클래스당 최대 샘플 수
        class_balance_ratio: 클래스 2의 샘플 비율
        normalize_coords: 좌표 정규화 여부
        coord_scale_factor: 좌표 스케일 팩터
    """
    # 캐시 파일 경로 가져오기 (정규화 파라미터 포함)
    train_cache, val_cache, test_cache = get_cache_path(
        pkl_path, train_ratio, val_ratio, cache_dir,
        balance_classes, max_samples_per_class, class_balance_ratio,
        samples_per_class_dict, normalize_coords, coord_scale_factor
    )
    
    # 캐시 파일이 모두 존재하고, use_cache가 True인 경우 캐시에서 로드
    if use_cache and os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
        print(f"캐시 파일을 사용하여 데이터 로드 중...")
        print(f"  캐시 파일 경로: {train_cache}")
        if samples_per_class_dict is not None:
            print(f"  ⚠️  주의: samples_per_class_dict가 설정되었지만 이전 캐시를 사용 중입니다.")
            print(f"  ⚠️  새로운 샘플링 설정을 적용하려면 캐시 파일을 삭제하거나 use_cache=False로 설정하세요.")
        
        start_time = time.time()
        
        # 캐시에서 서브그래프 로드
        with open(train_cache, 'rb') as f:
            train_subgraph = pickle.load(f)
        with open(val_cache, 'rb') as f:
            val_subgraph = pickle.load(f)
        with open(test_cache, 'rb') as f:
            test_subgraph = pickle.load(f)
            
        print(f"캐시에서 데이터 로드 완료 (소요 시간: {time.time() - start_time:.2f}초)")
    else:
        print(f"원본 데이터에서 로드 중... (이후 캐시에 저장됩니다)")
        start_time = time.time()
        
        # Pickle 파일에서 데이터 로드
        pyg_data = load_pyg_data_from_pickle(pkl_path)
        
        # 좌표 정규화 적용
        if normalize_coords:
            pyg_data = normalize_coordinates(pyg_data, normalize_coords, coord_scale_factor)
        
        # 클래스 불균형 조절 적용
        if balance_classes:
            print("원본 데이터에 클래스 불균형 조절 적용 중...")
            pyg_data = balance_class_samples(
                pyg_data, 
                balance_classes=balance_classes,
                max_samples_per_class=max_samples_per_class,
                class_balance_ratio=class_balance_ratio,
                samples_per_class_dict=samples_per_class_dict
            )
        
        # 데이터 인덱스 생성 및 분할
        indices = list(range(pyg_data.num_nodes))
        
        # train, temp 분할 (train_ratio)
        train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
        
        # temp에서 validation, test 분할
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio_adjusted, random_state=42)
        
        # 서브그래프 생성 함수
        def create_subgraph(data, node_indices):
            # 노드 인덱스 집합으로 변환
            node_set = set(node_indices)
            
            # 엣지 필터링 (양쪽 노드가 모두 선택된 노드 집합에 있는 엣지만 유지)
            edge_mask = []
            
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
                if src in node_set and dst in node_set:
                    edge_mask.append(i)
                    
            # 새로운 엣지 인덱스와 특성 생성
            if len(edge_mask) > 0:
                edge_mask = torch.tensor(edge_mask, dtype=torch.long)
                filtered_edge_index = data.edge_index[:, edge_mask]
                
                # 노드 인덱스 재매핑
                node_idx_map = {idx: i for i, idx in enumerate(node_indices)}
                remapped_edge_index = torch.zeros_like(filtered_edge_index)
                for i in range(filtered_edge_index.size(1)):
                    remapped_edge_index[0, i] = node_idx_map[filtered_edge_index[0, i].item()]
                    remapped_edge_index[1, i] = node_idx_map[filtered_edge_index[1, i].item()]
                
                # 엣지 특성이 있다면 필터링
                filtered_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            else:
                # 엣지가 없는 경우 빈 텐서 생성
                remapped_edge_index = torch.zeros((2, 0), dtype=torch.long)
                filtered_edge_attr = torch.zeros((0, data.edge_attr.size(1))) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            
            # 노드 특성과 레이블 추출
            node_x = data.x[node_indices]
            node_y = data.y[node_indices] if hasattr(data, 'y') and data.y is not None else None
            
            # 새로운 Data 객체 생성
            subgraph = Data(
                x=node_x,
                edge_index=remapped_edge_index,
                edge_attr=filtered_edge_attr,
                y=node_y
            )
            
            return subgraph
        
        # 각 분할에 대한 서브그래프 생성
        print("서브그래프 생성 중...")
        train_subgraph = create_subgraph(pyg_data, train_idx)
        val_subgraph = create_subgraph(pyg_data, val_idx)
        test_subgraph = create_subgraph(pyg_data, test_idx)
        
        # 캐시 저장
        if use_cache:
            print("데이터를 캐시에 저장 중...")
            os.makedirs(os.path.dirname(train_cache), exist_ok=True)
            with open(train_cache, 'wb') as f:
                pickle.dump(train_subgraph, f)
            with open(val_cache, 'wb') as f:
                pickle.dump(val_subgraph, f)
            with open(test_cache, 'wb') as f:
                pickle.dump(test_subgraph, f)
            print(f"캐시에 데이터 저장 완료 (캐시 경로: {cache_dir})")
            
        print(f"데이터 처리 완료 (소요 시간: {time.time() - start_time:.2f}초)")
    
    # 서브그래프 정보 출력
    print(f"Train 서브그래프: {train_subgraph.num_nodes} 노드, {train_subgraph.num_edges} 엣지")
    print(f"Validation 서브그래프: {val_subgraph.num_nodes} 노드, {val_subgraph.num_edges} 엣지")
    print(f"Test 서브그래프: {test_subgraph.num_nodes} 노드, {test_subgraph.num_edges} 엣지")
    
    # GraphDataset 생성
    train_dataset = GraphDataset([train_subgraph])
    val_dataset = GraphDataset([val_subgraph])
    test_dataset = GraphDataset([test_subgraph])
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# Example usage
# json_path = "/path/to/your/json_file.json"
# create_data_loaders(json_path, batch_size=32)