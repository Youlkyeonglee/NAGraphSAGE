import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import os
import datetime

def create_visualization_dir(epoch, prefix, args, project_name=None):
    """체계적인 시각화 디렉토리 구조 생성"""
    # project_name이 제공되지 않은 경우 기본값 사용
    if project_name is None:
        # 날짜별 폴더 (시분 추가)
        project_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # 프로젝트별 시각화 폴더
    vis_dir = os.path.join('runs', args.project_name, 'train', project_name, 'visualizations')
    print(vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 에포크별 폴더
    exp_name = f"{prefix.replace(' ', '_')}_epoch_{epoch}"
    exp_dir = os.path.join(vis_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    return exp_dir

def save_image_with_description(fig, filepath, description):
    """이미지와 설명 저장"""
    # 이미지 저장
    fig.savefig(filepath)
    plt.close(fig)
    
    # 설명 텍스트 저장
    desc_path = filepath.replace('.png', '_description.txt')
    with open(desc_path, 'w') as f:
        f.write(description)

def visualize_results_static(batch, predictions, epoch, args, project_name=None):
    """
    그래프 구조와 노드 예측을 시각화하는 함수
    """
    # 입력 매개변수 로깅
    print(f"Visualizing for epoch {epoch}, save_dir: {args.project_name}, project_name: {project_name}")
    
    # 시각화 디렉토리 생성 (project_name 전달)
    vis_dir = create_visualization_dir(epoch, "Graph", args, project_name)
    
    # 결과 요약 파일 초기화
    summary_path = os.path.join(vis_dir, "visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"# Graph 시각화 요약 (에포크 {epoch})\n\n")
        f.write(f"생성 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"노드 수: {batch.num_nodes}\n")
        f.write(f"엣지 수: {batch.edge_index.shape[1]}\n\n")
    
    # 1. 예측 클래스 분포 시각화
    fig1 = plt.figure(figsize=(10, 6))
    unique_classes, counts = np.unique(predictions.cpu().numpy(), return_counts=True)
    plt.bar(unique_classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(f'Graph - Class Distribution (Epoch {epoch})')
    
    # 클래스 분포 이미지 및 설명 저장
    class_dist_path = os.path.join(vis_dir, "class_distribution.png")
    class_dist_desc = (
        f"# 클래스 분포 분석\n\n"
        f"이 그래프는 모델의 예측 결과에 따른 클래스 분포를 보여줍니다.\n\n"
        f"## 클래스별 통계:\n"
    )
    
    # 클래스별 통계 추가
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(predictions)) * 100
        class_dist_desc += f"- 클래스 {cls}: {count}개 ({percentage:.2f}%)\n"
    
    class_dist_desc += f"\n## 해석:\n"
    
    # 고르게 분포되어 있는지 또는 불균형이 있는지 분석
    if len(unique_classes) > 1:
        std_dev = np.std(counts)
        mean_count = np.mean(counts)
        cv = std_dev / mean_count if mean_count > 0 else 0
        
        if cv > 0.5:
            class_dist_desc += "클래스 분포가 불균형한 것으로 보입니다. 일부 클래스에 편향된 예측을 하고 있을 수 있습니다.\n"
        else:
            class_dist_desc += "클래스 분포가 비교적 균형을 이루고 있습니다.\n"
        
        max_class = unique_classes[np.argmax(counts)]
        class_dist_desc += f"가장 많이 예측된 클래스는 {max_class}입니다 ({np.max(counts)}개).\n"
    else:
        class_dist_desc += "모든 예측이 단일 클래스로 되어있습니다. 모델이 다양성을 캡처하지 못하고 있을 수 있습니다.\n"
    
    save_image_with_description(fig1, class_dist_path, class_dist_desc)
    
    # 요약 파일 업데이트
    with open(summary_path, 'a') as f:
        f.write(f"## 클래스 분포\n")
        f.write(f"파일: class_distribution.png\n")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / len(predictions)) * 100
            f.write(f"- 클래스 {cls}: {count}개 ({percentage:.2f}%)\n")
        f.write("\n")
    
    # 2. 그래프 네트워크 시각화
    edge_index = batch.edge_index.cpu().numpy()
    
    # 대규모 그래프인 경우 일부만 시각화
    if batch.num_nodes > 100:
        mask = edge_index[0] < 100
        mask &= edge_index[1] < 100
        edge_index = edge_index[:, mask]
    
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가
    for i in range(min(batch.num_nodes, 100)):
        G.add_node(i)
    
    # 엣지 추가
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < 100 and dst < 100:
            G.add_edge(src, dst)
    
    # 노드 색상 설정 (클래스별 색상)
    node_colors = []
    for i in range(min(batch.num_nodes, 100)):
        if i < len(predictions):
            pred_class = predictions[i].item()
            node_colors.append(plt.cm.tab10(pred_class / 10))
        else:
            node_colors.append('gray')
    
    # 그래프 시각화
    fig2 = plt.figure(figsize=(12, 12))
    
    # 그래프 네트워크 설명 준비
    network_desc = (
        f"# 네트워크 구조 시각화\n\n"
        f"이 그래프는 데이터의 네트워크 구조를 시각화합니다. 노드 색상은 예측된 클래스를 나타냅니다.\n\n"
        f"## 네트워크 통계:\n"
        f"- 표시된 노드 수: {len(G.nodes)}\n"
        f"- 표시된 엣지 수: {len(G.edges)}\n"
        f"- 전체 노드 중 표시 비율: {(len(G.nodes) / batch.num_nodes) * 100:.2f}%\n\n"
    )
    
    # 노드가 존재하는 경우 그래프 그리기
    if len(G.nodes) > 0:
        if len(G.nodes) < 500:
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = nx.random_layout(G)
            
        nx.draw(G, pos, 
                node_color=node_colors, 
                with_labels=False, 
                node_size=50, 
                arrowsize=10, 
                width=0.5, 
                alpha=0.7)
        
        plt.title(f'Graph - Network Structure (Epoch {epoch}, {len(G.nodes)} nodes shown)')
        
        # 커뮤니티 구조 분석 추가
        if len(G.nodes) > 10:
            try:
                communities = nx.community.greedy_modularity_communities(G.to_undirected())
                network_desc += f"## 커뮤니티 분석:\n"
                network_desc += f"- 감지된 커뮤니티 수: {len(communities)}\n"
                
                if len(communities) > 1:
                    network_desc += "- 커뮤니티 구조가 감지되었습니다. 데이터에 클러스터 패턴이 존재할 수 있습니다.\n"
                else:
                    network_desc += "- 뚜렷한 커뮤니티 구조가 감지되지 않았습니다.\n"
            except:
                network_desc += "- 커뮤니티 구조 분석을 수행할 수 없습니다.\n"
                
        # 중심성 분석 추가
        if len(G.nodes) > 1:
            try:
                centrality = nx.degree_centrality(G)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                network_desc += f"\n## 중심성 분석:\n"
                network_desc += f"- 가장 중요한 노드 (degree centrality 기준):\n"
                
                for node, cent in top_nodes:
                    if node < len(predictions):
                        node_class = predictions[node].item()
                        network_desc += f"  - 노드 {node}: 중심성 {cent:.4f}, 클래스 {node_class}\n"
            except:
                network_desc += "\n- 중심성 분석을 수행할 수 없습니다.\n"
    else:
        plt.title(f'Graph - No Nodes to Visualize (Epoch {epoch})')
        network_desc += "\n## 해석:\n시각화할 노드가 없습니다. 데이터셋이 비어있거나 max_nodes 값이 너무 작게 설정되었을 수 있습니다."
        
    # 네트워크 이미지 및 설명 저장
    network_path = os.path.join(vis_dir, "network_structure.png")
    save_image_with_description(fig2, network_path, network_desc)
    
    # 요약 파일 업데이트
    with open(summary_path, 'a') as f:
        f.write(f"## 네트워크 구조\n")
        f.write(f"파일: network_structure.png\n")
        f.write(f"표시된 노드: {len(G.nodes)} / {batch.num_nodes}\n")
        f.write(f"표시된 엣지: {len(G.edges)} / {batch.edge_index.shape[1]}\n\n")
    
    # 3. 클래스별 정확도 시각화 (라벨이 있는 경우)
    if hasattr(batch, 'y') and batch.y is not None:
        y_true = batch.y.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # 클래스별 정확도 계산
        class_accuracies = {}
        for cls in np.unique(y_true):
            cls_mask = (y_true == cls)
            if np.sum(cls_mask) > 0:
                cls_acc = np.mean(y_pred[cls_mask] == cls)
                class_accuracies[int(cls)] = float(cls_acc)
        
        # 클래스별 정확도 시각화
        if class_accuracies:
            fig3 = plt.figure(figsize=(10, 6))
            classes = list(class_accuracies.keys())
            accs = list(class_accuracies.values())
            plt.bar(classes, accs)
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.title(f'Graph - Class-wise Accuracy (Epoch {epoch})')
            
            # 정확도 설명 준비
            accuracy_desc = (
                f"# 클래스별 정확도 분석\n\n"
                f"이 그래프는 각 클래스에 대한 예측 정확도를 보여줍니다.\n\n"
                f"## 클래스별 정확도:\n"
            )
            
            # 클래스별 정확도 추가
            for cls, acc in zip(classes, accs):
                cls_count = np.sum(y_true == cls)
                accuracy_desc += f"- 클래스 {cls}: {acc:.4f} (샘플 수: {cls_count})\n"
            
            # 정확도 해석 추가
            accuracy_desc += f"\n## 해석:\n"
            
            avg_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            accuracy_desc += f"- 평균 정확도: {avg_acc:.4f}\n"
            accuracy_desc += f"- 정확도 표준편차: {std_acc:.4f}\n\n"
            
            if std_acc > 0.2 and len(accs) > 1:
                accuracy_desc += "클래스 간 정확도 차이가 큽니다. 일부 클래스에 대해 성능이 크게 좋거나 나쁠 수 있습니다.\n"
            elif len(accs) > 1:
                accuracy_desc += "클래스 간 정확도가 비교적 균등합니다.\n"
                
            max_cls = classes[np.argmax(accs)]
            min_cls = classes[np.argmin(accs)]
            
            accuracy_desc += f"- 가장 높은 정확도: 클래스 {max_cls} ({np.max(accs):.4f})\n"
            accuracy_desc += f"- 가장 낮은 정확도: 클래스 {min_cls} ({np.min(accs):.4f})\n"
            
            accuracy_path = os.path.join(vis_dir, "class_accuracy.png")
            save_image_with_description(fig3, accuracy_path, accuracy_desc)
            
            # 요약 파일 업데이트
            with open(summary_path, 'a') as f:
                f.write(f"## 클래스별 정확도\n")
                f.write(f"파일: class_accuracy.png\n")
                for cls, acc in zip(classes, accs):
                    f.write(f"- 클래스 {cls}: {acc:.4f}\n")
                f.write(f"- 평균 정확도: {avg_acc:.4f}\n\n")
    
    # 전체 요약 마무리
    with open(summary_path, 'a') as f:
        f.write(f"\n# 종합 분석\n\n")
        
        # 전반적인 성능 분석
        if hasattr(batch, 'y') and batch.y is not None:
            overall_acc = np.mean(y_pred == y_true)
            f.write(f"- 전체 정확도: {overall_acc:.4f}\n")
            
            if overall_acc > 0.8:
                f.write("- 모델이 매우 좋은 성능을 보이고 있습니다.\n")
            elif overall_acc > 0.6:
                f.write("- 모델이 적절한 성능을 보이고 있습니다.\n")
            else:
                f.write("- 모델 성능이 개선이 필요합니다.\n")
        
        # 클래스 분포 불균형 분석
        if len(unique_classes) > 1:
            if np.std(counts) / np.mean(counts) > 0.5:
                f.write("- 예측 클래스 분포가 불균형합니다. 데이터 불균형 문제를 확인해 보세요.\n")
        else:
            f.write("- 모든 예측이 단일 클래스입니다. 모델이 다양성을 캡처하지 못하고 있을 수 있습니다.\n")
    
    print(f"Visualization results saved to '{vis_dir}' directory")
    print(f"Summary report: {summary_path}")
    
    # 이 부분이 중요: 파일로 저장하는 기능 추가
    if vis_dir:
        # 여러 개의 그래프를 저장할 경우 이름을 다르게 해줍니다
        for i, fig in enumerate(plt.get_fignums()):
            figure = plt.figure(fig)
            filename = os.path.join(vis_dir, f'graph_viz_epoch{epoch}_fig{i}.png')
            print(f"Saving figure to: {filename}")
            figure.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close('all')  # 모든 그림 닫기
    else:
        plt.show()
        plt.close('all')
    
    return vis_dir