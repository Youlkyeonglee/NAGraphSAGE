import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import os
import datetime

def create_visualization_dir(epoch, prefix, args, project_name=None):
    """Create systematic visualization directory structure"""
    # Use default value if project_name is not provided
    if project_name is None:
        # Date folder (add hour and minute)
        project_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # Project-specific visualization folder
    vis_dir = os.path.join('runs', args.project_name, 'train', project_name, 'visualizations')
    print(vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Epoch-specific folder
    exp_name = f"{prefix.replace(' ', '_')}_epoch_{epoch}"
    exp_dir = os.path.join(vis_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    return exp_dir

def save_image_with_description(fig, filepath, description):
    """Save image and description"""
    # Save image
    fig.savefig(filepath)
    plt.close(fig)
    
    # Save description text
    desc_path = filepath.replace('.png', '_description.txt')
    with open(desc_path, 'w') as f:
        f.write(description)

def visualize_results_static(batch, predictions, epoch, args, project_name=None):
    """
    Function to visualize graph structure and node predictions
    """
    # Log input parameters
    print(f"Visualizing for epoch {epoch}, save_dir: {args.project_name}, project_name: {project_name}")
    
    # Create visualization directory (pass project_name)
    vis_dir = create_visualization_dir(epoch, "Graph", args, project_name)
    
    # Initialize result summary file
    summary_path = os.path.join(vis_dir, "visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"# Graph Visualization Summary (Epoch {epoch})\n\n")
        f.write(f"Creation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Nodes: {batch.num_nodes}\n")
        f.write(f"Number of Edges: {batch.edge_index.shape[1]}\n\n")
    
    # 1. Visualize Predicted Class Distribution
    fig1 = plt.figure(figsize=(10, 6))
    unique_classes, counts = np.unique(predictions.cpu().numpy(), return_counts=True)
    plt.bar(unique_classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(f'Graph - Class Distribution (Epoch {epoch})')
    
    # Save class distribution image and description
    class_dist_path = os.path.join(vis_dir, "class_distribution.png")
    class_dist_desc = (
        f"# Class Distribution Analysis\n\n"
        f"This graph shows the class distribution according to the model's prediction results.\n\n"
        f"## Statistics per Class:\n"
    )
    
    # Add statistics per class
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(predictions)) * 100
        class_dist_desc += f"- Class {cls}: {count} items ({percentage:.2f}%)\n"
    
    class_dist_desc += f"\n## Interpretation:\n"
    
    # Analyze whether it is evenly distributed or imbalanced
    if len(unique_classes) > 1:
        std_dev = np.std(counts)
        mean_count = np.mean(counts)
        cv = std_dev / mean_count if mean_count > 0 else 0
        
        if cv > 0.5:
            class_dist_desc += "The class distribution appears to be imbalanced. The model may be making biased predictions towards some classes.\n"
        else:
            class_dist_desc += "The class distribution is relatively balanced.\n"
        
        max_class = unique_classes[np.argmax(counts)]
        class_dist_desc += f"The most predicted class is {max_class} ({np.max(counts)} items).\n"
    else:
        class_dist_desc += "All predictions are for a single class. The model may not be capturing diversity.\n"
    
    save_image_with_description(fig1, class_dist_path, class_dist_desc)
    
    # Update summary file
    with open(summary_path, 'a') as f:
        f.write(f"## Class Distribution\n")
        f.write(f"File: class_distribution.png\n")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / len(predictions)) * 100
            f.write(f"- Class {cls}: {count} items ({percentage:.2f}%)\n")
        f.write("\n")
    
    # 2. Graph Network Visualization
    edge_index = batch.edge_index.cpu().numpy()
    
    # Visualize only a part if it is a large scale graph
    if batch.num_nodes > 100:
        mask = edge_index[0] < 100
        mask &= edge_index[1] < 100
        edge_index = edge_index[:, mask]
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(min(batch.num_nodes, 100)):
        G.add_node(i)
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < 100 and dst < 100:
            G.add_edge(src, dst)
    
    # Set node colors (colors per class)
    node_colors = []
    for i in range(min(batch.num_nodes, 100)):
        if i < len(predictions):
            pred_class = predictions[i].item()
            node_colors.append(plt.cm.tab10(pred_class / 10))
        else:
            node_colors.append('gray')
    
    # Visualize graph
    fig2 = plt.figure(figsize=(12, 12))
    
    # Prepare graph network description
    network_desc = (
        f"# Network Structure Visualization\n\n"
        f"This graph visualizes the network structure of the data. Node colors represent the predicted classes.\n\n"
        f"## Network Statistics:\n"
        f"- Number of nodes shown: {len(G.nodes)}\n"
        f"- Number of edges shown: {len(G.edges)}\n"
        f"- Display ratio among total nodes: {(len(G.nodes) / batch.num_nodes) * 100:.2f}%\n\n"
    )
    
    # Draw graph if nodes exist
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
        
        # Add community structure analysis
        if len(G.nodes) > 10:
            try:
                communities = nx.community.greedy_modularity_communities(G.to_undirected())
                network_desc += f"## Community Analysis:\n"
                network_desc += f"- Number of detected communities: {len(communities)}\n"
                
                if len(communities) > 1:
                    network_desc += "- Community structure detected. Cluster patterns may exist in the data.\n"
                else:
                    network_desc += "- No distinct community structure detected.\n"
            except:
                network_desc += "- Cannot perform community structure analysis.\n"
                
        # Add centrality analysis
        if len(G.nodes) > 1:
            try:
                centrality = nx.degree_centrality(G)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                network_desc += f"\n## Centrality Analysis:\n"
                network_desc += f"- Most important nodes (based on degree centrality):\n"
                
                for node, cent in top_nodes:
                    if node < len(predictions):
                        node_class = predictions[node].item()
                        network_desc += f"  - Node {node}: Centrality {cent:.4f}, Class {node_class}\n"
            except:
                network_desc += "\n- Cannot perform centrality analysis.\n"
    else:
        plt.title(f'Graph - No Nodes to Visualize (Epoch {epoch})')
        network_desc += "\n## Interpretation:\nNo nodes to visualize. The dataset might be empty or max_nodes value might be set too small."
        
    # Save network image and description
    network_path = os.path.join(vis_dir, "network_structure.png")
    save_image_with_description(fig2, network_path, network_desc)
    
    # Update summary file
    with open(summary_path, 'a') as f:
        f.write(f"## Network Structure\n")
        f.write(f"File: network_structure.png\n")
        f.write(f"Nodes shown: {len(G.nodes)} / {batch.num_nodes}\n")
        f.write(f"Edges shown: {len(G.edges)} / {batch.edge_index.shape[1]}\n\n")
    
    # 3. Visualize Accuracy per Class (if labels exist)
    if hasattr(batch, 'y') and batch.y is not None:
        y_true = batch.y.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # Calculate accuracy per class
        class_accuracies = {}
        for cls in np.unique(y_true):
            cls_mask = (y_true == cls)
            if np.sum(cls_mask) > 0:
                cls_acc = np.mean(y_pred[cls_mask] == cls)
                class_accuracies[int(cls)] = float(cls_acc)
        
        # Visualize accuracy per class
        if class_accuracies:
            fig3 = plt.figure(figsize=(10, 6))
            classes = list(class_accuracies.keys())
            accs = list(class_accuracies.values())
            plt.bar(classes, accs)
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.title(f'Graph - Class-wise Accuracy (Epoch {epoch})')
            
            # Prepare accuracy description
            accuracy_desc = (
                f"# Class-wise Accuracy Analysis\n\n"
                f"This graph shows the prediction accuracy for each class.\n\n"
                f"## Accuracy per Class:\n"
            )
            
            # Add accuracy per class
            for cls, acc in zip(classes, accs):
                cls_count = np.sum(y_true == cls)
                accuracy_desc += f"- Class {cls}: {acc:.4f} (Sample count: {cls_count})\n"
            
            # Add accuracy interpretation
            accuracy_desc += f"\n## Interpretation:\n"
            
            avg_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            accuracy_desc += f"- Average Accuracy: {avg_acc:.4f}\n"
            accuracy_desc += f"- Accuracy Standard Deviation: {std_acc:.4f}\n\n"
            
            if std_acc > 0.2 and len(accs) > 1:
                accuracy_desc += "Large difference in accuracy between classes. Performance may be significantly better or worse for some classes.\n"
            elif len(accs) > 1:
                accuracy_desc += "Accuracy between classes is relatively even.\n"
                
            max_cls = classes[np.argmax(accs)]
            min_cls = classes[np.argmin(accs)]
            
            accuracy_desc += f"- Highest Accuracy: Class {max_cls} ({np.max(accs):.4f})\n"
            accuracy_desc += f"- Lowest Accuracy: Class {min_cls} ({np.min(accs):.4f})\n"
            
            accuracy_path = os.path.join(vis_dir, "class_accuracy.png")
            save_image_with_description(fig3, accuracy_path, accuracy_desc)
            
            # Update summary file
            with open(summary_path, 'a') as f:
                f.write(f"## Accuracy per Class\n")
                f.write(f"File: class_accuracy.png\n")
                for cls, acc in zip(classes, accs):
                    f.write(f"- Class {cls}: {acc:.4f}\n")
                f.write(f"- Average Accuracy: {avg_acc:.4f}\n\n")
    
    # Finalize overall summary
    with open(summary_path, 'a') as f:
        f.write(f"\n# Comprehensive Analysis\n\n")
        
        # Overall Performance Analysis
        if hasattr(batch, 'y') and batch.y is not None:
            overall_acc = np.mean(y_pred == y_true)
            f.write(f"- Overall Accuracy: {overall_acc:.4f}\n")
            
            if overall_acc > 0.8:
                f.write("- The model is showing very good performance.\n")
            elif overall_acc > 0.6:
                f.write("- The model is showing adequate performance.\n")
            else:
                f.write("- Model performance needs improvement.\n")
        
        # Class Distribution Imbalance Analysis
        if len(unique_classes) > 1:
            if np.std(counts) / np.mean(counts) > 0.5:
                f.write("- Predicted class distribution is imbalanced. Check for data imbalance issues.\n")
        else:
            f.write("- All predictions are for a single class. The model may not be capturing diversity.\n")
    
    print(f"Visualization results saved to '{vis_dir}' directory")
    print(f"Summary report: {summary_path}")
    
    # Important: Add function to save to file
    if vis_dir:
        # Use different names when saving multiple graphs
        for i, fig in enumerate(plt.get_fignums()):
            figure = plt.figure(fig)
            filename = os.path.join(vis_dir, f'graph_viz_epoch{epoch}_fig{i}.png')
            print(f"Saving figure to: {filename}")
            figure.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close('all')  # Close all figures
    else:
        plt.show()
        plt.close('all')
    
    return vis_dir