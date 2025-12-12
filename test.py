import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, precision_recall_curve, average_precision_score
from tqdm import tqdm
import yaml
import traceback

from dataloader.dataloader import create_data_loaders
from models.model import NeighborAwareGraphSAGE

def setup_device():
    """Set up the device for evaluation."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_model(args, device):
    """Load the model architecture and weights."""
    print(f"Loading model from {args.model_path}...")
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    # Try to load training parameters if available to infer model config
    train_dir = os.path.dirname(args.model_path)
    params_file = os.path.join(train_dir, 'training_params.yaml')
    
    if os.path.exists(params_file):
        print(f"Found training params at {params_file}. Loading config...")
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)

        # to prioritize the yaml file.
        # Example: args.hidden_channels = params['model']['hidden_channels']
        
    # Initialize model
    # Note: We need to know input_dim (node features) and num_classes

    return None # We will create model after data loading

def evaluate(model, test_loader, device, args, save_dir):
    """Evaluate the model on the test dataset."""
    model.eval()
    test_preds = []
    test_labels = []
    test_pred_proba = []
    test_correct = 0
    test_total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            
            test_correct += (pred == batch.y).sum().item()
            test_total += batch.y.size(0)
            
            test_preds.extend(pred.cpu().tolist())
            test_labels.extend(batch.y.cpu().tolist())
            test_pred_proba.append(torch.softmax(out, dim=1).cpu())
            
    test_acc = test_correct / test_total if test_total > 0 else 0
    test_pred_proba = torch.cat(test_pred_proba, dim=0).numpy()
    test_labels_np = np.array(test_labels)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = test_acc
    
    if len(test_preds) > 0:
        metrics['macro_f1'] = f1_score(test_labels, test_preds, average='macro')
        metrics['micro_f1'] = f1_score(test_labels, test_preds, average='micro')
        
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, test_preds, average=None
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_per_class'] = f1
        metrics['support'] = support
        
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Micro F1: {metrics['micro_f1']:.4f}")
    
    # Save results
    save_results(metrics, test_labels, test_preds, test_pred_proba, save_dir)
    
    return metrics

def save_results(metrics, y_true, y_pred, y_prob, save_dir):
    """Save evaluation results and visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save text summary
    results_file = os.path.join(save_dir, 'test_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Test Evaluation Results\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics.get('macro_f1', 0):.4f}\n")
        f.write(f"Micro F1: {metrics.get('micro_f1', 0):.4f}\n")
        f.write(f"Macro Precision: {metrics.get('macro_precision', 0):.4f}\n")
        f.write(f"Macro Recall: {metrics.get('macro_recall', 0):.4f}\n\n")
        
        if 'f1_per_class' in metrics:
            f.write("Per-Class Performance:\n")
            for i in range(len(metrics['f1_per_class'])):
                f.write(f"  Class {i}: Precision={metrics['precision'][i]:.4f}, "
                        f"Recall={metrics['recall'][i]:.4f}, "
                        f"F1={metrics['f1_per_class'][i]:.4f}, "
                        f"Support={metrics['support'][i]}\n")
    
    print(f"Results saved to {results_file}")
    
    # 2. Confusion Matrix
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved.")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
        
    # 3. Precision-Recall Curve
    try:
        plt.figure(figsize=(10, 8))
        n_classes = y_prob.shape[1]
        for i in range(n_classes):
            p, r, _ = precision_recall_curve((np.array(y_true) == i).astype(int), y_prob[:, i])
            ap = average_precision_score((np.array(y_true) == i).astype(int), y_prob[:, i])
            plt.plot(r, p, label=f'Class {i} (AP = {ap:.2f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Precision-Recall curve saved.")
    except Exception as e:
        print(f"Error saving PR curve: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GNN model')
    
    # Required argument: Path to the model checkpoint
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint (.pth)')
    
    # Data arguments (should match training)
    parser.add_argument('--dataset', type=str, default='gongeoptap', choices=['gongeoptap', 'DRIFT'])
    parser.add_argument('--graph_data_type', type=str, default='image', help='image, world')
    parser.add_argument('--num_vehicles', type=int, default=4)
    parser.add_argument('--edge_dim', type=int, default=7)
    parser.add_argument('--site', type=str, default='')
    
    # Model arguments (should match training)
    parser.add_argument('--model_type', type=str, default='NeighborAwareGraphSAGE')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--attention_type', type=str, default='mlp')
    
    # Execution arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results. Defaults to model directory.')
    
    # Data loading specific args
    parser.add_argument('--pkl_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Set paths if not provided
    if args.pkl_path is None:
        if args.dataset == 'gongeoptap':
            if args.graph_data_type == 'image':
                args.pkl_path = f'./4_20250814graph_data/graph_data/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}.pkl'
                args.cache_dir = f'./4_20250814graph_data/graph_data/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}'
            elif args.graph_data_type == 'world':
                args.pkl_path = f'./4_20250814graph_data/world_graph_data/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}.pkl'
                args.cache_dir = f'./4_20250814graph_data/world_graph_data/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}'
        else:
            if args.graph_data_type == 'image':
                args.pkl_path = f'./5_20250909graph_data/{args.site}/graphdata/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}.pkl'
                args.cache_dir = f'./5_20250909graph_data/{args.site}/graphdata/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}'
            elif args.graph_data_type == 'world':
                args.pkl_path = f'./5_20250909graph_data/{args.site}/world_graphdata/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}.pkl'
                args.cache_dir = f'./5_20250909graph_data/{args.site}/world_graphdata/combined_vehicle_data_{args.num_vehicles}_{args.edge_dim}'

    device = setup_device()
    
    # Load Data
    print("Loading data...")

    # Note: create_data_loaders in train.py uses default train_ratio=0.7, val_ratio=0.15
    train_loader, val_loader, test_loader = create_data_loaders(
        pkl_path=args.pkl_path,
        batch_size=args.batch_size,
        train_ratio=0.7, # Default from train.py
        val_ratio=0.15,  # Default from train.py
        use_cache=True,
        cache_dir=args.cache_dir
    )
    
    # Get data info
    sample_batch = next(iter(test_loader))
    node_features = sample_batch.x.size(1)
    num_classes = len(torch.unique(sample_batch.y)) # This might be risky if test batch doesn't have all classes
    
    all_labels = []
    for batch in test_loader:
        all_labels.append(batch.y)
    all_labels = torch.cat(all_labels)
    num_classes = len(torch.unique(all_labels))
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    print(f"Inferred node features: {node_features}")
    
    # Initialize Model
    model = NeighborAwareGraphSAGE(
        in_channels=node_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes, # This might need correction
        edge_dim=args.edge_dim,
        num_layers=args.num_layers,
        graph_data_type=args.graph_data_type,
        aggr=args.aggr
    )
    
    # Handle DataParallel
    if list(checkpoint.keys())[0].startswith('module.'):
        # Checkpoint was saved with DataParallel
        model = torch.nn.DataParallel(model)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to fix size mismatch for output layer...")
        raise e
        
    model = model.to(device)
    print("Model loaded successfully.")
    
    # Output directory
    if args.output_dir is None:
        save_dir = os.path.join(os.path.dirname(args.model_path), 'test_results')
    else:
        save_dir = args.output_dir
        
    evaluate(model, test_loader, device, args, save_dir)

if __name__ == "__main__":
    main()
