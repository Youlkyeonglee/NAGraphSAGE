import os
import datetime
from datetime import timedelta
import numpy as np
import time
from tqdm import tqdm  # Added tqdm
import yaml
import traceback
import argparse
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import print_model_summary, get_loss_function, calculate_class_weights  # Assumption: This module is defined
from dataloader.dataloader import create_data_loaders  # Use previously provided function
from models.model import NeighborAwareGraphSAGE
from models.convlayer import NeighborAwareSAGEConv
# from utils.visualization import visualize_results_static  # New visualization function

# Add date-based project folder creation and increment function
def create_run_directories(project_name=None, args=None):
    """Create directory structure to save training results."""
    # Use timestamp up to day unit
    date_stamp = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # If no project name, use date as project name
    if project_name is None:
        project_name = date_stamp
    else:
        # Add date to project name (num_vehicles and edge_dim are already included in project_name)
        project_name = f"{project_name}_{date_stamp}"
    
    run_dir = os.path.join('runs', args.project_name)
    
    # Check if project with same name exists and increment number
    base_project_name = project_name
    count = 1
    
    while True:
        # Create path with current project name
        train_dir = os.path.join(run_dir, 'train', project_name)
        
        # Check if already exists
        if os.path.exists(train_dir):
            # If exists, increment number
            project_name = f"{base_project_name}_{count}"
            count += 1
        else:
            # If not exists, break loop
            break
    
    # Create final path
    train_dir = os.path.join(run_dir, 'train', project_name)
    val_dir = os.path.join(run_dir, 'val', project_name)
    
    # Create all necessary subdirectories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'visualizations'), exist_ok=True)
    
    print(f"Created directories for project: {project_name}")
    print(f"  - Training directory: {train_dir}")
    print(f"  - Validation directory: {val_dir}")
    
    return {
        'project_name': project_name,
        'timestamp': date_stamp,
        'run_dir': run_dir,
        'train_dir': train_dir,
        'val_dir': val_dir
    }

# Print model summary and save to file
def save_model_summary(model, directory, device):
    """Save model summary to a text file."""
    os.makedirs(directory, exist_ok=True)
    summary_file = os.path.join(directory, "model_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Device: {device}\n")  # Now device is available
        f.write(f"Model Summary - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Save model architecture
        f.write("Model Architecture:\n")
        f.write(str(model) + "\n\n")
        
        # Redirect print_model_summary output to file
        with redirect_stdout(f):
            print_model_summary(model)
        
        # Save additional model info
        f.write("\n" + "="*80 + "\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        
        # Calculate parameters by layer
        f.write("\nParameters by Layer:\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.numel()} parameters\n")
        
        f.write(f"Device: {device}\n")  # Add current device info

    # Print to console
    print_model_summary(model)
    
    return summary_file

# Save improved data statistics
def save_data_statistics(loaders, dirs):
    stats_file = os.path.join(dirs['train_dir'], 'data_statistics.txt')
    
    with open(stats_file, 'w') as f:
        f.write(f"Data Statistics - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for name, loader in [('Training', loaders[0]), ('Validation', loaders[1]), ('Test', loaders[2])]:
            total_nodes = 0
            class_counts = {}
            num_graphs = 0
            
            # Iterate through all batches in data loader
            for batch in loader:
                total_nodes += batch.y.size(0)
                num_graphs += batch.num_graphs
                
                # Count per class
                for label in batch.y.cpu().numpy():
                    if label not in class_counts:
                        class_counts[label] = 0
                    class_counts[label] += 1
            
            f.write(f"{name} Dataset:\n")
            f.write(f"  Total graphs: {num_graphs}\n")
            f.write(f"  Total nodes: {total_nodes}\n")
            f.write(f"  Class distribution:\n")
            
            for label, count in sorted(class_counts.items()):
                f.write(f"    Class {label}: {count} nodes ({count/total_nodes*100:.2f}%)\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    # Print to console
    print("\n" + "="*80)
    print("Data Statistics:")
    for name, loader in [('Train', loaders[0]), ('Val', loaders[1]), ('Test', loaders[2])]:
        total_nodes = sum(data.y.size(0) for data in loader)
        print(f"  {name} Dataset: {total_nodes} nodes")
        
        # Calculate class distribution
        class_counts = {}
        for data in loader:
            for label in data.y.cpu().numpy():
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
                
        # Print class distribution
        for label, count in sorted(class_counts.items()):
            print(f"    - Class {label}: {count} nodes ({count/total_nodes*100:.2f}%)")
            
    print("="*80)
    print(f"Refer to {stats_file} for detailed statistics.\n")
    
    return stats_file

# Multi-GPU setup
def setup_device():
    """Device setup for multi-GPU configuration"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"\nMulti-GPU detected: using {num_gpus} GPUs")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            device = torch.device('cuda')
            return device, num_gpus
        else:
            print(f"\nUsing single GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
            return device, 1
    else:
        print("\nUsing CPU")
        device = torch.device('cpu')
        return device, 0

device, num_gpus = setup_device()

# Model complexity calculation function
def calculate_model_complexity(model, input_dim, edge_dim, num_nodes=100, num_edges=300):
    """Approximate model complexity (FLOPs)."""
    try:
        total_flops = 0
        total_params = 0
        
        # Check model type and print details
        model_type = model.__class__.__name__
        print(f"Calculating complexity for {model_type} model...")
        print(f"Model structure: {model}")
        
        # Use specialized calculation by model type
        is_gcn_model = 'GCN' in model_type
        is_sage_model = 'SAGE' in model_type or 'GraphSAGE' in model_type
        
        # Calculate FLOPs for each layer of GCN or GraphSAGE model
        layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias') and isinstance(module.weight, torch.Tensor):
                # Calculate number of parameters in module
                layer_params = module.weight.numel()
                if module.bias is not None:
                    layer_params += module.bias.numel()
                total_params += layer_params
                
                # Get input and output dimensions (handle dimension errors safely)
                try:
                    # If weight is at least 2D (most linear layers)
                    if len(module.weight.shape) >= 2:
                        in_features = module.weight.size(1)
                        out_features = module.weight.size(0)
                    # If weight is 1D (some special layers)
                    else:
                        in_features = 1
                        out_features = module.weight.size(0)
                        print(f"Warning: {name} has 1D weight with shape {module.weight.shape}. Assuming in_features=1.")
                except Exception as e:
                    print(f"Error getting dimensions for {name}: {e}")
                    print(f"Weight shape: {module.weight.shape}")
                    # Use safe default values
                    in_features = 1
                    out_features = module.weight.size(0) if len(module.weight.shape) > 0 else module.weight.numel()
                
                # Calculate FLOPs for Linear layer: in_features multiply-adds (MAC) per output
                # MAC operation is 2 FLOPs (1 multiply + 1 add)
                linear_flops = num_nodes * in_features * out_features * 2
                
                # Message passing cost per edge: output features per edge
                message_passing_flops = 0
                if any(conv_name in name.lower() for conv_name in ["conv", "sage", "gcn"]):
                    # GCN message passing includes neighbor feature aggregation and normalization
                    if is_gcn_model:
                        # GCN: feature passing per edge (num edges * num features)
                        # Edge normalization (num edges * 1)
                        message_passing_flops = num_edges * (out_features + 1)
                    # GraphSAGE averages neighbor features and combines with self features
                    elif is_sage_model:
                        # SAGE: feature passing per edge + aggregation (averaging) + combination with self features
                        message_passing_flops = num_edges * out_features + num_nodes * out_features * 2
                    else:
                        # Basic message passing cost
                        message_passing_flops = num_edges * out_features
                
                # Activation function cost (operations equal to output dimension per node)
                # ReLU = num nodes * output features (max(0, x) operation per feature)
                activation_flops = num_nodes * out_features
                
                # Total layer FLOPs
                layer_flops = linear_flops + message_passing_flops + activation_flops
                total_flops += layer_flops
                
                # Save layer info
                layers.append({
                    'name': name,
                    'params': layer_params,
                    'flops': layer_flops,
                    'in_features': in_features,
                    'out_features': out_features,
                    'linear_flops': linear_flops,
                    'message_passing_flops': message_passing_flops,
                    'activation_flops': activation_flops
                })
                
                # Print layer complexity details
                print(f"Layer {name}: {layer_params:,} params, {layer_flops/1e6:.2f} MFLOPs")
                print(f"  - Linear: {linear_flops/1e6:.2f} MFLOPs")
                print(f"  - Message Passing: {message_passing_flops/1e6:.2f} MFLOPs")
                print(f"  - Activation: {activation_flops/1e6:.2f} MFLOPs")
        
        # Print total complexity summary
        print(f"Total model complexity: {total_flops/1e6:.2f} MFLOPs")
        print(f"Total model parameters: {total_params:,}")
        
        return total_flops
    except Exception as e:
        print(f"Error calculating model complexity: {e}")
        traceback.print_exc()
        return 0

# Loss and accuracy graph visualization function
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Visualize and save training and validation loss/accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    
    # Set Figure size
    plt.figure(figsize=(12, 10))
    
    # Loss graph
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Set Y-axis range from 0 to 10% above max value
    max_loss = max(max(train_losses), max(val_losses))
    plt.ylim(0, max_loss * 1.1)
    
    # Accuracy graph
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Set Y-axis range from 0 to 1.0
    plt.ylim(0, 1.0)
    
    # Adjust graph layout
    plt.tight_layout()
    
    # Save graph
    plot_file = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Training curves saved to: {plot_file}")
    
    # Save training loss data to CSV file
    csv_file = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_file, 'w') as f:
        f.write('epoch,train_loss,train_accuracy,val_loss,val_accuracy\n')
        for i in range(len(epochs)):
            f.write(f'{epochs[i]},{train_losses[i]},{train_accs[i]},{val_losses[i]},{val_accs[i]}\n')
    print(f"Training metrics saved to: {csv_file}")
    
    plt.close()  # Close to prevent memory leak



# Function to calculate model parameter count and size
def get_model_stats(model):
    """Calculate number of parameters and size (MB) of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    # Calculate model size in MB (4 bytes per float32 parameter)
    model_size_mb = total_params * 4 / (1024 * 1024)
    return total_params, model_size_mb

# GPU memory usage measurement function
def get_gpu_memory_usage():
    """Returns current GPU memory usage in MiB."""
    try:
        import torch
        if torch.cuda.is_available():
            # Currently allocated memory (MiB)
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            # Usage including cache (MiB)
            memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return memory_allocated, memory_reserved
        return 0, 0
    except Exception as e:
        print(f"Error measuring GPU memory usage: {e}")
        return 0, 0

def get_all_gpu_memory_usage():
    """Returns memory usage of all GPUs."""
    if not torch.cuda.is_available():
        return {}
    
    gpu_memory = {}
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
        gpu_memory[f'gpu_{i}'] = {'allocated': allocated, 'reserved': reserved}
    
    # Return to default GPU
    torch.cuda.set_device(0)
    return gpu_memory

class KLRegularizedLoss(nn.Module):
    """Loss function normalizing attention weight distribution using KL divergence"""
    def __init__(self, base_criterion, kl_weight=0.1, target_distribution='uniform'):
        super().__init__()
        self.base_criterion = base_criterion  # Basic classification loss (CE, Focal etc.)
        self.kl_weight = kl_weight  # KL loss weight
        self.target_distribution = target_distribution
        
    def forward(self, output, target, model=None):
        # Calculate basic classification loss
        base_loss = self.base_criterion(output, target)
        
        # Calculate KL regularization loss
        kl_loss = torch.tensor(0.0, device=output.device)
        
        if model is not None:
            attention_weights = []
            
            # Collect weights from all AttributeAwareRGCN_SAGEConv layers in model
            for module in model.modules():
                if isinstance(module, NeighborAwareSAGEConv):
                    weights = module.get_attention_weights()
                    if weights is not None:
                        attention_weights.append(weights)
            
            if attention_weights:
                # Combine all attention weights into one tensor
                all_weights = torch.cat(attention_weights, dim=0)
                
                # Create target distribution
                if self.target_distribution == 'uniform':
                    # Uniform distribution (all values have equal probability)
                    target_dist = torch.ones_like(all_weights) / all_weights.size(0)
                elif self.target_distribution == 'normal':
                    # Normal distribution (mean 0.5, std 0.1)
                    target_dist = torch.distributions.Normal(0.5, 0.1).sample(all_weights.shape).to(all_weights.device)
                    target_dist = target_dist / target_dist.sum()
                
                # Calculate KL divergence (between weight distribution and target distribution)
                # Add small value for numerical stability if weight is 0
                weight_dist = all_weights / (all_weights.sum() + 1e-10)
                kl_loss = F.kl_div(
                    (weight_dist + 1e-10).log(),  # Add small value before log for numerical stability
                    target_dist,
                    reduction='batchmean'
                )
        
        # Total loss = basic loss + KL weight * KL loss
        total_loss = base_loss + self.kl_weight * kl_loss
        return total_loss

# Training and validation loop
def train_and_evaluate(model, args, train_loader, val_loader, test_loader, dirs, device,
                      criterion, optimizer_name, lr, weight_decay, scheduler_name, 
                      num_epochs, project_name="default_run", optimizer=None, scheduler=None,
                      use_kl_regularization=False, kl_weight=0.1, use_feature_loss=False):
    best_val_acc = 0
    best_epoch = 0
    start_time = time.time()
    
    # Add TensorBoardX SummaryWriter
    tb_log_dir = os.path.join(dirs['train_dir'], 'tensorboard')
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Record loss and accuracy
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Set training log file path
    log_file = os.path.join(dirs['train_dir'], 'training_log.txt')
    best_model_path = os.path.join(dirs['train_dir'], 'best_model.pth')
    # Wrap existing criterion with KLRegularizedLoss if using KL regularization
    if use_kl_regularization:
        criterion = KLRegularizedLoss(
            base_criterion=criterion, 
            kl_weight=kl_weight,
            target_distribution='uniform'  # 'uniform' 또는 'normal'
        )
    # Set loss function
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()  # Basic loss function
    
    with open(log_file, 'w') as f:
        f.write(f"Training Log - Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Device: {device}\n")  # Add current device info
        f.write(f"Model summary file: {os.path.join(dirs['train_dir'], 'model_summary.txt')}\n\n")
        f.write(f"Training Parameters:\n")
        f.write(f"- Epochs: {num_epochs}\n")
        
        # Define your optimizer first
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if scheduler is None and scheduler_name != "None":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Then use it in your logging
        f.write(f"- Optimizer: {optimizer.__class__.__name__}\n")
        f.write(f"- Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"- Weight Decay: {optimizer.param_groups[0]['weight_decay']}\n")
        f.write(f"- Loss Function: {criterion.__class__.__name__}\n")
        
        # Check if scheduler exists before writing its info
        if scheduler is not None:
            f.write(f"- Scheduler: {scheduler.__class__.__name__}\n\n")
        else:
            f.write("- Scheduler: None\n\n")
        
        f.write("="*80 + "\n\n")
        f.write("Training Progress:\n\n")
    
    print("\n" + "="*80)
    print(f"Starting training for {num_epochs} epochs")
    print(f"Log file: {log_file}")
    print("="*80)
    
    # early_stopping = EarlyStopping(patience=50)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Show progress with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         total=len(train_loader), leave=False, ncols=100)
        
        for batch in train_pbar:
            # Move batch to GPU
            batch = batch.to(device)
            
            # Check class index
            # Check actual class range in data
            min_class = batch.y.min().item()
            max_class = batch.y.max().item()
            if min_class < 0 or max_class >= len(torch.unique(batch.y)):
                print(f"Invalid class index found: min={batch.y.min().item()}, max={batch.y.max().item()}, num_classes={len(torch.unique(batch.y))}")
                print(batch.y)
                exit(1)
            # edge_index check
            if batch.edge_index.min() < 0 or batch.edge_index.max() >= batch.x.size(0):
                print("Invalid edge_index found!")
                print("x.shape:", batch.x.shape)
                print("edge_index.shape:", batch.edge_index.shape)
                print("edge_index.min():", batch.edge_index.min().item(), "edge_index.max():", batch.edge_index.max().item())
                print("edge_index:", batch.edge_index)
                exit(1)
            
            optimizer.zero_grad()
            
            # Get feature vectors if using loss function that uses them
            if use_feature_loss:
                out, features = model(batch, return_features=True)
                ce_loss = torch.nn.functional.cross_entropy(out, batch.y)  # Basic classification loss
                
                if use_kl_regularization:
                    loss = criterion(out, batch.y, model=model)
                else:
                    loss = ce_loss
            else:
                # Use normal loss function
                out = model(batch)
                if use_kl_regularization:
                    loss = criterion(out, batch.y, model=model)
                else:
                    loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            # Update every batch if using OneCycleLR
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
                
            train_loss += loss.item()
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct = (pred == batch.y).sum().item()
            total = batch.y.size(0)
            train_correct += correct
            train_total += total
            
            # Update current batch results in tqdm
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
            
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # 손실과 정확도 기록
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # === TensorBoardX Record: Train ===
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_acc, epoch+1)
        
        # Record memory usage per GPU if using multi-GPU
        if num_gpus > 1:
            all_gpu_memory = get_all_gpu_memory_usage()
            for gpu_id, memory_info in all_gpu_memory.items():
                writer.add_scalar(f'GPU_Memory/{gpu_id}_allocated', memory_info['allocated'], epoch+1)
                writer.add_scalar(f'GPU_Memory/{gpu_id}_reserved', memory_info['reserved'], epoch+1)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds, val_labels = [], []
        
        # Show validation progress with tqdm
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", 
                       total=len(val_loader), leave=False, ncols=100)
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                
                # Get feature vectors if using loss function that uses them
                if use_feature_loss:
                    out, features = model(batch, return_features=True)
                    ce_loss = torch.nn.functional.cross_entropy(out, batch.y)  # Basic classification loss
                    
                    # Use only basic classification loss
                    batch_loss = ce_loss.item()
                else:
                    # Use normal loss function
                    out = model(batch)
                    batch_loss = criterion(out, batch.y).item()  # <= Ensure this line always runs!
                
                val_loss += batch_loss
                
                pred = out.argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total = batch.y.size(0)
                val_correct += correct
                val_total += total
                
                val_preds.extend(pred.cpu().tolist())
                val_labels.extend(batch.y.cpu().tolist())
                
                # Update current batch results in tqdm
                val_pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'acc': f"{correct/total:.4f}"
                })
                
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Calculate F1 score for validation set
        if val_labels and val_preds:
            val_micro_f1 = f1_score(val_labels, val_preds, average='micro')
            val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        else:
            val_micro_f1 = val_macro_f1 = 0.0
        
        # Record loss and accuracy
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # === TensorBoardX Record: Validation ===
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/val', val_acc, epoch+1)
        writer.add_scalar('F1/val_macro', val_macro_f1, epoch+1)
        writer.add_scalar('F1/val_micro', val_micro_f1, epoch+1)
        writer.add_scalar('LearningRate', current_lr, epoch+1)
        
        # Print current learning rate
        
        # Check if current epoch is best (can use macro-F1 instead of accuracy)
        is_best = False
        if val_acc > best_val_acc:  # Accuracy criteria (or change to val_macro_f1 > best_val_f1)
            best_val_acc = val_acc
            best_epoch = epoch + 1
            is_best = True
            torch.save(model.state_dict(), best_model_path)
            print(f'\nNew best model saved with validation accuracy: {val_acc:.4f} (micro-F1: {val_micro_f1:.4f}, macro-F1: {val_macro_f1:.4f})')
        
        # Epoch duration
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time / (epoch + 1) * num_epochs
        remaining_time = estimated_total - elapsed_time
        
        # Print training status (improved version)
        status_message = "\r" + "-"*80 + "\n"
        status_message += f'| Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | Progress: {(epoch+1)/num_epochs*100:.1f}% |\n'
        status_message += f'| LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} |\n'
        status_message += f'| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_macro_f1:.4f} |\n'
        status_message += f'| {"★ NEW BEST ★" if is_best else "Not best"} | Elapsed: {str(timedelta(seconds=int(elapsed_time)))} | ETA: {str(timedelta(seconds=int(remaining_time)))} |\n'
        status_message += "-"*80
        
        print(status_message)
        
        # Record to log file
        with open(log_file, 'a') as f:
            f.write(status_message + "\n")
        
        # Visualize graph every 10 epochs or when best model is saved
        if (epoch + 1) % 10 == 0 or is_best:
            with torch.no_grad():
                # Get first batch from train_loader
                train_batch = next(iter(train_loader)).to(device)
                
                # Get feature vectors if using feature loss
                if use_feature_loss:
                    train_out, _ = model(train_batch, return_features=True)
                else:
                    train_out = model(train_batch)
                    
                train_preds = train_out.argmax(dim=1)
                
                # Move to CPU for visualization
                train_batch = train_batch.cpu()
                train_preds = train_preds.cpu()
                
            # Graph visualization (every 10 epochs)
            # if (epoch + 1) % 10 == 0:
            #     try:
            #         visualize_results_static(train_batch, train_preds, epoch + 1, args, project_name=dirs['project_name'])
            #         print(f"Visualization saved successfully")
            #     except Exception as e:
            #         print(f"Error during visualization: {str(e)}")

    # Print final results
    final_message = "\n" + "="*80 + "\n"
    final_message += f"Training completed in {str(timedelta(seconds=int(time.time() - start_time)))}\n"
    final_message += f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n"
    final_message += "="*80
    
    print(final_message)
    
    # Record final results to log file
    with open(log_file, 'a') as f:
        f.write(final_message + "\n")

    # Test evaluation after training completion
    model.eval()
    test_preds, test_labels = [], []
    test_pred_proba = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
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

    # Generate Precision-Recall Curve
    average_precision = plot_precision_recall_curve(
        np.array(test_labels), 
        test_pred_proba, 
        dirs['val_dir']
    )

    # Calculate F1 score and other metrics
    if len(test_preds) > 0 and len(test_labels) > 0:
        try:
            # Calculate existing metrics
            macro_f1 = f1_score(test_labels, test_preds, average='macro')
            micro_f1 = f1_score(test_labels, test_preds, average='micro')
            
            # Calculate precision, recall, F1 score per class
            precision, recall, f1, support = precision_recall_fscore_support(
                test_labels, test_preds, average=None
            )
            
            # Calculate overall precision and recall (macro average)
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            
            print(f'Test Accuracy: {test_acc:.4f}')
            print(f'Micro-F1 Score: {micro_f1:.4f}')
            print(f'Macro-F1 Score: {macro_f1:.4f}')
            print(f'Macro-Precision: {macro_precision:.4f}')
            print(f'Macro-Recall: {macro_recall:.4f}')
            
            print("\nPerformance per class:")
            for i in range(len(precision)):
                print(f'  Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}')
        
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            traceback.print_exc()
    else:
        print("Cannot calculate F1 score because there are no test predictions or labels.")
        macro_f1 = micro_f1 = 0.0

    # Test data visualization
    if len(test_loader) > 0:
        with torch.no_grad():
            test_batch = next(iter(test_loader)).to(device)
            test_out = model(test_batch)
            test_preds = test_out.argmax(dim=1)
            
            # Move to CPU for visualization
            test_batch = test_batch.cpu()
            test_preds = test_preds.cpu()
            
        
        # try:
        #     visualize_results_static(test_batch, test_preds, num_epochs, args, project_name=dirs['project_name'])
        #     print(f"Test visualization saved successfully")
        # except Exception as e:
        #     print(f"Error during test visualization: {str(e)}")

    # Save final results to file
    test_results_file = os.path.join(dirs['val_dir'], 'test_results.txt')
    with open(test_results_file, 'w') as f:
        f.write(f"Test Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Macro-Precision: {macro_precision:.4f}\n")
        f.write(f"Macro-Recall: {macro_recall:.4f}\n")
        f.write(f"Micro-F1 Score: {micro_f1:.4f}\n")
        f.write(f"Macro-F1 Score: {macro_f1:.4f}\n\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n\n")
        
        # Save performance per class
        if len(test_preds) > 0 and len(test_labels) > 0:
            f.write("Performance per class:\n")
            for i in range(len(precision)):
                f.write(f"  Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}\n")
            
            cm = confusion_matrix(test_labels, test_preds)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm) + "\n")

    # Generate final training curve graph
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, dirs['train_dir'])
    
    # Visualize and save confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(test_labels, test_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save
        cm_path = os.path.join(dirs['val_dir'], 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {cm_path}.")
    except Exception as e:
        print(f"Error visualizing confusion matrix: {e}")

    # Add t-SNE visualization after confusion matrix visualization
    print("\nGenerating t-SNE visualization...")

    # Calculate model statistics
    total_params, model_size_mb = get_model_stats(model)
    gpu_memory_allocated, gpu_memory_reserved = get_gpu_memory_usage()
    
    # Print memory usage of all GPUs if using multi-GPU
    if num_gpus > 1:
        print(f"\nMulti-GPU memory usage:")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
            print(f"  GPU {i}: Allocated {allocated:.2f}MB, Reserved {reserved:.2f}MB")
        # Set back to default GPU
        torch.cuda.set_device(0)
    
    # Calculate model complexity
    try:
        # Get number of nodes and edges from test batch
        sample_batch = next(iter(test_loader))
        num_nodes = sample_batch.x.size(0)
        num_edges = sample_batch.edge_index.size(1)
        input_dim = sample_batch.x.size(1)
        edge_dim = 0
        
        if hasattr(sample_batch, 'edge_attr') and sample_batch.edge_attr is not None:
            if isinstance(sample_batch.edge_attr, torch.Tensor):
                if len(sample_batch.edge_attr.shape) > 0:
                    edge_dim = sample_batch.edge_attr.size(1) if len(sample_batch.edge_attr.shape) > 1 else 1
        
        print(f"Calculating complexity with real dataset values:")
        print(f"  - Nodes: {num_nodes}")
        print(f"  - Edges: {num_edges}")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Edge dimension: {edge_dim}")
        
        # Calculate model complexity (MFLOPs)
        model_complexity = calculate_model_complexity(model, input_dim, edge_dim, num_nodes, num_edges)
        model_complexity_mflops = model_complexity / 1e6
        print(f"Model complexity: {model_complexity_mflops:.2f} MFLOPs")
    except Exception as e:
        print(f"Error calculating model complexity: {e}")
        traceback.print_exc()
        
        # Retry with default values if failed
        try:
            print("Retry complexity calculation with default values:")
            default_nodes = 100
            default_edges = 300
            model_complexity = calculate_model_complexity(model, input_dim=64, edge_dim=9, 
                                                         num_nodes=default_nodes, num_edges=default_edges)
            model_complexity_mflops = model_complexity / 1e6
            print(f"Model complexity based on default values: {model_complexity_mflops:.2f} MFLOPs")
        except Exception as e2:
            print(f"Error during retry with default values: {e2}")
            model_complexity_mflops = 0

    # Extract result folder name from project name
    result_folder_name = os.path.basename(dirs['train_dir'])
    
    # Save result summary info
    with open(test_results_file, 'a') as f:
        f.write("\n\nAdditional model info:\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Model size: {model_size_mb:.2f} MB\n")
        f.write(f"Model complexity: {model_complexity_mflops:.2f} MFLOPs\n")
        f.write(f"GPU allocated memory: {gpu_memory_allocated:.2f} MiB\n")
        f.write(f"GPU reserved memory: {gpu_memory_reserved:.2f} MiB\n")
    # Result summary file path (collect all run results in one place)
    RESULTS_SUMMARY_FILE = f'./runs/{args.project_name}/{args.project_name}.csv'
    # Add to result summary file
    os.makedirs(os.path.dirname(RESULTS_SUMMARY_FILE), exist_ok=True)
    
    # Add header if summary file does not exist
    if not os.path.exists(RESULTS_SUMMARY_FILE):
        with open(RESULTS_SUMMARY_FILE, 'w') as f:
            f.write("Result_Folder_Name,ACC,Micro-F1,Macro-F1,Precision,Recall,Best Epoch,Num_Layers,hidden_channels,edge_dim,num_vehicles,kl_weight,scheduler,aggr,Total Parameters,Model size,Complexity(MFLOPs),GPU Usage,Num_GPUs,Graph_data_type,Site\n")
    
    # Add new row to result table
    with open(RESULTS_SUMMARY_FILE, 'a') as f:
        # Extract info from result folder name
        parts = result_folder_name.split('_')
        model_type = parts[0]
        
        # Extract edge_dim, hidden_channels, scheduler, num_layers, aggr
        # 예: "GraphSAGE_vehicle_edge_5_256_CosineAnnealingLR_layer4_aggrpool"
        edge_dim = None
        hidden_channels = None
        scheduler_name = None
        num_layers = None
        aggr_type = None
        
        for i, part in enumerate(parts):
            if part == "edge" and i+1 < len(parts):
                edge_dim = parts[i+1]
            elif part.startswith("layer"):
                num_layers = part.replace("layer", "")
            elif part.startswith("aggr"):
                aggr_type = part.replace("aggr", "")
        
        # hidden_channels is just the number part
        for part in parts:
            if part.isdigit() and part != edge_dim and part != num_layers:
                hidden_channels = part
                break
        
        # scheduler is one of the remaining parts
        scheduler_candidates = ["CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR", "None"]
        for part in parts:
            if part in scheduler_candidates:
                scheduler_name = part
                break
                
        # Write data in CSV format
        f.write(f"{result_folder_name},{test_acc:.4f},{micro_f1:.4f},{macro_f1:.4f},{macro_precision:.4f},{macro_recall:.4f},{best_epoch},{num_layers},{hidden_channels},{edge_dim},{args.num_vehicles},{args.kl_weight},{scheduler_name},{aggr_type},{total_params},{model_size_mb:.2f},{model_complexity_mflops:.2f},{gpu_memory_allocated:.2f},{num_gpus},{args.graph_data_type},{args.site}\n")
    print(f"\nResult summary added to {RESULTS_SUMMARY_FILE}.")

    return test_acc, micro_f1, macro_f1, best_epoch

def run_single_training(args):
    """Train model with single hyperparameter setting."""
    # Check and replace aggr parameter
    valid_aggrs = ['mean', 'max', 'add', 'mul']
    if args.aggr not in valid_aggrs:
        print(f"Warning: aggr='{args.aggr}' is not supported. Replacing with 'mean'.")
        args.aggr = 'mean'
        
    # Set file path according to edge_dim
    args.pkl_path = args.pkl_path
    args.cache_dir = args.cache_dir
    
    
    # Define project name (include num_vehicles info, add class balance info)
    balance_suffix = f"_balance{args.class_balance_ratio}" if args.balance_classes else "_nobalance"
    
    project_name = f"{args.model_type}_vehicle_edge_{args.edge_dim}_{args.hidden_channels}_{args.scheduler}_layer{args.num_layers}_aggr{args.aggr}_{args.time}_{args.attention_type}_num_vehicles{args.num_vehicles}_kl_weight{args.kl_weight}_Site_{args.site}{balance_suffix}"
    
    # Create directory structure for saving results (pass vehicle_type)
    dirs = create_run_directories(project_name, args)
    
    # Save current parameters to YAML file
    current_params = {
        'data': {
            'pkl_path': args.pkl_path,
            'cache_dir': args.cache_dir,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'num_relations': args.num_relations,  # Added argument
            'balance_classes': args.balance_classes,  # Class balance adjustment (script specific)
            'max_samples_per_class': args.max_samples_per_class,  # Max samples per class (script specific)
            'class_balance_ratio': args.class_balance_ratio  # Sample ratio applied equally to all classes (script specific)
        },
        'model': {
            'type': args.model_type,
            'hidden_channels': args.hidden_channels,
            'num_layers': args.num_layers,
            'aggr': args.aggr,
            'edge_dim': args.edge_dim,
            'attention_type': args.attention_type  # Added argument
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'loss': args.loss,
            'focal_gamma': args.focal_gamma,
            'smoothing_alpha': args.smoothing_alpha,
            'use_class_weights': args.use_class_weights,
            'optimizer': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler': args.scheduler,
            'time': args.time,  # Added argument
            'use_kl_regularization': args.use_kl_regularization if hasattr(args, 'use_kl_regularization') else False,  # KL divergence related argument
            'kl_weight': args.kl_weight if hasattr(args, 'kl_weight') else 0.1  # KL divergence related argument
        }
    }
    
    # Add additional parameters if using centroid or contrastive loss function
    if args.loss in ['centroid', 'contrastive']:
        current_params['training'].update({
            'feat_dim': args.hidden_channels,  # Set feature vector dimension same as hidden_channels
            'lambda_inter': args.lambda_inter,
            'lambda_intra': args.lambda_intra,
            'margin': args.margin,
            'temperature': args.temperature,
            'lambda_val': args.lambda_val
        })
    
    # Save current settings to YAML file at start
    config_save_path = os.path.join(dirs['train_dir'], 'used_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(current_params, f, default_flow_style=False, sort_keys=False)
    
    print(f"Current training settings saved to {config_save_path}.")
    
    # Create data loaders
    os.makedirs(args.cache_dir, exist_ok=True)
    train_loader, val_loader, test_loader = create_data_loaders(
        pkl_path=args.pkl_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_cache=True,
        cache_dir=args.cache_dir,
        balance_classes=args.balance_classes,
        max_samples_per_class=args.max_samples_per_class,
        class_balance_ratio=args.class_balance_ratio,
        normalize_coords=args.normalize_coords,
        coord_scale_factor=args.coord_scale_factor
    )
    
    # Print and save data loader statistics
    stats_file = save_data_statistics([train_loader, val_loader, test_loader], dirs)
    
    # Get input feature dimension and number of classes
    sample_batch = next(iter(train_loader))
    node_features = sample_batch.x.size(1)
    num_classes = len(torch.unique(sample_batch.y))
    
    print(f"Detected node features dimension: {node_features}")
    print(f"Detected number of classes: {num_classes}")
    
    # Set class weights
    class_weights = None
    if args.use_class_weights or args.loss == 'weighted_ce':
        class_weights = calculate_class_weights(train_loader, num_classes, device)
        print(f"Using class weights: {class_weights}")
    
    # Set loss function parameters
    loss_params = {
        'focal_gamma': args.focal_gamma,
        'smoothing_alpha': args.smoothing_alpha
    }
    
    # Create loss function
    criterion = get_loss_function(args.loss, class_weights, num_classes, **loss_params)
    print(f"Using loss function: {criterion.__class__.__name__}")
    
    # Select and create model
    model = NeighborAwareGraphSAGE(
        in_channels=node_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        edge_dim=args.edge_dim,
        num_layers=args.num_layers,
        graph_data_type=args.graph_data_type,
        aggr=args.aggr
    )
    
    # Multi-GPU setup
    if num_gpus > 1:
        print(f"Use DataParallel for multi-GPU training")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    # Save model summary info
    summary_file = save_model_summary(model, dirs['train_dir'], device)
    
    # Save multi-GPU info
    if num_gpus > 1:
        gpu_info_file = os.path.join(dirs['train_dir'], 'gpu_info.txt')
        with open(gpu_info_file, 'w') as f:
            f.write(f"Multi-GPU training setup\n")
            f.write(f"="*50 + "\n")
            f.write(f"Number of GPUs used: {num_gpus}\n")
            f.write(f"Actual batch size: {args.batch_size * num_gpus}\n")
            f.write(f"Recommended learning rate: {args.lr * num_gpus}\n")
            f.write(f"\nGPU detailed info:\n")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                f.write(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)\n")
        print(f"GPU info saved to {gpu_info_file}.")
    
    # Initialize optimizer
    optimizer_dict = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": lambda p, lr, weight_decay: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=weight_decay),
        "RMSprop": lambda p, lr, weight_decay: torch.optim.RMSprop(p, lr=lr, momentum=0.9, weight_decay=weight_decay),
        "RAdam": torch.optim.RAdam
    }
    
    # Access actual model parameters when using multi-GPU
    if num_gpus > 1 and isinstance(model, torch.nn.DataParallel):
        model_params = model.module.parameters()
    else:
        model_params = model.parameters()
    
    optimizer = optimizer_dict[args.optimizer](
        model_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler
    if args.scheduler == "OneCycleLR":
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.num_epochs
        scheduler = OneCycleLR(optimizer, max_lr=args.lr*10, total_steps=total_steps)
        scheduler_info = {
            'type': 'OneCycleLR',
            'max_lr': args.lr*10,
            'total_steps': total_steps
        }
    
    # Print batch size adjustment info when using multi-GPU
    if num_gpus > 1:
        effective_batch_size = args.batch_size * num_gpus
        print(f"Multi-GPU training: actual batch size = {args.batch_size} × {num_gpus} = {effective_batch_size}")
        print(f"Recommended learning rate adjustment: {args.lr} → {args.lr * num_gpus} (Proportional to batch size)")
    
    # Save training parameters (including more detailed info)
    training_params = {
        'model': {
            'type': args.model_type,
            'in_channels': node_features,
            'hidden_channels': args.hidden_channels,
            'out_channels': num_classes,
            'num_layers': args.num_layers,
            'aggr': args.aggr,
            'edge_dim': args.edge_dim,
            'attention_type': args.attention_type,  # Added argument
            'num_relations': args.num_relations  # Added argument
        },
        'optimizer': {
            'type': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        'scheduler': scheduler_info,
        'training': {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'device': str(device),
            'time': args.time,  # Added argument
            'use_kl_regularization': args.use_kl_regularization if hasattr(args, 'use_kl_regularization') else False,  # KL divergence related argument
            'kl_weight': args.kl_weight if hasattr(args, 'kl_weight') else 0.1  # KL divergence related argument
        },
        'timestamp': dirs['timestamp'],
        'project_name': dirs['project_name'],
        'loss_function': {
            'type': criterion.__class__.__name__,
            'params': {
                'loss_name': args.loss,
                'use_class_weights': args.use_class_weights,
                'focal_gamma': args.focal_gamma,
                'smoothing_alpha': args.smoothing_alpha
            }
        },
        'data': {
            'pkl_path': args.pkl_path,
            'cache_dir': args.cache_dir,
            'num_relations': args.num_relations  # Added argument
        },
        'hardware': {
            'num_gpus': num_gpus,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(num_gpus)] if num_gpus > 0 else [],
            'effective_batch_size': args.batch_size * num_gpus if num_gpus > 1 else args.batch_size,
            'recommended_lr': args.lr * num_gpus if num_gpus > 1 else args.lr
        }
    }
    
    # Save additional parameters if using centroid or contrastive loss function
    if args.loss in ['centroid', 'contrastive']:
        training_params['loss_function']['params'].update({
            'feat_dim': args.hidden_channels,
            'lambda_inter': args.lambda_inter,
            'lambda_intra': args.lambda_intra,
            'margin': args.margin,
            'temperature': args.temperature,
            'lambda_val': args.lambda_val
        })
    
    # Save detailed training parameters to YAML file
    params_file = os.path.join(dirs['train_dir'], 'training_params.yaml')
    with open(params_file, 'w') as f:
        yaml.dump(training_params, f, default_flow_style=False, sort_keys=False)
    
    print(f"Training parameters saved to: {params_file}")
    
    # Print multi-GPU training performance optimization tips
    if num_gpus > 1:
        print(f"\n🚀 Multi-GPU training performance optimization tips:")
        print(f"  • Actual batch size: {args.batch_size * num_gpus}")
        print(f"  • Recommended learning rate: {args.lr * num_gpus:.6f} (Current: {args.lr:.6f})")
        print(f"  • Consider reducing batch size and increasing learning rate for memory efficiency")
        print(f"  • Monitor memory usage per GPU in TensorBoard")
    
    # Execute training
    test_acc, micro_f1, macro_f1, best_epoch = train_and_evaluate(
        model, args, train_loader, val_loader, test_loader, dirs, device,
        criterion=criterion, optimizer_name=args.optimizer, 
        lr=args.lr, weight_decay=args.weight_decay,
        scheduler_name=args.scheduler, num_epochs=args.num_epochs,
        project_name=dirs['project_name'], optimizer=optimizer, 
        scheduler=scheduler,
        use_kl_regularization=args.use_kl_regularization,
        kl_weight=args.kl_weight,
        use_feature_loss=args.use_feature_loss
    )
    
    return test_acc, micro_f1, macro_f1, best_epoch, dirs['project_name']

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='GNN model training for vehicle state classification')
    
    parser.add_argument('--attention_type', type=str, default='mlp')
    parser.add_argument('--time', type=str, default='1',
                        help='Number of training runs')
    parser.add_argument('--project_name', type=str, default='default_run',
                        help='Project name')
    # Data related arguments
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation data ratio')
    parser.add_argument('--num_relations', type=int, default=7,
                        help='Number of relations')
    parser.add_argument('--graph_data_type', type=str, default='image', help='image, world')
    # Model related arguments
    parser.add_argument('--model_type', type=str, default='NeighborAwareGraphSAGE')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--num_vehicles', type=int, default=4,
                        help='Number of vehicles')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of GNN layers')
    parser.add_argument('--edge_dim', type=int, default=7,
                        help='Number of edge attributes')
    parser.add_argument('--aggr', type=str, default='mean',
                        choices=['mean', 'max', 'add'],
                        help='Graph aggregation function')
    parser.add_argument('--dataset', type=str, default='gongeoptap',
                        choices=['gongeoptap', 'DRIFT'],
                        help='Dataset name')
    # Training related arguments
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, 
                        help='Number of training epochs')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['ce', 'weighted_ce', 'focal', 'label_smoothing'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma parameter')
    parser.add_argument('--smoothing_alpha', type=float, default=0.1,
                        help='Label Smoothing alpha parameter')
    parser.add_argument('--use_class_weights', type=bool, default=False,
                        help='Whether to use class weights')
    # Optimizer related arguments
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'AdamW', 'SGD', 'RMSprop', 'RAdam'],
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay coefficient')
    
    # Scheduler related arguments
    parser.add_argument('--scheduler', type=str, default='OneCycleLR')
    parser.add_argument('--site', type=str, default='',
                        choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                        help='Site name')
    
    parser.add_argument('--use_feature_loss', type=bool, default=True,
                        help='Whether to use Feature Loss')
    parser.add_argument('--use_kl_regularization', type=bool, default=True,
                        help='Whether to use KL divergence regularization')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='KL divergence regularization weight')
    
    # Add multi-GPU related arguments
    parser.add_argument('--use_multi_gpu', type=bool, default=True,
                        help='Whether to use multi-GPU (auto-detect)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='List of GPU IDs to use (e.g., "0,1,2,3"). Auto-detect if None')
    
    # Add class imbalance adjustment parameters (script specific)
    parser.add_argument('--balance_classes', type=bool, default=True,
                        help='Whether to adjust class imbalance (script specific)')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Max samples per class (No limit if None, script specific)')
    parser.add_argument('--class_balance_ratio', type=float, default=1.0,
                        help='Sample ratio applied equally to all classes (1.0=original 100%%, 0.5=50%%, 0.8=80%% etc., script specific)')
    
    # Add coordinate normalization parameters
    parser.add_argument('--normalize_coords', type=bool, default=True,
                        help='Whether to normalize coordinates')
    parser.add_argument('--coord_scale_factor', type=float, default=0.001,
                        help='Coordinate scale factor (convert 4K coordinates to normal coordinates)')
    
    args = parser.parse_args()
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
    
    # GPU setup (consider CUDA_VISIBLE_DEVICES env var)
    device, num_gpus = setup_device()
    
    # Set GPU ID (if specified by user)
    if args.gpu_ids is not None and torch.cuda.is_available():
        available_gpu_count = torch.cuda.device_count()
        requested_gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        # Use only within GPU range limited by CUDA_VISIBLE_DEVICES
        valid_gpu_ids = [gpu_id for gpu_id in requested_gpu_ids if gpu_id < available_gpu_count]
        
        if valid_gpu_ids:
            torch.cuda.set_device(valid_gpu_ids[0])  # Set default GPU
            print(f"GPU ID specified by user: {valid_gpu_ids}")
        else:
            print(f"Warning: Specified GPU ID {requested_gpu_ids} is unavailable. Automatically using GPU 0.")
            torch.cuda.set_device(0)
    
    # YAML config file load function is currently not used
    # Can be implemented later if needed
    
    # Run single training
    print("\n" + "="*80)
    print(f"🚀 Start training: model={args.model_type}, edge_dim={args.edge_dim}, hidden={args.hidden_channels}, "
          f"layers={args.num_layers}, scheduler={args.scheduler}, "
          f"loss={args.loss}, optimizer={args.optimizer}, aggr={args.aggr}")
    if num_gpus > 1:
        print(f"🎯 Multi-GPU training: {num_gpus} GPUs used")
    print("="*80)
    
    try:
        run_single_training(args)
        print(f"\n✅ Training completed!\n")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        traceback.print_exc()

def calculate_combinations(grid):
    """Calculate total combinations in hyperparameter grid."""
    total = 1
    for values in grid.values():
        total *= len(values)
    return total

def plot_precision_recall_curve(y_true, y_pred_proba, save_dir):
    """Draw and save Precision-Recall Curve."""
    plt.figure(figsize=(10, 8))
    
    # Calculate Precision-Recall Curve for each class
    n_classes = y_pred_proba.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            (y_true == i).astype(int), 
            y_pred_proba[:, i]
        )
        average_precision[i] = average_precision_score(
            (y_true == i).astype(int), 
            y_pred_proba[:, i]
        )
        
        # Draw PR Curve for each class
        plt.plot(recall[i], precision[i],
                label=f'Class {i} (AP = {average_precision[i]:0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Class')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 그래프 저장
    pr_curve_path = os.path.join(save_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return average_precision

if __name__ == "__main__":
    main()
