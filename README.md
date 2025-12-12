## Data Folder

- Explanation of data in URL
    - Gongeoptap: [Link to repository](https://github.com/Youlkyeonglee/traffic_graph_data)
    - Download links:
    1. [graph_data](https://drive.google.com/drive/folders/1m7u_9K9q3s5-zPCxdXg6gT4iBLGHYcdi) (Google Drive, image coorindates)
    2. [world_graph_data](https://drive.google.com/drive/folders/1bz7fisGKwMBUmbzaFgpFabr54Ze6ZGr_) (Google Drive, world coorindates)
    
```
gongeoptap_graph_data
├── graph_data/ 
│    ├── combined_vehicle_data_4_5.pkl
│    └── combined_vehicle_data_4_5 (train/val/test cache file)
└── world_graph_data/
│    └── ...

DRIFT_graph_data
├── A/ (Site A, B, C, D, E, I)
│    ├── graph_data/ 
│    │    ├── combined_vehicle_data_4_5.pkl
│    │    └── combined_vehicle_data_4_5 (train/val/test cache file)
│    └── world_graph_data/
│    │    └── ...
```

## Train
```bash
    python train.py \
    --model_type "NeighborAwareGraphSAGE" \
    --edge_dim 5 \
    --num_vehicles 4 \
    --attention_type mlp \
    --time 1 \
    --graph_data_type image \
    --project_name project_gongeoptap \
    --hidden_channels 128 \
    --num_layers 5 \
    --batch_size 64 \
    --num_epochs 1000 \
    --loss "focal" \
    --optimizer Adam \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --scheduler "OneCycleLR" \
    --aggr "add" \
    --kl_weight 0.3 \
    --use_kl_regularization True \
    --use_feature_loss True \
    --balance_classes True \
    --class_balance_ratio 1.0 \
    --dataset gongeoptap
```

## Test
```bash
python test.py \
  --model_path runs/project_gongeoptap/train/NeighborAwareGraphSAGE_vehicle_edge_5_128_OneCycleLR_layer5_aggradd_1_mlp_num_vehicles4_kl_weight0.3_Site__balance1.0_2025-12-11/best_model.pth \
  --pkl_path ./4_20250814graph_data/graph_data/combined_vehicle_data_4_5.pkl \
  --cache_dir ./4_20250814graph_data/graph_data/combined_vehicle_data_4_5 \
  --hidden_channels 128 \
  --num_layers 5 \
  --aggr add \
  --edge_dim 5 \
  --attention_type mlp \
  --num_vehicles 4 \
  --dataset gongeoptap \
  --graph_data_type image
```