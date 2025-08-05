# Face Expression System & Dataset Utils

A comprehensive system for face expression analysis, reconstruction, and dataset processing with deep learning models.

## Overview

This repository contains a complete face expression system with the following components:

- **Expression Transformer**: Extracts expression tokens from face images using subject embeddings
- **Expression Reconstruction Model**: Reconstructs face images from expression tokens and subject embeddings  
- **Joint Training System**: Combined training of expression extraction and reconstruction
- **Dataset Processing**: Cloud-based dataset serialization and video processing utilities
- **Demo Applications**: Webcam demos and video generation tools

## Key Features

### 🤖 Face Model Training System

#### Expression Transformer
- **Subject Embeddings**: Learnable embeddings for identity representation
- **DINOv2 Integration**: Uses DINOv2 tokenization for patch extraction
- **Delta Positional Embeddings**: Learnable positional embeddings for reconstruction
- **Multi-head Attention**: Configurable transformer architecture

#### Expression Reconstruction Model  
- **Cross-Attention**: Uses `nn.TransformerDecoder` for cross-attention
- **Self-Attention**: Uses `nn.TransformerEncoder` for self-attention
- **Subject Integration**: Takes subject embeddings and expression tokens
- **Positional Embeddings**: Supports delta embeddings from expression transformer

#### Joint Training System
- **Combined Loss**: Reconstruction, temporal, and diversity losses
- **Dynamic Weight Scheduling**: LR scheduler with loss weight scheduling
- **Freeze Options**: Can freeze expression transformer for fine-tuning
- **Comprehensive Checkpointing**: Saves joint and individual model checkpoints

### 📊 Dataset Processing

#### Cloud Dataset Serialization
- **Keyword-based Clips**: Extract clips based on keyword timestamps
- **Random Clip Generation**: Generate random clips for training
- **Multi-threaded Processing**: Efficient cloud-based processing
- **Progress Tracking**: Robust progress bars and logging

#### Video Processing Utilities
- **Frame Extraction**: Extract frames at specific rates
- **Face Detection**: MediaPipe-based face detection and cropping
- **Audio Processing**: Extract and process audio tracks
- **Batch Processing**: Process multiple videos efficiently

### 🎥 Demo Applications

#### Webcam Demo
- **Real-time Processing**: Live face expression analysis
- **Model Loading**: Loads trained models from checkpoints
- **Token Extraction**: Extracts expression tokens from live video

#### Video Generation
- **Reconstruction Videos**: Side-by-side original vs reconstructed
- **Identity Swap Videos**: Swap identities while preserving expressions
- **Token Visualization**: Visualize expression token extraction

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ozgecanoz/dataset_utils.git
cd dataset_utils

# Install dependencies
pip install -r requirements.txt

# For GPU training (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Training Models

#### Expression Prediction Training
```bash
cd face_model
python run_expression_prediction_training.py
```

#### Joint Expression & Reconstruction Training
```bash
cd face_model
python run_expression_and_reconstruction_training.py
```

#### Frozen Training (Fine-tuning)
```python
# In run_expression_and_reconstruction_training.py
'freeze_expression_transformer': True  # Freeze expression transformer
```

### Running Demos

#### Webcam Demo
```bash
cd app_demo
python webcam_demo.py
```

#### Generate Reconstruction Video
```bash
cd app_demo
python generate_reconstruction_video.py \
    --input_video path/to/video.mp4 \
    --expression_transformer_checkpoint path/to/expr_transformer.pt \
    --expression_reconstruction_checkpoint path/to/expr_reconstruction.pt \
    --output_video path/to/output.mp4
```

#### Generate Identity Swap Video
```bash
cd app_demo
python generate_identity_swap_video.py \
    --input_video path/to/video.mp4 \
    --target_subject_id 123 \
    --expression_transformer_checkpoint path/to/expr_transformer.pt \
    --expression_reconstruction_checkpoint path/to/expr_reconstruction.pt \
    --output_video path/to/output.mp4
```

## Architecture

### Model Components

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Face Image    │───▶│ Expression          │───▶│ Expression          │
│   (518x518x3)   │    │ Transformer        │    │ Reconstruction      │
│                 │    │ (Subject Embedding) │    │ Model              │
└─────────────────┘    └─────────────────────┘    └─────────────────────┘
                              │                           │
                              ▼                           ▼
                       ┌─────────────┐           ┌─────────────┐
                       │ Expression  │           │ Reconstructed│
                       │ Tokens      │           │ Face Image   │
                       │ (384 dim)   │           │ (518x518x3)  │
                       └─────────────┘           └─────────────┘
```

### Training Pipeline

1. **Data Preparation**: Extract face clips from videos
2. **Tokenization**: DINOv2 patch extraction and normalization
3. **Expression Extraction**: Expression transformer with subject embeddings
4. **Reconstruction**: Reconstruct faces from tokens and embeddings
5. **Loss Computation**: Reconstruction, temporal, and diversity losses
6. **Weight Scheduling**: Dynamic loss weight and learning rate scheduling

## Configuration

### Model Architecture
```python
# Expression Transformer
expr_embed_dim = 384
expr_num_heads = 4
expr_num_layers = 4
expr_ff_dim = 768
expr_max_subjects = 501

# Expression Reconstruction
recon_embed_dim = 384
recon_num_cross_layers = 2
recon_num_self_layers = 2
recon_num_heads = 4
recon_ff_dim = 768
```

### Training Parameters
```python
# Loss Weights
initial_lambda_reconstruction = 0.01
initial_lambda_temporal = 0.4
initial_lambda_diversity = 0.3

# Scheduler
warmup_steps = 3000
learning_rate = 5e-5
min_lr = 1e-6
```

## File Structure

```
dataset_utils/
├── face_model/                    # Face model training system
│   ├── training/                  # Training scripts
│   │   ├── train_expression_prediction.py
│   │   ├── train_expression_and_reconstruction.py
│   │   └── train_face_id.py
│   ├── models/                    # Model architectures
│   │   ├── expression_transformer.py
│   │   ├── expression_reconstruction_model.py
│   │   └── dinov2_tokenizer.py
│   ├── utils/                     # Utilities
│   │   ├── checkpoint_utils.py
│   │   ├── scheduler_utils.py
│   │   └── visualization_utils.py
│   └── run_*.py                   # Training runners
├── app_demo/                      # Demo applications
│   ├── webcam_demo.py
│   ├── generate_reconstruction_video.py
│   ├── generate_identity_swap_video.py
│   └── model_loader.py
├── clip_generation/               # Clip generation utilities
├── batch_download_to_gcs*.py      # Cloud download scripts
├── serialize_dataset*.py          # Dataset serialization
├── mp4_utils.py                   # Video processing utilities
└── requirements.txt
```

## Key Design Decisions

### Subject Embeddings vs Face ID Tokens
- **Switched to Subject Embeddings**: More flexible than face ID tokens
- **Learnable Representations**: Can be fine-tuned during training
- **Scalable**: Supports thousands of subjects

### Delta Positional Embeddings
- **Learnable Offsets**: Added to DINOv2 positional embeddings
- **Reconstruction Influence**: Affected by reconstruction loss
- **Consistent Architecture**: Used across training and inference

### Joint Training Strategy
- **Combined Loss**: Multiple loss components with dynamic weights
- **Curriculum Learning**: Weight scheduling with warmup and decay
- **Checkpoint Flexibility**: Can load joint or individual checkpoints

## Cloud Training

### GCP Setup
- **VM Configuration**: Optimized for GPU training
- **Data Storage**: GCS bucket for datasets and checkpoints
- **Multi-threading**: Efficient data loading and processing

### Dataset Processing
- **Keyword-based**: Extract clips based on keyword timestamps
- **Random Generation**: Generate random clips for training
- **Progress Tracking**: Robust progress bars and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Context for Development

**Key Design Summary**: Face expression system with subject embeddings, DINOv2 tokenization, and joint training of expression transformer and reconstruction model. Uses delta positional embeddings and dynamic loss weight scheduling.

**Current Focus**: Training pipeline optimization, demo applications, and cloud-based dataset processing.

**Best Practices**: 
- Start with context summary for new requests
- Reference specific file names
- Keep requests focused on specific changes
- Mention relevant design decisions