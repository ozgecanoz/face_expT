# Linear Probing with DINOv2 Patch Features

This folder contains utilities for analyzing DINOv2 patch features using Principal Component Analysis (PCA) and visualizing the results.

## Overview

The utilities in this folder allow you to:
1. **Extract patch features** from face images using pre-trained DINOv2 models
2. **Compute PCA directions** from large collections of patch features
3. **Visualize PCA projections** as RGB images and videos
4. **Support multiple DINOv2 models** including Hugging Face models

## Files

### Core Utilities

- **`compute_pca_directions.py`**: Main script for collecting patch features and computing PCA
- **`visualize_pca_features.py`**: Script for creating PCA visualizations from video clips
- **`README.md`**: This documentation file

### Supported Models

The utilities support three DINOv2 model variants:

1. **`vit_small_patch14_dinov2.lvd142m`** (default)
   - Input: 518×518×3 images
   - Output: 1,369 patch tokens (37×37 grid)
   - Embedding dimension: 384
   - Source: timm library

2. **`vit_small_patch16_224.augreg_in21k`**
   - Input: 224×224×3 images
   - Output: 196 patch tokens (14×14 grid)
   - Embedding dimension: 384
   - Source: timm library

3. **`facebook/dinov2-base`** (new!)
   - Input: 518×518×3 images
   - Output: 1,369 patch tokens (37×37 grid)
   - Embedding dimension: 768
   - Source: Hugging Face Transformers

## Usage

### Computing PCA Directions

```bash
# Using default model (518x518, 384-dim)
python compute_pca_directions.py \
    --dataset_path /path/to/dataset \
    --output_path pca_directions.json \
    --image_size 518

# Using 224x224 model
python compute_pca_directions.py \
    --dataset_path /path/to/dataset \
    --output_path pca_directions_224.json \
    --image_size 224

# Using Hugging Face base model (518x518, 768-dim)
python compute_pca_directions.py \
    --dataset_path /path/to/dataset \
    --output_path pca_directions_base.json \
    --image_size 518 \
    --model_name dinov2-base
```

### Visualizing PCA Features

```bash
# Visualize with 518x518 model
python visualize_pca_features.py \
    --video_path /path/to/video.mp4 \
    --pca_path pca_directions.json \
    --output_path visualization.mp4

# Visualize with 224x224 model
python visualize_pca_features.py \
    --video_path /path/to/video.mp4 \
    --pca_path pca_directions_224.json \
    --output_path visualization_224.mp4

# Visualize with base model
python visualize_pca_features.py \
    --video_path /path/to/video.mp4 \
    --pca_path pca_directions_base.json \
    --output_path visualization_base.mp4
```

### Visualizing Different PCA Component Ranges

You can now visualize different groups of PCA components using the `--start_component` parameter:

```bash
# Visualize first 3 components (0, 1, 2) - Red, Green, Blue
python visualize_pca_features.py \
    --video_path video.mp4 \
    --pca_path pca_directions.json \
    --output_path visualization_components_0_2.mp4 \
    --start_component 0 \
    --n_components 3

# Visualize next 3 components (3, 4, 5) - Red, Green, Blue
python visualize_pca_features.py \
    --video_path video.mp4 \
    --pca_path pca_directions.json \
    --output_path visualization_components_3_5.mp4 \
    --start_component 3 \
    --n_components 3

# Visualize components 6, 7, 8
python visualize_pca_features.py \
    --video_path video.mp4 \
    --pca_path pca_directions.json \
    --output_path visualization_components_6_8.mp4 \
    --start_component 6 \
    --n_components 3

# Visualize only 2 components starting from component 1
python visualize_pca_features.py \
    --video_path video.mp4 \
    --pca_path pca_directions.json \
    --output_path visualization_components_1_2.mp4 \
    --start_component 1 \
    --n_components 2
```

### Understanding Component Ranges

- **Components 0-2**: First three principal components (highest variance)
- **Components 3-5**: Fourth through sixth principal components (medium variance)
- **Components 6-8**: Seventh through ninth principal components (lower variance)

Each component represents different patterns in the data:
- **Early components**: Capture major variations (face structure, lighting)
- **Middle components**: Capture medium variations (expression details, texture)
- **Later components**: Capture fine details and noise

## Configuration

### Model Selection

The utilities automatically detect the model type based on:
- **`image_size`**: 224 or 518
- **`model_name`**: Specific model identifier

For the Hugging Face base model, use:
- **`image_size=518`** and **`model_name=dinov2-base`**

### Output Configuration

PCA results are saved with comprehensive metadata:
```json
{
  "configuration": {
    "image_size": 518,
    "model_name": "facebook/dinov2-base",
    "embed_dim": 768,
    "patches_per_frame": 1369,
    "grid_size": 37,
    "patch_size": 14,
    "total_patches": 10000,
    "unique_subjects": 50,
    "unique_clips": 200,
    "pca_components": 10,
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

## Dependencies

### Required
- `torch` >= 1.9.0
- `opencv-python` >= 4.5.0
- `numpy` >= 1.21.0
- `scikit-learn` >= 1.0.0
- `matplotlib` >= 3.5.0

### Optional
- `transformers` >= 4.20.0 (for Hugging Face models)
- `timm` >= 0.6.0 (for timm models)

## Workflow

1. **Data Collection**: Run `compute_pca_directions.py` on your dataset
2. **PCA Computation**: Automatically computes PCA directions and saves results
3. **Visualization**: Use `visualize_pca_features.py` to create visualizations
4. **Analysis**: Examine the RGB visualizations to understand feature patterns

## Understanding the Visualizations

### RGB Mapping
- **Red Channel**: First principal component
- **Green Channel**: Second principal component  
- **Blue Channel**: Third principal component

### Patch Grid
- Each small square represents one image patch
- Color intensity indicates feature activation strength
- Patterns reveal what each patch "sees" in the image

## Analysis Examples

### Face Features
- **Eyes**: Often show strong activation in specific patches
- **Mouth**: Expression-related features appear in mouth region
- **Background**: Minimal activation in background patches

### Temporal Patterns
- **Expression changes**: Visible as color variations across frames
- **Head movement**: Shows as spatial shifts in activation patterns
- **Lighting changes**: Affects overall activation intensity

## Tips

1. **Model Selection**: Use base model for richer features (768-dim vs 384-dim)
2. **Image Resolution**: Higher resolution (518x518) provides more detailed analysis
3. **Batch Size**: Adjust based on available memory
4. **Sample Size**: More samples = better PCA directions
5. **Visualization**: Side-by-side videos help compare input vs. features

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use CPU
2. **Model Loading**: Ensure correct model name and dependencies
3. **Dimension Mismatch**: Check that PCA file matches tokenizer configuration
4. **Import Errors**: Verify Python path and dependencies

### Performance Optimization

1. **GPU Usage**: Use CUDA for faster processing
2. **Batch Processing**: Larger batches improve throughput
3. **Worker Count**: Adjust based on CPU cores and memory
4. **Model Caching**: Models are cached after first load

## Advanced Usage

### Custom Model Integration

To add support for new DINOv2 models:

1. Create a new tokenizer class in `models/dinov2_tokenizer.py`
2. Implement the required interface methods
3. Update the model selection logic in PCA utilities
4. Add appropriate tests

### Batch Processing

For large datasets:
```bash
# Process in chunks
python compute_pca_directions.py --max_samples 1000
python compute_pca_directions.py --max_samples 1000 --output_path pca_chunk2.json
# Combine results manually or extend the script
```

### Custom Visualizations

Modify `create_patch_visualization_frame()` to create different visualization styles:
- Different color mappings
- Alternative patch arrangements
- Additional metadata overlays 

### Cloud A100 Deployment

The script is optimized for cloud GPU deployment with automatic memory management:

#### **GPU Memory Optimization**
- **Automatic batch size detection**: Optimizes batch size based on available GPU memory
- **Batch CPU transfers**: Accumulates features on GPU, transfers to CPU in batches
- **Memory monitoring**: Tracks GPU memory usage throughout processing
- **OOM recovery**: Automatically handles out-of-memory errors with cache clearing

#### **Recommended A100 Configuration**
```bash
# For A100 (80GB VRAM) - automatic optimization
python compute_pca_directions.py \
    --dataset_path /path/to/dataset \
    --output_path pca_directions_dinov2_base.json \
    --max_samples 1000 \
    --device auto \
    --image_size 518 \
    --model_name dinov2-base

# Manual configuration (if needed)
python compute_pca_directions.py \
    --dataset_path /path/to/dataset \
    --output_path pca_directions_dinov2_base.json \
    --max_samples 1000 \
    --batch_size 16 \
    --device cuda \
    --num_workers 8 \
    --image_size 518 \
    --model_name dinov2-base
```

#### **Performance Expectations**
- **DINOv2 base model**: ~86.6M parameters, loads to GPU
- **1000 clips × ~10 frames × 1369 patches × 768 dims** = ~10.5M patch features
- **Memory usage**: ~2-4GB GPU memory for model + features
- **Processing time**: ~30-60 minutes on A100 (depending on clip lengths)
- **Auto-optimization**: Automatically selects optimal batch size and worker count

#### **Memory Management Features**
- **Smart batching**: Features accumulated on GPU, transferred to CPU every 100 samples
- **Progress monitoring**: Memory usage logged every 10 batches
- **Automatic recovery**: Handles OOM errors with cache clearing and retry
- **Memory cleanup**: Automatic cleanup after each stage 