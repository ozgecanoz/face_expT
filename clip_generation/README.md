# Clip Generation Package

This package provides utilities for analyzing videos and extracting expressive face sequences for dataset preparation.

## Overview

The clip generation package consists of three main components:

1. **VideoExpressionAnalyzer**: Analyzes videos to find sequences with high expression variation and stable face position
2. **ClipExtractor**: Extracts video clips based on analysis results
3. **generate_clips_from_video.py**: Main script that combines both components

## Features

- **Face Detection**: Uses MediaPipe for robust face detection
- **Expression Analysis**: Extracts expression tokens using pre-trained Expression Transformer
- **Sequence Scoring**: Combines expression variation and position stability metrics
- **Clip Extraction**: Extracts high-quality face sequences for training
- **Metadata Generation**: Creates detailed reports and metadata files

## Usage

### Basic Usage

```bash
python clip_generation/generate_clips_from_video.py \
    --video_path /path/to/input/video.mp4 \
    --expression_transformer_checkpoint /path/to/expression_transformer.pt \
    --output_dir ./output \
    --num_sequences 10 \
    --sequence_length 30
```

### Advanced Usage

```bash
python clip_generation/generate_clips_from_video.py \
    --video_path /path/to/input/video.mp4 \
    --expression_transformer_checkpoint /path/to/expression_transformer.pt \
    --output_dir ./output \
    --subject_id 0 \
    --sequence_length 30 \
    --num_sequences 20 \
    --expression_weight 0.8 \
    --position_weight 0.2 \
    --face_confidence_threshold 0.8 \
    --device cpu
```

## Output Structure

```
output/
├── analysis/
│   ├── frame_data.json          # Frame-by-frame analysis results
│   └── sequences.json           # Best sequences with scores
├── clips/
│   ├── clip_001_frames_000000_000029.mp4
│   ├── clip_002_frames_000045_000074.mp4
│   ├── ...
│   ├── clip_metadata.json      # Metadata for all clips
│   └── extraction_summary.txt  # Summary report
```

## Configuration Options

### Analysis Parameters
- `--sequence_length`: Length of sequences to extract (default: 30)
- `--num_sequences`: Number of sequences to extract (default: 10)
- `--expression_weight`: Weight for expression variation in scoring (default: 0.7)
- `--position_weight`: Weight for position stability in scoring (default: 0.3)
- `--face_confidence_threshold`: Minimum confidence for face detection (default: 0.7)

### Output Options
- `--output_dir`: Directory to save results (default: ./clip_generation_output)
- `--clip_format`: Format for output clips (default: mp4)
- `--skip_analysis`: Skip analysis and use existing results
- `--skip_extraction`: Skip clip extraction and only run analysis

## Scoring Algorithm

The sequence scoring combines two metrics:

1. **Expression Variation**: Measures how much facial expressions change across the sequence
2. **Position Stability**: Measures how stable the face position is across the sequence

**Combined Score** = `(expression_weight × expression_variation) + (position_weight × position_stability)`

## Use Cases

- **Training Data Preparation**: Extract expressive face sequences for model training
- **Dataset Curation**: Find the most dynamic expression sequences
- **Quality Control**: Identify sequences with stable face tracking and varied expressions

## Dependencies

- OpenCV for video processing
- MediaPipe for face detection
- PyTorch for deep learning models
- NumPy for numerical operations
- tqdm for progress bars

## Examples

### Extract 15 sequences of 45 frames each
```bash
python clip_generation/generate_clips_from_video.py \
    --video_path input_video.mp4 \
    --expression_transformer_checkpoint model.pt \
    --sequence_length 45 \
    --num_sequences 15
```

### Prioritize expression variation over position stability
```bash
python clip_generation/generate_clips_from_video.py \
    --video_path input_video.mp4 \
    --expression_transformer_checkpoint model.pt \
    --expression_weight 0.9 \
    --position_weight 0.1
```

### Use GPU for faster processing
```bash
python clip_generation/generate_clips_from_video.py \
    --video_path input_video.mp4 \
    --expression_transformer_checkpoint model.pt \
    --device cuda
``` 