# Dataset Utils

A collection of utilities for working with datasets and media files.

## Description

This repository contains various tools and utilities for dataset manipulation, processing, and analysis, including MP4 video processing utilities.

## Features

### MP4 Utilities (`mp4_utils.py`)

A comprehensive module for extracting frames and audio from MP4 video files:

- **Frame Extraction**: Extract frames at specific rates or timestamps
- **Audio Extraction**: Extract audio tracks from videos in various formats
- **Video Information**: Get detailed metadata about video files
- **Thumbnail Generation**: Create thumbnail images from videos
- **Batch Processing**: Process multiple video files at once

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ozgecanoz/dataset_utils.git
cd dataset_utils
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### MP4 Utilities

```python
from mp4_utils import extract_frames, extract_audio, get_video_info

# Get video information
info = get_video_info("video.mp4")
print(f"Duration: {info['duration_seconds']} seconds")

# Extract frames (1 frame per second)
frames = extract_frames("video.mp4", frame_rate=1)

# Extract audio
audio_file = extract_audio("video.mp4", audio_format="mp3")
```

### Example Usage

Run the example script to see all functions in action:
```bash
python example_usage.py
```

## Dependencies

- `opencv-python`: Video processing and frame extraction
- `moviepy`: Audio extraction and video manipulation
- `numpy`: Numerical operations
- `Pillow`: Image processing

## License

[Add your license information here] 

## Cursor Prompts
After a restart, start with:
"Context: Face expression system with subject embeddings. Working on serialize_dataset_cloud.py progress bar and mp4_utils.py logging reduction. Current issue: tqdm last_print_t errors resolved by simplifying progress bar."

Key Design Summary:
We're building a face expression system with Components A (Face ID), C (Expression Transformer), D (Expression Predictor), and E (Face Reconstruction). We switched from face ID tokens to learnable subject embeddings. We're using DINOv2 tokenization, MediaPipe face detection, and training on cloud VMs. Current focus is on dataset serialization with keyword-based and random clip generation modes.

Model architecture: 
ExpressionTransformer uses subject embeddings instead of face ID tokens, FaceReconstructionModel takes subject embeddings directly, and we have a webcam demo that loads these models from checkpoints.

File Structure Reference:
Key files: serialize_dataset_cloud.py (main cloud processing), mp4_utils.py (face extraction), face_model/ (training scripts). 

We're working on reducing mp4_utils logging output.

"Current task: Fixing tqdm progress bar issues in serialize_dataset_cloud.py and reducing verbose logging in mp4_utils.py. We just simplified the progress bar to avoid last_print_t errors."

Best Practices for Speed:
Start with a 1-2 sentence context summary before asking new questions
Reference specific file names rather than asking me to read them
Use "remind me of..." instead of "read this file..."
Keep new requests focused on one specific change
Mention key design decisions that are relevant to your request

When I'm Getting Slow:
Restart the conversation with a brief design summary
Ask for specific file sections rather than entire files
Use "continue from where we left off" with key context points
The key is giving me the essential context without overwhelming me with full file contents. A few sentences about the current design state and what you're working on is usually enough to keep me fast and accurate.