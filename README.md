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