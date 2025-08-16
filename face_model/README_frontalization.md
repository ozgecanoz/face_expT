# Face Frontalization Video Processor

This tool processes face videos by applying frontalization to each frame, creating a new video where all faces are front-facing.

## Features

- **Frame-by-frame processing**: Applies frontalization to each frame of the input video
- **Side-by-side comparison**: Option to create comparison videos (original vs frontalized)
- **Flexible frame ranges**: Process specific segments of long videos
- **Progress tracking**: Real-time progress updates during processing
- **Error handling**: Robust error handling with fallback to original frames
- **Multiple output formats**: Support for various video codecs

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_frontalization.txt
```

Or install manually:

```bash
pip install torch torchvision numpy Pillow opencv-python
```

## Usage

### Basic Usage

```bash
# Process entire video
python test_frontalization_video.py input_video.mp4

# Create side-by-side comparison
python test_frontalization_video.py input_video.mp4 --side-by-side

# Specify custom output path
python test_frontalization_video.py input_video.mp4 -o output_video.mp4
```

### Advanced Options

```bash
# Process specific frame range
python test_frontalization_video.py input_video.mp4 --start-frame 100 --end-frame 200

# Use different model file
python test_frontalization_video.py input_video.mp4 -m ./my_model.pt

# Hide progress bar
python test_frontalization_video.py input_video.mp4 --no-progress

# Combine options
python test_frontalization_video.py input_video.mp4 --side-by-side --start-frame 0 --end-frame 100 -o test_segment.mp4
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `input_video` | - | Path to input video file (required) |
| `--output` | `-o` | Path to output video file |
| `--model` | `-m` | Path to frontalization model file |
| `--start-frame` | - | Starting frame number (0-indexed) |
| `--end-frame` | - | Ending frame number |
| `--side-by-side` | `-s` | Create side-by-side comparison video |
| `--no-progress` | - | Hide progress bar |
| `--help` | `-h` | Show help message |

## Model Requirements

The frontalization model should:

- Accept input tensors of shape `[1, 3, 128, 128]` (batch, channels, height, width)
- Output tensors with values in the range `[-1, 1]`
- Be compatible with PyTorch 1.12+

## Input/Output

### Input Video
- **Format**: Any format supported by OpenCV (MP4, AVI, MOV, etc.)
- **Resolution**: Any resolution (will be resized to 128x128 for processing)
- **FPS**: Any frame rate (output maintains original FPS)

### Output Video
- **Format**: MP4 (H.264 codec)
- **Resolution**: Same as input video
- **FPS**: Same as input video
- **Content**: Each frame processed through the frontalization model

### Side-by-Side Output
- **Format**: MP4 (H.264 codec)
- **Resolution**: Double width (original + frontalized side by side)
- **FPS**: Same as input video
- **Content**: Original frame | Frontalized frame

## Examples

### Example 1: Basic Processing
```bash
python test_frontalization_video.py face_video.mp4
```
Creates: `face_video_frontalized.mp4`

### Example 2: Comparison Video
```bash
python test_frontalization_video.py face_video.mp4 --side-by-side
```
Creates: `face_video_frontalized_comparison.mp4`

### Example 3: Process Segment
```bash
python test_frontalization_video.py face_video.mp4 --start-frame 500 --end-frame 1000 --side-by-side
```
Creates: `face_video_frontalized_comparison.mp4` with frames 500-999

### Example 4: Custom Output
```bash
python test_frontalization_video.py face_video.mp4 -o my_frontalized_video.mp4
```
Creates: `my_frontalized_video.mp4`

## Performance Tips

1. **Frame Range Testing**: Use `--start-frame` and `--end-frame` to test on short segments first
2. **Model Location**: Keep the model file in the same directory or specify the full path
3. **Disk Space**: Ensure sufficient disk space for output videos (typically 2x input size for side-by-side)
4. **Processing Time**: Depends on video length, resolution, and model complexity

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `generator_v0.pt` is in the current directory or specify path with `-m`
2. **Video not opening**: Check video format compatibility with OpenCV
3. **Memory issues**: Process shorter video segments or reduce batch size
4. **Output errors**: Ensure write permissions in output directory

### Error Messages

- `❌ Model file not found`: Check model file path and existence
- `❌ Error: Could not open input video`: Check video file format and path
- `❌ Error: Could not create output video`: Check write permissions and disk space
- `⚠️ Processing interrupted by user`: User pressed Ctrl+C

## File Structure

```
face_model/
├── test_frontalization_video.py      # Main processing script
├── example_frontalization_usage.py   # Usage examples and dependency checker
├── requirements_frontalization.txt   # Python dependencies
├── README_frontalization.md         # This file
└── generator_v0.pt                  # Your frontalization model
```

## Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Video processing and computer vision
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing
- **torchvision**: PyTorch computer vision utilities

## License

This tool is part of the face expression system project. Please refer to the main project license for usage terms.
