# MediaPipe Pose Detection with Face Keypoints

This directory contains scripts for testing MediaPipe pose detection capabilities, specifically focusing on face keypoint detection and visualization.

## Files

- `test_mediapipe_pose.py` - Main script for processing videos/images with face keypoint visualization
- `test_mediapipe_init.py` - Simple test to verify MediaPipe components can be initialized
- `requirements_mediapipe.txt` - Required dependencies

## Installation

Install the required dependencies:

```bash
pip install -r requirements_mediapipe.txt
```

### Download Required Models

Download the MediaPipe face landmarker task file:

```bash
wget -O face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Or manually download from: [MediaPipe Face Landmarker Model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker_v2_with_blendshapes.task

## Usage

### Basic Video Processing

Process an MP4 video to detect faces and visualize keypoints:

```bash
python3 test_mediapipe_pose.py --input input_video.mp4 --output output_video.mp4
```

### Image Processing

Process a single image:

```bash
python3 test_mediapipe_pose.py --input face_image.jpg --output annotated_face.png
```

### Adjusting Confidence Thresholds

Customize face detection confidence:

```bash
python3 test_mediapipe_pose.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --face-detection-confidence 0.7
```

### Using ArcFace for Similarity Calculations

Enable face similarity calculations using ArcFace:

```bash
python3 test_mediapipe_pose.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --arcface-model /path/to/arcface_model.onnx
```

### Custom Face Landmarker Task File

Use a custom MediaPipe face landmarker task file:

```bash
python3 test_mediapipe_pose.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --face-landmarker-task /path/to/custom_landmarker.task
```

## Output Format

The script creates side-by-side output:

- **Left side**: Original frame cropped to the face bounding box with ArcFace similarity score
- **Right side**: Cropped face with detailed face mesh, contours, and iris connections overlaid

### Face Mesh Visualization

The right side displays:
- **Face Mesh**: Full facial landmark connections with tessellation
- **Face Contours**: Outline of facial features (eyes, nose, mouth)
- **Iris Connections**: Detailed eye landmark connections
- **468 Landmarks**: Comprehensive facial feature detection

## Supported Input Formats

- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`

## MediaPipe Components Used

1. **Face Landmarker**: Uses MediaPipe Tasks API with face_landmarker_v2_with_blendshapes.task model
2. **Drawing Utilities**: For visualizing face mesh, contours, and iris connections
3. **468 Facial Landmarks**: Provides detailed facial feature detection

## ArcFace Integration

When the `--arcface-model` argument is provided, the script also:

1. **Extracts embeddings**: Uses ArcFace ONNX model to generate face embeddings
2. **Computes similarity**: Calculates cosine similarity between first frame and all subsequent frames
3. **Displays results**: Shows similarity scores in the upper left corner of the left frame
4. **Reference frame**: First detected face becomes the reference for all comparisons

The similarity score ranges from -1 to 1, where:
- **1.0**: Identical faces
- **0.0**: Unrelated faces  
- **-1.0**: Completely opposite faces

## Key Features

- **Real-time processing**: Can handle video streams efficiently
- **Confidence thresholds**: Adjustable detection sensitivity
- **Multi-format support**: Works with both videos and images
- **Detailed landmarks**: Uses MediaPipe Tasks API with 468 facial landmarks
- **Face mesh visualization**: Shows full face mesh, contours, and iris connections
- **ArcFace integration**: Optional face similarity calculations using ArcFace embeddings
- **Cosine similarity**: Computes similarity between first frame and all subsequent frames

## Example Use Cases

1. **Face Analysis**: Study facial expressions and movements
2. **Pose Estimation**: Track head pose and facial orientation
3. **Animation**: Extract facial keypoints for animation systems
4. **Research**: Analyze facial landmark patterns in video data
5. **Quality Assessment**: Evaluate face detection and tracking accuracy

## Troubleshooting

### Common Issues

1. **No faces detected**: Lower the confidence thresholds
2. **Poor keypoint accuracy**: Lower the face detection confidence for more sensitive detection
3. **Performance issues**: Reduce video resolution or use GPU acceleration if available

### Performance Tips

- Use appropriate confidence thresholds for your use case
- Consider processing videos at lower resolutions for faster processing
- MediaPipe automatically uses GPU acceleration when available
- The simplified approach (no face mesh) is faster and more efficient

## Dependencies

- OpenCV (cv2) >= 4.5.0
- MediaPipe >= 0.10.0  
- NumPy >= 1.19.0

## Testing

Test MediaPipe initialization:

```bash
python3 test_mediapipe_init.py
```

This will verify that all MediaPipe components can be initialized correctly.
