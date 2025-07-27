# Webcam Demo Application

Real-time face detection, token extraction, and expression prediction using trained models.

## 🎯 Features

- **Real-time Face Detection**: Uses MediaPipe for robust face detection
- **Token Extraction**: Efficiently extracts face ID and expression tokens
- **Expression Prediction**: Predicts next frame's expression token every 29 frames
- **Token Caching**: Optimized DINOv2 token extraction (single call per frame)
- **Circular Buffer**: Efficient sliding window for continuous prediction
- **Visual Feedback**: Green detection boxes and real-time status display

## 📁 File Structure

```
app_demo/
├── webcam_demo.py          # Main demo application
├── model_loader.py         # Model loading utilities
├── face_detector.py        # MediaPipe face detection
├── token_extractor.py      # Token extraction and caching
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python webcam_demo.py \
    --face_id_checkpoint /path/to/face_id_model.pth \
    --expression_transformer_checkpoint /path/to/expression_transformer.pth \
    --expression_predictor_checkpoint /path/to/expression_predictor.pth \
    --device cpu \
    --camera 0
```

### 3. Controls

- **'q'**: Quit the application
- **'r'**: Reset the token buffer
- **Console Output**: Token statistics and predictions

## 🔧 Components

### Model Loader (`model_loader.py`)
- Loads all three model checkpoints with proper architecture detection
- Handles different checkpoint formats and architectures
- Sets models to evaluation mode

### Face Detector (`face_detector.py`)
- MediaPipe-based face detection
- Extracts largest face from frame
- Draws green detection boxes
- Resizes face to 518x518 for model input

### Token Extractor (`token_extractor.py`)
- **Efficient Token Caching**: Single DINOv2 call per frame
- **Model Inference**: Uses cached tokens for all models
- **Circular Buffer**: 30-frame sliding window for predictions
- **Prediction Engine**: Uses existing `_forward_single_clip` method

### Webcam Demo (`webcam_demo.py`)
- **Real-time Processing**: Processes each frame as it comes
- **Visual Display**: Shows webcam feed with detection boxes
- **Token Logging**: Prints token statistics to console
- **Performance Tracking**: FPS monitoring and frame counting

## 📊 Processing Pipeline

### Frame Processing
```
Webcam Frame → MediaPipe Detection → Face Extraction → Token Extraction → Prediction
```

### Token Flow
```
Raw Frame (518x518x3)
    ↓
DINOv2 Tokenizer (cached)
    ↓
patch_tokens (1369, 384) + pos_embeddings (1369, 384)
    ↓
Face ID Model → face_id_token (1, 384)
    ↓
Expression Transformer → expression_token (1, 384)
    ↓
Store in Circular Buffer
```

### Prediction Flow
```
29-frame sequence (expression tokens)
    ↓
ExpressionTransformerDecoder._forward_single_clip()
    ↓
predicted_expression_token (1, 384)
```

## ⚡ Performance Optimizations

### Token Caching
- **Single DINOv2 Call**: Extract tokens once per frame
- **Model Inference**: Use cached tokens for all models
- **Memory Efficiency**: Fixed-size circular buffer

### Sliding Window
- **Circular Buffer**: 30-frame efficient storage
- **Continuous Prediction**: Every frame gets a prediction after initial 29 frames
- **Memory Constant**: No memory growth over time

### Real-time Processing
- **Frame-by-frame**: Process each frame as it comes
- **Minimal Latency**: Optimized for real-time performance
- **GPU/CPU Support**: Works on both CPU and GPU

## 🎮 Usage Examples

### Basic Usage
```bash
python webcam_demo.py \
    --face_id_checkpoint checkpoints/face_id_epoch_5.pth \
    --expression_transformer_checkpoint checkpoints/expression_transformer_epoch_5.pt \
    --expression_predictor_checkpoint checkpoints/expression_predictor_epoch_5.pt
```

### GPU Acceleration
```bash
python webcam_demo.py \
    --face_id_checkpoint checkpoints/face_id_epoch_5.pth \
    --expression_transformer_checkpoint checkpoints/expression_transformer_epoch_5.pt \
    --expression_predictor_checkpoint checkpoints/expression_predictor_epoch_5.pt \
    --device cuda
```

### Custom Settings
```bash
python webcam_demo.py \
    --face_id_checkpoint checkpoints/face_id_epoch_5.pth \
    --expression_transformer_checkpoint checkpoints/expression_transformer_epoch_5.pt \
    --expression_predictor_checkpoint checkpoints/expression_predictor_epoch_5.pt \
    --device cpu \
    --camera 1 \
    --confidence 0.7 \
    --no_print_tokens
```

## 📈 Expected Output

### Console Output
```
📊 Frame 30:
   Face ID Token: mean=0.1234, std=0.5678
   Expression Token: mean=0.2345, std=0.6789
   🔮 Prediction: mean=0.3456, std=0.7890
```

### Visual Display
- **Green Boxes**: Face detection boundaries
- **Frame Counter**: Current frame number
- **FPS Display**: Real-time performance
- **Status Text**: "COLLECTING FRAMES" / "PREDICTION READY" / "NO FACE DETECTED"

## 🔍 Troubleshooting

### Common Issues

1. **Webcam Not Found**
   ```bash
   # Try different camera index
   python webcam_demo.py --camera 1
   ```

2. **Model Loading Errors**
   - Ensure checkpoint paths are correct
   - Check model architecture compatibility
   - Verify checkpoint format

3. **Performance Issues**
   - Use `--device cpu` for CPU-only processing
   - Reduce `--confidence` threshold for faster detection
   - Use `--no_print_tokens` to reduce console output

4. **Memory Issues**
   - The demo uses fixed memory (30-frame buffer)
   - No memory growth over time
   - Automatic garbage collection

## 🎯 Key Features

- **Efficient Token Caching**: Single DINOv2 call per frame
- **Real-time Performance**: Optimized for live video processing
- **Robust Face Detection**: MediaPipe with confidence threshold
- **Continuous Prediction**: Every frame gets a prediction after initial setup
- **Visual Feedback**: Real-time status and performance metrics
- **Memory Efficient**: Fixed-size circular buffer, no memory leaks

## 🔬 Technical Details

### Model Architecture Support
- **Face ID Model**: Supports different architectures from checkpoints
- **Expression Transformer**: Compatible with various head/layer configurations
- **Expression Predictor**: Uses existing `_forward_single_clip` method

### Token Dimensions
- **DINOv2 Tokens**: 1369 patches × 384 dimensions
- **Face ID Tokens**: 1 × 384 dimensions
- **Expression Tokens**: 1 × 384 dimensions
- **Predictions**: 1 × 384 dimensions

### Performance Characteristics
- **Frame Rate**: 15-30 FPS depending on hardware
- **Memory Usage**: ~50MB for token buffers
- **Latency**: <100ms per frame processing
- **Accuracy**: Depends on model training quality 