#!/bin/bash
# Start Face ID Model training on n2-standard-16 VM

echo "🚀 Starting Face ID Model Training"
echo "=================================="

# Check if we're in the right directory
if [[ ! -d "face_model" ]]; then
    echo "❌ face_model directory not found!"
    echo "Please run this script from the dataset_utils directory"
    exit 1
fi

# Check if dataset exists
if [[ ! -d "/mnt/dataset-storage/dbs/CCA_train_db1" ]]; then
    echo "❌ Dataset not found at /mnt/dataset-storage/dbs/CCA_train_db1"
    echo "Please download the dataset first:"
    echo "gsutil -m cp -r gs://face-training-datasets/CCA_train_db1/* /mnt/dataset-storage/dbs/CCA_train_db1/"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d "face_training_env" ]]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup script first"
    exit 1
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source face_training_env/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import torch, tensorboard, matplotlib, tqdm, pillow, google.cloud.storage" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "❌ Missing dependencies. Installing..."
    pip install torch torchvision torchaudio
    pip install tensorboard matplotlib tqdm pillow google-cloud-storage google-auth
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p /mnt/dataset-storage/face_model/logs
mkdir -p /mnt/dataset-storage/face_model/checkpoints

# Check dataset size
echo "📊 Checking dataset size..."
dataset_count=$(find /mnt/dataset-storage/dbs/CCA_train_db1 -name "*.h5" | wc -l)
echo "Found $dataset_count HDF5 files in dataset"

# Start training
echo "🎯 Starting training..."
cd face_model
python3 run_training.py

echo "✅ Training script completed!" 