#!/bin/bash
# Setup script for multithreaded batch download

echo "🔧 Setting up multithreaded batch download dependencies..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt-get update

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements_multithreaded.txt

# Install system tools for monitoring
echo "📊 Installing monitoring tools..."
sudo apt-get install -y htop iftop iotop

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x test_thread_performance.sh
chmod +x setup_multithreaded.sh

echo "✅ Setup complete!"
echo ""
echo "🚀 You can now run:"
echo "python3 batch_download_to_gcs_multithreaded.py --dataset-config CCV2_dataset_urls.json --max-workers 4" 