#!/bin/bash
# Automated setup script for Google Cloud training VM

echo "🚀 Setting up Google Cloud Training VM"
echo "======================================"

# Check if running on Google Cloud VM
if [[ ! -f /etc/google_cloud_config ]]; then
    echo "⚠️  This script is designed to run on a Google Cloud VM"
    echo "Please run this on your training VM after creation"
    exit 1
fi

echo "✅ Detected Google Cloud VM"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install package if not exists
install_if_missing() {
    if ! command_exists "$1"; then
        echo "📦 Installing $1..."
        sudo apt-get install -y "$1"
    else
        echo "✅ $1 already installed"
    fi
}

echo ""
echo "🔧 Step 1: Mounting additional disk..."
# Check if dataset disk exists
if [[ -e /dev/sdb ]]; then
    echo "📁 Found additional disk /dev/sdb"
    
    # Check if already mounted
    if ! mountpoint -q /mnt/dataset-storage; then
        echo "🔧 Mounting disk..."
        sudo mkfs.ext4 /dev/sdb
        sudo mkdir -p /mnt/dataset-storage
        sudo mount /dev/sdb /mnt/dataset-storage
        sudo chown $USER:$USER /mnt/dataset-storage
        
        # Make mount permanent
        if ! grep -q "/dev/sdb /mnt/dataset-storage" /etc/fstab; then
            echo "/dev/sdb /mnt/dataset-storage ext4 defaults 0 2" | sudo tee -a /etc/fstab
        fi
        
        echo "✅ Disk mounted successfully"
    else
        echo "✅ Disk already mounted"
    fi
else
    echo "⚠️  No additional disk found at /dev/sdb"
    echo "Creating dataset directory in home..."
    mkdir -p ~/dataset-storage
    ln -sf ~/dataset-storage /mnt/dataset-storage
fi

echo ""
echo "🔧 Step 2: Installing system dependencies..."
sudo apt-get update
sudo apt-get upgrade -y

# Install basic packages
install_if_missing python3
install_if_missing python3-pip
install_if_missing python3-venv
install_if_missing htop
install_if_missing tmux
install_if_missing git
install_if_missing wget
install_if_missing curl

# Install CUDA if GPU is available
if command_exists nvidia-smi; then
    echo "🎮 GPU detected, installing CUDA..."
    if [[ ! -d "/usr/local/cuda" ]]; then
        echo "📦 Installing CUDA toolkit..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
        sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-11-8
        echo "✅ CUDA installed"
    else
        echo "✅ CUDA already installed"
    fi
    
    # Install nvtop for GPU monitoring
    install_if_missing nvtop
else
    echo "💻 No GPU detected, skipping CUDA installation"
fi

echo ""
echo "🔧 Step 3: Setting up Python environment..."
# Create virtual environment
if [[ ! -d "face_training_env" ]]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv face_training_env
else
    echo "✅ Virtual environment already exists"
fi

# Activate environment
source face_training_env/bin/activate

echo "📦 Installing Python packages..."
# Upgrade pip
pip install --upgrade pip

# Install PyTorch
if command_exists nvidia-smi; then
    echo "🎮 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing PyTorch for CPU..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo "📦 Installing additional packages..."
pip install tensorboard matplotlib tqdm pillow
pip install google-cloud-storage google-auth
pip install requests urllib3

echo ""
echo "🔧 Step 4: Creating project structure..."
# Create project directories
mkdir -p /mnt/dataset-storage/face_model
mkdir -p /mnt/dataset-storage/datasets
mkdir -p /mnt/dataset-storage/logs
mkdir -p /mnt/dataset-storage/checkpoints

echo ""
echo "🔧 Step 5: Setting up monitoring..."
# Create monitoring script
cat > /mnt/dataset-storage/monitor_resources.sh << 'EOF'
#!/bin/bash
echo "📊 Resource Monitor"
echo "=================="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "Disk Usage:"
df -h /mnt/dataset-storage
echo ""
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
fi
EOF

chmod +x /mnt/dataset-storage/monitor_resources.sh

# Create auto-shutdown script
cat > /mnt/dataset-storage/auto_shutdown.sh << 'EOF'
#!/bin/bash
echo "Training completed, shutting down VM in 5 minutes..."
sleep 300
sudo shutdown -h now
EOF

chmod +x /mnt/dataset-storage/auto_shutdown.sh

echo ""
echo "🔧 Step 6: Setting up tmux session..."
# Create tmux configuration
cat > ~/.tmux.conf << 'EOF'
# Enable mouse support
set -g mouse on

# Set default shell
set -g default-shell /bin/bash

# Increase scrollback buffer
set -g history-limit 10000

# Enable vi mode
setw -g mode-keys vi

# Status bar
set -g status-bg black
set -g status-fg white
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your code:"
echo "   gcloud compute scp --recurse ./face_model/* face-training-vm:/mnt/dataset-storage/face_model/ --zone=us-central1-a"
echo ""
echo "2. Download datasets:"
echo "   gsutil -m cp -r gs://your-bucket/datasets/CCv2/* /mnt/dataset-storage/datasets/"
echo ""
echo "3. Start training:"
echo "   cd /mnt/dataset-storage/face_model"
echo "   source face_training_env/bin/activate"
echo "   python3 run_training.py"
echo ""
echo "4. Monitor resources:"
echo "   /mnt/dataset-storage/monitor_resources.sh"
echo ""
echo "5. Use tmux for persistent sessions:"
echo "   tmux new-session -d -s training"
echo "   tmux attach-session -t training"
echo ""
echo "🎯 Your training VM is ready!" 