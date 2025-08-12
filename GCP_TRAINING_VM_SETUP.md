# Google Cloud VM Setup for Model Training

# stop the instance from command line : much better:
gcloud compute instances stop trainer-a100-co-cuda12 --zone us-central1-a --discard-local-ssd=false
gcloud compute instances start trainer-a100-co-cuda12 --zone us-central1-a

gcloud auth list
# make sure to use right account: owhiting@eqlabsai.com
# gcloud auth login if now, this brings up the chrome browser

## myenv on the machine trainer-gpu-co
(base) ozgewhiting@trainer-gpu-co:/$ cd /home/ozgewhiting/
(base) ozgewhiting@trainer-gpu-co:~$ ls -la | grep -E "(myenv|venv|env|\.venv)"
drwxr-xr-x 6 ozgewhiting ozgewhiting 4096 Jul 30 19:17 myenv
(base) ozgewhiting@trainer-gpu-co:~$ source ./myenv/bin/activate
(myenv) (base) ozgewhiting@trainer-gpu-co:~$ 

# Download checkpoints from VM, just from the terminal on Mac:
gcloud compute scp --recurse trainer-gpu-co:/mnt/dataset-storage/face_model/checkpoints_with_keywords ./local_checkpoints/

# Download data from VM, just from the terminal on Mac:
gcloud compute scp --recurse trainer-gpu-co:/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0 /Users/ozgewhiting/Documents/projects/cloud_dbs/CCA_train_db4_no_padding_keywords_offset_1.0/

## 🎯 Recommended VM Configuration

### **For Face Model Training:**

#### **Option 1: GPU VM (Recommended for Training)**
```bash
# e2-standard-8 with GPU
Machine Type: e2-standard-8 (8 vCPUs, 32 GB memory)
GPU: 1x NVIDIA T4 or 1x NVIDIA V100
Boot Disk: 100 GB SSD
Additional Disk: 500 GB SSD for datasets
Zone: us-central1-a (good GPU availability)
```

#### **Option 2: High-Performance CPU VM (Recommended for CPU Training)**
```bash
# n2-standard-16 for CPU training
Machine Type: n2-standard-16 (16 vCPUs, 64 GB memory)
GPU: None (CPU training)
Boot Disk: 100 GB SSD
Additional Disk: 500 GB SSD for datasets
Zone: us-central1-a
```

## 🚀 Quick Setup Commands

### **1. Create VM with GPU:**
```bash
gcloud compute instances create face-training-vm \
  --machine-type=e2-standard-8 \
  --zone=us-central1-a \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --image-family=debian-11-gpu \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --create-disk=name=dataset-disk,size=500GB,type=pd-ssd \
  --metadata=install-nvidia-driver=True
```

### **2. Create High-Performance CPU VM (n2-standard-16):**
```bash
gcloud compute instances create face-training-vm-cpu \
  --machine-type=n2-standard-16 \
  --zone=us-central1-a \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --create-disk=name=dataset-disk,size=500GB,type=pd-ssd
```

## 🔧 VM Setup Script

### **1. Connect to VM:**
```bash
gcloud compute ssh face-training-vm --zone=us-central1-a
## for eqlabs cloud account
gcloud compute ssh --zone "us-central1-c" "trainer-gpu-co" --project "facedemo-467418"
```

### **2. Mount Additional Disk:** # didn't do this
```bash
# Format and mount the dataset disk
lsblk  # command to see the discs (I see as 500GB) --> see it as nvme0n2 
sudo mkfs.ext4 /dev/nvme0n2
sudo mkdir /mnt/dataset-storage
sudo mount /dev/nvme0n2 /mnt/dataset-storage
sudo chown $USER:$USER /mnt/dataset-storage

# Make mount permanent
echo "/dev/nvme0n2 /mnt/dataset-storage ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

### **3. Install Dependencies:**
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Install CUDA (for GPU VMs only)
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA already installed"
else
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-11-8
fi
(base) ozgewhiting@trainer-gpu-co:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0

 nvidia-smi --> shows the GPU


# Install monitoring tools
sudo apt-get install -y htop tmux
```

### **4. Setup Python Environment:**
```bash
# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install PyTorch (CPU version for CPU VMs, GPU version for GPU VMs)
# dino v2 base required torch 2.1 and torch 2 required min cuda 12
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Install other dependencies
pip install tensorboard matplotlib tqdm pillow
pip install google-cloud-storage google-auth
```

## 📁 Project Setup

### **1. Clone/Upload Your Code:**
```bash
# Create project directory
mkdir -p /mnt/dataset-storage/face_model
cd /mnt/dataset-storage/face_model   ## or just stay in dataset-storage

# Upload your code (from local machine)
#gcloud compute scp --recurse ./face_model/* face-training-vm:/mnt/dataset-storage/face_model/ --zone=us-central1-a
# get it from git
git clone git@github.com:ozgecanoz/dataset_utils.git dataset_utils

```
### upload key to bucket from local to VM
gcloud compute scp --recurse /Users/ozgewhiting/Documents/projects/dataset_utils/dataset-uploader-key.json trainer-a100-co-cuda12:/mnt/dataset-storage/ --zone=us-central1-a

gcloud auth activate-service-account --key-file=/path/to/your/service-account-key.json

### **2. Download Datasets:**
```bash
# Download CCv2 datasets from GCS
gsutil -m cp -r gs://your-bucket/datasets/CCv2/* /mnt/dataset-storage/datasets/
gsutil -m cp -r gs://face-training-datasets/CCA_train_db4_no_padding_keywords_offset_1.0/* /mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0/

## use with a service key:

gsutil -m cp -r gs://face-training-datasets/CCA_train_db2/* /mnt/dataset-storage/dbs/CCA_train_db2/

# Or use your multithreaded download script
python3 batch_download_to_gcs_multithreaded.py --dataset-config CCV2_dataset_urls.json --max-workers 4
```

## 🎯 Training Configuration

### **1. GPU Training Config:**
```python
# face_model/run_training.py
config = {
    'training': {
        'batch_size': 8,  # Larger with GPU
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cuda'  # Use GPU
    },
    'face_id_model': {
        'embed_dim': 384,
        'num_heads': 8,  # Can use more with GPU
        'num_layers': 2,
        'dropout': 0.1
    }
}
```

### **2. High-Performance CPU Training Config (n2-standard-16):**
```python
# face_model/run_training.py
config = {
    'training': {
        'batch_size': 12,  # Larger with 64GB RAM
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cpu',
        'num_workers': 8  # More workers with 16 vCPUs
    },
    'face_id_model': {
        'embed_dim': 384,
        'num_heads': 6,  # Can use more with 16 vCPUs
        'num_layers': 2,
        'dropout': 0.1
    }
}
```

### **3. Standard CPU Training Config (e2-standard-4):**
```python
# face_model/run_training.py
config = {
    'training': {
        'batch_size': 2,  # Smaller for CPU
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cpu'
    },
    'face_id_model': {
        'embed_dim': 384,
        'num_heads': 4,  # Reduced for CPU
        'num_layers': 1,
        'dropout': 0.1
    }
}
```

## 📊 n2-standard-16 Performance Expectations

### **Expected Performance:**
- **Batch size**: 12-16 (vs 2-4 on smaller instances)
- **Training speed**: 3-4x faster than e2-standard-4
- **Memory utilization**: 40-60GB during training
- **CPU utilization**: 80-90% across all 16 cores
- **Data loading**: 8-10 workers for parallel processing

### **Cost Analysis:**
- **Hourly cost**: ~$0.38/hour (vs $0.50-1.00/hour for GPU)
- **Training time**: 5-10x longer than GPU, but much cheaper
- **Total cost**: Similar or cheaper than GPU for long training runs

## 🔍 Monitoring and Management

### **1. Start Training with tmux:**
```bash
# Create persistent session
tmux new-session -d -s training

# Attach to session
tmux attach-session -t training

# Start training
cd /mnt/dataset-storage/face_model
source face_training_env/bin/activate
python3 run_training.py

# Detach from session (Ctrl+B, then D)
# Reattach later: tmux attach-session -t training
```

### **2. Monitor Resources:**
```bash
# Monitor GPU usage (for GPU VMs)
nvidia-smi

# Monitor system resources
htop

# Monitor disk usage
df -h /mnt/dataset-storage

# Monitor CPU usage specifically
top -bn1 | grep "Cpu(s)"

# Monitor memory usage
free -h
```

### **3. TensorBoard:**
```bash
# Start TensorBoard
tensorboard --logdir=/mnt/dataset-storage/face_model/logs --port=8080

# Access via browser (set up port forwarding)
gcloud compute ssh face-training-vm --zone=us-central1-a -- -L 8080:localhost:8080
```

## 💰 Cost Optimization

### **1. Spot Instances (Save 60-80%):**
```bash
gcloud compute instances create face-training-vm-spot \
  --machine-type=n2-standard-16 \
  --zone=us-central1-a \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --create-disk=name=dataset-disk,size=500GB,type=pd-ssd \
  --preemptible
```

### **2. Auto-shutdown Script:**
```bash
# Create auto-shutdown script
cat > /mnt/dataset-storage/auto_shutdown.sh << 'EOF'
#!/bin/bash
# Shutdown VM after training completes
echo "Training completed, shutting down VM in 5 minutes..."
sleep 300
sudo shutdown -h now
EOF

chmod +x /mnt/dataset-storage/auto_shutdown.sh
```

## 🚀 Quick Start Commands

### **Complete Setup (run on VM):**
```bash
# 1. Mount disk
sudo mkfs.ext4 /dev/sdb
sudo mkdir /mnt/dataset-storage
sudo mount /dev/sdb /mnt/dataset-storage
sudo chown $USER:$USER /mnt/dataset-storage

# 2. Setup environment
python3 -m venv face_training_env
source face_training_env/bin/activate
pip install torch torchvision torchaudio
pip install tensorboard matplotlib tqdm pillow google-cloud-storage

# 3. Download datasets
gsutil -m cp -r gs://your-bucket/datasets/CCv2/* /mnt/dataset-storage/datasets/

# 4. Start training
cd /mnt/dataset-storage/face_model
python3 run_training.py
```

## ⚠️ Important Notes

1. **n2-standard-16 is excellent** for CPU training (~$0.38/hour)
2. **Use spot instances** for cost savings
3. **Monitor usage** with `htop` and `top`
4. **Save checkpoints** frequently to GCS
5. **Use tmux** for persistent sessions
6. **Set up auto-shutdown** to avoid unnecessary costs
7. **No CUDA needed** for CPU-only training

This setup will give you a powerful environment for training your face model! 🎯 