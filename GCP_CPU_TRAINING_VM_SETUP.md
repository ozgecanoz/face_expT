# Google Cloud CPU VM Setup for Model Training

## 🎯 VM Configuration

### **High-Performance CPU VM for Training:**
```bash
# n2-standard-16 for CPU training
Machine Type: n2-standard-16 (16 vCPUs, 64 GB memory)
GPU: None (CPU training)
Boot Disk: 100 GB SSD
Additional Disk: 500 GB SSD for datasets
Zone: us-central1-b
VM Name: trainer-cpu-vm
```

## 🚀 Quick Setup Commands

### **Create High-Performance CPU VM:**
```bash
gcloud compute instances create trainer-cpu-vm \
  --machine-type=n2-standard-16 \
  --zone=us-central1-b \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --create-disk=name=dataset-disk,size=500GB,type=pd-ssd
```

## 🔧 VM Setup Script

### **1. Connect to VM:**
```bash
gcloud compute ssh trainer-cpu-vm --zone=us-central1-b
```

### **2. Mount Fresh 500GB Disk:**
```bash
lsblk  # command to see the discs (I see sdb as 500GB)
# Format and mount the fresh dataset disk
sudo mkfs.ext4 /dev/sdb
sudo mkdir /mnt/dataset-storage
sudo mount /dev/sdb /mnt/dataset-storage
sudo chown $USER:$USER /mnt/dataset-storage

# Make mount permanent
echo "/dev/sdb /mnt/dataset-storage ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

### **3. Install Dependencies:**
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Install Git and SSH tools
sudo apt-get install -y git openssh-client

# Install monitoring tools
sudo apt-get install -y htop tmux
```

### **4. Setup SSH Key for GitHub:**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "ozgecanozcanli@gmail.com" -f ~/.ssh/github_key

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to SSH agent
ssh-add ~/.ssh/github_key

# Display public key to copy to GitHub
cat ~/.ssh/github_key.pub
```

### **5. Add SSH Key to GitHub:**
```bash
# Copy the output from the previous command and add it to GitHub:
# 1. Go to GitHub.com → Settings → SSH and GPG keys
# 2. Click "New SSH key"
# 3. Paste the public key content
# 4. Give it a title like "trainer-cpu-vm"
# 5. Click "Add SSH key"
```

### **6. Test GitHub Connection:**
```bash
# Test SSH connection to GitHub
ssh -T git@github.com

# You should see: "Hi username! You've successfully authenticated..."
```

### **7. Setup Python Environment:**
```bash
# Create virtual environment
python3 -m venv face_training_env
source face_training_env/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install tensorboard matplotlib tqdm pillow
pip install google-cloud-storage google-auth

# Install requirements from dataset_utils repo
pip install -r requirements.txt
```

## 📁 Project Setup

### **1. Clone Code from GitHub:**
```bash
# Create project directory
mkdir -p /mnt/dataset-storage/face_model
cd /mnt/dataset-storage/face_model

# Clone your repository
git clone git@github.com:ozgecanoz/dataset_utils.git .

# Or if you want to clone to a specific directory:
git clone git@github.com:ozgecanoz/dataset_utils.git dataset_utils
```


### **2. Download Datasets:**
```bash
# Download CCv2 datasets from GCS
# List the contents of your bucket to see the correct path
gsutil ls gs://face-training-datasets/
gsutil -m cp -r gs://your-bucket/datasets/CCv2/* /mnt/dataset-storage/datasets/
gsutil -m cp -r gs://face-training-datasets/CCA_train_db1/* ./dbs/CCA_train_db1/ 

# Or use your multithreaded download script
python3 batch_download_to_gcs_multithreaded.py --dataset-config CCV2_dataset_urls.json --max-workers 4
```
### **2. Download the model checkpoints after training to local:** 
gcloud compute scp trainer-cpu-vm:/mnt/dataset-storage/face_model/checkpoints/face_id_epoch_0.pth ./Documents/projects/dataset_utils/cloud_checkpoints/ --zone=us-central1-b


## 🎯 Training Configuration

### **High-Performance CPU Training Config (n2-standard-16):**
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
# Create persistent session --> protects from ssh disconnect. training keeps running on the machine
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
�� Manual TensorBoard Setup:
Step 1: Copy Logs from VM to Local
Use gcloud compute scp to copy log directories
Copy from: /mnt/dataset-storage/face_model/logs/ on VM
Copy to: Your local machine
Frequency: Every few epochs or when you want to check progress
Step 2: Run TensorBoard Locally
Start TensorBoard on your local machine
Point to the copied log directory
Access via: http://localhost:6006 in your browser
Real-time updates: Refresh browser to see new data

## 💰 Cost Optimization

### **1. Spot Instances (Save 60-80%):**
```bash
gcloud compute instances create trainer-cpu-vm-spot \
  --machine-type=n2-standard-16 \
  --zone=us-central1-b \
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
# 1. Mount fresh disk
sudo mkfs.ext4 /dev/sdb
sudo mkdir /mnt/dataset-storage
sudo mount /dev/sdb /mnt/dataset-storage
sudo chown $USER:$USER /mnt/dataset-storage

# 2. Setup SSH key for GitHub
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/github_key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_key
cat ~/.ssh/github_key.pub
# Copy this output and add to GitHub SSH keys

# 3. Test GitHub connection
ssh -T git@github.com

# 4. Setup environment
python3 -m venv face_training_env
source face_training_env/bin/activate
pip install torch torchvision torchaudio
pip install tensorboard matplotlib tqdm pillow google-cloud-storage

# 5. Install requirements from dataset_utils repo
pip install -r requirements.txt

# 5. Clone code from GitHub
cd /mnt/dataset-storage
git clone git@github.com:ozgecanoz/dataset_utils.git face_model

# 6. Install requirements from cloned repo
cd face_model
pip install -r requirements.txt

# 7. Download datasets
gsutil -m cp -r gs://your-bucket/datasets/CCv2/* /mnt/dataset-storage/datasets/

# 8. Start training
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
8. **Fresh disk mounting** - the 500GB disk needs to be formatted and mounted
9. **GitHub SSH key** - Remember to add the public key to your GitHub account

This setup will give you a powerful environment for training your face model! 🎯 