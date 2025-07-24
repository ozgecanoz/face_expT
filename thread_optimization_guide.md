# Thread Optimization Guide for Batch Downloads

## 🎯 Quick Recommendations

### **For Google Cloud VMs:**
- **e2-standard-4** (4 vCPUs): Use **4-6 threads**
- **e2-standard-8** (8 vCPUs): Use **6-8 threads**
- **e2-standard-16** (16 vCPUs): Use **8-12 threads**
- **n2-standard-4** (4 vCPUs): Use **4-6 threads**
- **n2-standard-8** (8 vCPUs): Use **6-10 threads**

### **For Local Machines:**
- **4-core CPU**: Use **3-4 threads**
- **8-core CPU**: Use **4-6 threads**
- **16-core CPU**: Use **6-8 threads**

## 🔍 How to Determine Optimal Thread Count

### **1. Check Your System Resources**

```bash
# Check CPU cores
nproc
# or
lscpu | grep "CPU(s):"

# Check available memory
free -h

# Check current network speed (if available)
speedtest-cli
```

### **2. Test Different Thread Counts**

Create a test script to benchmark different thread counts:

```bash
#!/bin/bash
# test_thread_performance.sh

echo "Testing different thread counts..."
echo "=================================="

for threads in 2 4 6 8 10 12; do
    echo "Testing with $threads threads..."
    
    # Time the download of a small subset
    start_time=$(date +%s)
    
    python3 batch_download_to_gcs_multithreaded.py \
        --dataset-config CCV2_dataset_urls.json \
        --max-workers $threads
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "✅ $threads threads completed in $duration seconds"
    echo "---"
done
```

### **3. Monitor System Performance**

While running downloads, monitor these metrics:

```bash
# Monitor CPU usage
htop

# Monitor network usage
iftop

# Monitor disk I/O
iotop

# Monitor memory usage
free -h
```

## 📊 Thread Count Decision Matrix

| Factor | Low Threads (2-4) | Medium Threads (4-8) | High Threads (8-12) |
|--------|-------------------|----------------------|---------------------|
| **CPU Cores** | 2-4 cores | 4-8 cores | 8+ cores |
| **Network Speed** | < 100 Mbps | 100-500 Mbps | > 500 Mbps |
| **Memory** | < 8 GB | 8-16 GB | > 16 GB |
| **File Sizes** | Large files (>1GB) | Medium files (100MB-1GB) | Small files (<100MB) |
| **Dataset Count** | < 20 files | 20-50 files | > 50 files |

## 🚀 Performance Optimization Tips

### **For Large Files (CCv2 datasets):**
- **Start with 4 threads** (most CCv2 files are 1-5GB)
- **Monitor network utilization**
- **Increase if network isn't saturated**

### **For Many Small Files:**
- **Use more threads** (6-8) since files download quickly
- **Focus on CPU utilization**

### **For Limited Bandwidth:**
- **Use fewer threads** (2-3) to avoid overwhelming connection
- **Monitor network errors**

## 🔧 Practical Testing Approach

### **Step 1: Start Conservative**
```bash
python3 batch_download_to_gcs_multithreaded.py \
    --dataset-config CCV2_dataset_urls.json \
    --max-workers 4
```

### **Step 2: Monitor Performance**
- Watch CPU usage (should be 60-80%)
- Watch network usage (should be 70-90%)
- Check for errors or timeouts

### **Step 3: Adjust Based on Results**

**If CPU usage < 50% and no errors:**
```bash
# Increase threads
--max-workers 6
```

**If network errors or timeouts:**
```bash
# Decrease threads
--max-workers 2
```

**If memory usage > 80%:**
```bash
# Decrease threads
--max-workers 3
```

## 📈 Expected Performance Gains

| Threads | Expected Speedup | Best For |
|---------|------------------|----------|
| 1 | 1x (baseline) | Testing only |
| 2 | 1.5-2x | Limited bandwidth |
| 4 | 2.5-3.5x | **Recommended starting point** |
| 6 | 3.5-4.5x | Good balance |
| 8 | 4-5x | High-end systems |
| 12+ | 4-6x | Diminishing returns |

## ⚠️ Common Pitfalls

### **Too Many Threads:**
- Network timeouts
- Memory exhaustion
- CPU thrashing
- GCS rate limiting

### **Too Few Threads:**
- Underutilized resources
- Slower overall completion
- Wasted VM time

## 🎯 Recommended Starting Points

### **For CCv2 Dataset (87 files, 1-5GB each):**

**Google Cloud e2-standard-4 VM:**
```bash
--max-workers 4
```

**Google Cloud e2-standard-8 VM:**
```bash
--max-workers 6
```

**Local 8-core machine:**
```bash
--max-workers 4
```

## 🔍 Monitoring Commands

```bash
# Real-time monitoring
watch -n 1 'echo "CPU:" && top -bn1 | grep "Cpu(s)" && echo "Memory:" && free -h && echo "Network:" && ifconfig | grep "bytes"'

# Check for errors
tail -f /var/log/syslog | grep -i "network\|timeout\|error"

# Monitor GCS uploads
gsutil ls -l gs://your-bucket/datasets/ | wc -l
```

## 📊 Performance Tracking

Keep a log of your tests:

```bash
# performance_log.txt
Date: 2024-01-15
VM Type: e2-standard-4
Threads: 4
Files: 87
Total Time: 2h 15m
Success Rate: 87/87
Network Errors: 0
CPU Peak: 75%
Memory Peak: 6.2GB
```

This will help you optimize for future downloads! 