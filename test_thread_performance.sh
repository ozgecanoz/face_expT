#!/bin/bash
# Test different thread counts for batch downloads
# Usage: ./test_thread_performance.sh

echo "🧪 Thread Performance Testing for Batch Downloads"
echo "=================================================="

# Check if we're on a Google Cloud VM
if [[ -f /etc/google_cloud_config ]]; then
    echo "✅ Detected Google Cloud VM"
    VM_TYPE="Google Cloud"
else
    echo "🖥️  Detected local machine"
    VM_TYPE="Local"
fi

# Get system info
echo ""
echo "📊 System Information:"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "VM Type: $VM_TYPE"

# Check if dataset config exists
if [[ ! -f "CCV2_dataset_urls.json" ]]; then
    echo "❌ CCV2_dataset_urls.json not found!"
    echo "Please ensure your dataset config file exists."
    exit 1
fi

# Create a small test config with just 5 files for testing
echo ""
echo "🔧 Creating test configuration with 5 files..."
python3 -c "
import json
with open('CCV2_dataset_urls.json', 'r') as f:
    data = json.load(f)
test_data = {'datasets': data['datasets'][:5]}
with open('test_dataset_urls.json', 'w') as f:
    json.dump(test_data, f, indent=2)
print('✅ Created test_dataset_urls.json with 5 files')
"

# Test different thread counts
echo ""
echo "🚀 Testing different thread counts..."
echo "====================================="

# Define thread counts to test based on system
CPU_CORES=$(nproc)
if [[ $CPU_CORES -le 4 ]]; then
    THREAD_COUNTS=(2 3 4)
elif [[ $CPU_CORES -le 8 ]]; then
    THREAD_COUNTS=(2 4 6 8)
else
    THREAD_COUNTS=(2 4 6 8 10 12)
fi

# Results storage
declare -A RESULTS

for threads in "${THREAD_COUNTS[@]}"; do
    echo ""
    echo "🧵 Testing with $threads threads..."
    echo "-----------------------------------"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run the download with current thread count
    python3 batch_download_to_gcs_multithreaded.py \
        --dataset-config test_dataset_urls.json \
        --max-workers $threads \
        --bucket-name "test-bucket-$(date +%s)" 2>&1 | tee "test_output_${threads}threads.log"
    
    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Store result
    RESULTS[$threads]=$duration
    
    echo ""
    echo "✅ $threads threads completed in $duration seconds"
    
    # Wait a bit between tests
    sleep 5
done

# Display results
echo ""
echo "📊 Performance Results:"
echo "======================"
echo "Threads | Duration (s) | Speedup"
echo "--------|-------------|---------"

# Calculate baseline (slowest time)
baseline=0
for threads in "${THREAD_COUNTS[@]}"; do
    if [[ ${RESULTS[$threads]} -gt $baseline ]]; then
        baseline=${RESULTS[$threads]}
    fi
done

for threads in "${THREAD_COUNTS[@]}"; do
    duration=${RESULTS[$threads]}
    speedup=$(echo "scale=2; $baseline / $duration" | bc -l 2>/dev/null || echo "N/A")
    echo "$threads      | $duration        | ${speedup}x"
done

# Recommendations
echo ""
echo "🎯 Recommendations:"
echo "=================="

# Find best performing thread count
best_threads=2
best_time=${RESULTS[2]}
for threads in "${THREAD_COUNTS[@]}"; do
    if [[ ${RESULTS[$threads]} -lt $best_time ]]; then
        best_threads=$threads
        best_time=${RESULTS[$threads]}
    fi
done

echo "✅ Best performance: $best_threads threads ($best_time seconds)"

# Provide recommendations based on system
if [[ $CPU_CORES -le 4 ]]; then
    echo "💡 For your $CPU_CORES-core system, recommended range: 2-4 threads"
elif [[ $CPU_CORES -le 8 ]]; then
    echo "💡 For your $CPU_CORES-core system, recommended range: 4-6 threads"
else
    echo "💡 For your $CPU_CORES-core system, recommended range: 6-8 threads"
fi

echo ""
echo "🚀 For full CCv2 dataset (87 files), use:"
echo "python3 batch_download_to_gcs_multithreaded.py \\"
echo "    --dataset-config CCV2_dataset_urls.json \\"
echo "    --max-workers $best_threads"

# Cleanup
echo ""
echo "🧹 Cleaning up test files..."
rm -f test_dataset_urls.json
rm -f test_output_*threads.log

echo "✅ Testing complete!" 