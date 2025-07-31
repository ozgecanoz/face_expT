#!/bin/bash

# Test script to debug the split calculation
set -e

# Configuration
BUCKET_PATH="gs://face-training-datasets/CCA_train_db4_no_padding_keywords_offset_1.0"
TRAIN_RATIO=0.7  # 70% for training

# Step 1: List all files and filter MP4 files
echo "Step 1: Listing all files in bucket..."
all_files=()
while IFS= read -r gcs_file; do
    filename=$(basename "$gcs_file")
    
    # Only include MP4 files
    if [[ "$filename" =~ \.mp4$ ]]; then
        all_files+=("$gcs_file")
    fi
done < <(gsutil ls "$BUCKET_PATH/")

total_files=${#all_files[@]}
echo "Found $total_files MP4 files in bucket"

if [[ $total_files -eq 0 ]]; then
    echo "No MP4 files found in bucket. Exiting."
    exit 1
fi

# Step 2: Calculate split
# Calculate 70% of total files using integer arithmetic
train_count=$((total_files * 7 / 10))
val_count=$((total_files - train_count))

echo "Split calculation:"
echo "  Total files: $total_files"
echo "  Training files: $train_count (70%)"
echo "  Validation files: $val_count (30%)"
echo "  Debug: train_count + val_count = $((train_count + val_count))"

# Verify the split makes sense
if [[ $((train_count + val_count)) -ne $total_files ]]; then
    echo "ERROR: Split calculation doesn't add up!"
    echo "  train_count + val_count = $((train_count + val_count))"
    echo "  total_files = $total_files"
    exit 1
fi

# Step 3: Create random split
echo "Step 3: Creating random split..."

# Create array of indices and shuffle them (use sort -R for macOS compatibility)
indices=($(seq 0 $((total_files - 1))))
shuffled_indices=($(printf '%s\n' "${indices[@]}" | sort -R))

# Split into training and validation indices
train_indices=("${shuffled_indices[@]:0:train_count}")
val_indices=("${shuffled_indices[@]:train_count:val_count}")

echo "Random split created:"
echo "  Training indices: ${#train_indices[@]} files (${train_indices[*]:0:5}...)"
echo "  Validation indices: ${#val_indices[@]} files (${val_indices[*]:0:5}...)"
echo "  Debug: shuffled_indices length = ${#shuffled_indices[@]}"
echo "  Debug: train_count = $train_count, val_count = $val_count"
echo "  Debug: train_indices length = ${#train_indices[@]}"
echo "  Debug: val_indices length = ${#val_indices[@]}"

# Show first few files that would be downloaded
echo ""
echo "First 5 training files:"
for i in {0..4}; do
    if [[ $i -lt ${#train_indices[@]} ]]; then
        idx=${train_indices[$i]}
        gcs_file="${all_files[$idx]}"
        filename=$(basename "$gcs_file")
        echo "  $filename"
    fi
done

echo ""
echo "First 5 validation files:"
for i in {0..4}; do
    if [[ $i -lt ${#val_indices[@]} ]]; then
        idx=${val_indices[$i]}
        gcs_file="${all_files[$idx]}"
        filename=$(basename "$gcs_file")
        echo "  $filename"
    fi
done 