#!/bin/bash

# Download and organize clips from GCS bucket with random 70/30 split
# 70% to training, 30% to validation
# chmod +x download_and_organize_clips_simple.sh

set -e  # Exit on any error

# Configuration
BUCKET_PATH="gs://face-training-datasets/CCA_train_db4_no_padding_keywords_offset_1.0"
TRAIN_DIR="/mnt/dataset-storage/dbs/CCA_train_db4_no_padding"
VAL_DIR="/mnt/dataset-storage/dbs/CCA_val_db4_no_padding"
TRAIN_RATIO=0.7  # 70% for training

# Function to extract subject ID from filename
extract_subject_id() {
    local filename="$1"
    # Extract subject ID from filename like "subject_45_1184_09_faces_16_24.mp4"
    # Pattern: subject_<id>_<rest>
    local subject_id=$(echo "$filename" | sed -n 's/^subject_\([0-9]*\)_.*/\1/p')
    echo "$subject_id"
}

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
train_count=$(echo "$total_files * $TRAIN_RATIO" | bc | cut -d. -f1)
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

# Step 4: Download training files
echo ""
echo "Step 4: Downloading training files..."
train_downloaded=0
for idx in "${train_indices[@]}"; do
    gcs_file="${all_files[$idx]}"
    filename=$(basename "$gcs_file")
    
    echo "Downloading training file $((train_downloaded + 1))/${#train_indices[@]}: $filename"
    gsutil cp "$gcs_file" "$TRAIN_DIR/"
    train_downloaded=$((train_downloaded + 1))
    
    # Progress update every 10 files
    if [[ $((train_downloaded % 10)) -eq 0 ]]; then
        echo "Training progress: $train_downloaded/${#train_indices[@]} files downloaded"
    fi
done

# Step 5: Download validation files
echo ""
echo "Step 5: Downloading validation files..."
val_downloaded=0
for idx in "${val_indices[@]}"; do
    gcs_file="${all_files[$idx]}"
    filename=$(basename "$gcs_file")
    
    echo "Downloading validation file $((val_downloaded + 1))/${#val_indices[@]}: $filename"
    gsutil cp "$gcs_file" "$VAL_DIR/"
    val_downloaded=$((val_downloaded + 1))
    
    # Progress update every 10 files
    if [[ $((val_downloaded % 10)) -eq 0 ]]; then
        echo "Validation progress: $val_downloaded/${#val_indices[@]} files downloaded"
    fi
done

# Step 6: Final summary and statistics
echo ""
echo "=== Download Complete ==="
echo "Training files downloaded: $train_downloaded"
echo "Validation files downloaded: $val_downloaded"
echo "Total files downloaded: $((train_downloaded + val_downloaded))"
echo ""
echo "Directories:"
echo "  Training: $TRAIN_DIR"
echo "  Validation: $VAL_DIR"
echo ""

# Show some statistics about subject distribution
echo "=== Subject Distribution Analysis ==="
echo "Training directory subjects:"
ls "$TRAIN_DIR"/*.mp4 2>/dev/null | while read -r file; do
    filename=$(basename "$file")
    subject_id=$(extract_subject_id "$filename")
    echo "  $subject_id"
done | sort -n | uniq -c | head -10

echo ""
echo "Validation directory subjects:"
ls "$VAL_DIR"/*.mp4 2>/dev/null | while read -r file; do
    filename=$(basename "$file")
    subject_id=$(extract_subject_id "$filename")
    echo "  $subject_id"
done | sort -n | uniq -c | head -10

echo ""
echo "You can now use these directories for training:"
echo "  --train-data-dir $TRAIN_DIR"
echo "  --val-data-dir $VAL_DIR" 