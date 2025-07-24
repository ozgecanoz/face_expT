#!/bin/bash
# Check and clean up tmp directory

TMP_DIR="/mnt/dataset-storage/tmp"

echo "🧹 Checking and cleaning up tmp directory: $TMP_DIR"
echo "=================================================="

# Check if directory exists
if [[ ! -d "$TMP_DIR" ]]; then
    echo "📁 Creating tmp directory..."
    mkdir -p "$TMP_DIR"
    echo "✅ Created $TMP_DIR"
else
    echo "📁 Tmp directory exists"
fi

# Show disk usage
echo ""
echo "💾 Disk usage:"
df -h /mnt/dataset-storage

# Show tmp directory contents
echo ""
echo "📋 Contents of tmp directory:"
if [[ -z "$(ls -A $TMP_DIR 2>/dev/null)" ]]; then
    echo "   (empty)"
else
    ls -lah "$TMP_DIR"
fi

# Show file sizes
echo ""
echo "📊 File sizes in tmp directory:"
if [[ -n "$(ls -A $TMP_DIR 2>/dev/null)" ]]; then
    du -sh "$TMP_DIR"/*
    echo ""
    echo "Total size: $(du -sh $TMP_DIR | cut -f1)"
else
    echo "   No files found"
fi

# Clean up old files (older than 1 hour)
echo ""
echo "🧹 Cleaning up files older than 1 hour..."
find "$TMP_DIR" -type f -mmin +60 -exec rm -f {} \; 2>/dev/null
echo "✅ Cleanup complete"

# Show final status
echo ""
echo "📋 Final contents:"
if [[ -z "$(ls -A $TMP_DIR 2>/dev/null)" ]]; then
    echo "   (empty)"
else
    ls -lah "$TMP_DIR"
    echo ""
    echo "Total size: $(du -sh $TMP_DIR | cut -f1)"
fi

echo ""
echo "✅ Tmp directory check and cleanup complete!" 