#!/bin/bash
# Emergency cleanup script for disk space issues

echo "🚨 EMERGENCY DISK CLEANUP"
echo "========================="

# Check disk usage
echo "💾 Current disk usage:"
df -h

echo ""
echo "🧹 Cleaning up temporary files..."

# Clean up /tmp
echo "📁 Cleaning /tmp..."
sudo rm -rf /tmp/* 2>/dev/null

# Clean up /mnt/dataset-storage/tmp
echo "📁 Cleaning /mnt/dataset-storage/tmp..."
rm -rf /mnt/dataset-storage/tmp/* 2>/dev/null

# Clean up Google Cloud SDK cache
echo "☁️ Cleaning Google Cloud SDK cache..."
rm -rf ~/.config/gcloud/credentials.db 2>/dev/null
rm -rf ~/.config/gcloud/application_default_credentials.json 2>/dev/null

# Clean up pip cache
echo "🐍 Cleaning pip cache..."
pip3 cache purge 2>/dev/null

# Clean up apt cache
echo "📦 Cleaning apt cache..."
sudo apt-get clean 2>/dev/null

# Clean up logs
echo "📝 Cleaning old logs..."
sudo find /var/log -name "*.log" -mtime +7 -delete 2>/dev/null
sudo find /var/log -name "*.gz" -delete 2>/dev/null

# Clean up journal logs
echo "📋 Cleaning journal logs..."
sudo journalctl --vacuum-time=1d 2>/dev/null

# Show final disk usage
echo ""
echo "💾 Final disk usage:"
df -h

echo ""
echo "✅ Emergency cleanup complete!"
echo "🚀 You can now try running the download script again." 