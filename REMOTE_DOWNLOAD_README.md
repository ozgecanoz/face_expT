# Remote Dataset Download to Google Cloud Storage

Scripts to download datasets from remote URLs and upload them directly to Google Cloud Storage. Perfect for running on CPU VMs to avoid downloading large datasets to your local machine.

## Scripts Overview

### 1. `download_to_gcs.py` - Single Dataset Download
Downloads a single dataset from a URL and uploads it to GCS.

### 2. `batch_download_to_gcs.py` - Batch Download
Downloads multiple datasets from a JSON configuration file.

## Quick Start

### Single Dataset Download

```bash
python download_to_gcs.py \
  --url "https://example.com/dataset.zip" \
  --remote-path "datasets/my_dataset"
```

### Batch Download

1. **Create dataset configuration:**
```bash
python batch_download_to_gcs.py --create-sample
```

2. **Edit the configuration file** (`dataset_urls.json`):
```json
{
  "datasets": [
    {
      "name": "face_dataset_1",
      "url": "https://example.com/face_dataset_1.zip",
      "remote_path": "datasets/face_dataset_1"
    },
    {
      "name": "face_dataset_2",
      "url": "https://example.com/face_dataset_2.zip", 
      "remote_path": "datasets/face_dataset_2"
    }
  ]
}
```

3. **Run batch download:**
```bash
python batch_download_to_gcs.py
```

## Usage Examples

### Single Dataset
```bash
# Download a zip file
python download_to_gcs.py \
  --url "https://storage.googleapis.com/public-datasets/face_dataset.zip" \
  --remote-path "datasets/face_dataset"

# Download a tar.gz file
python download_to_gcs.py \
  --url "https://example.com/dataset.tar.gz" \
  --remote-path "datasets/compressed_dataset"
```

### Batch Processing
```bash
# Process all datasets in config
python batch_download_to_gcs.py

# Use custom config file
python batch_download_to_gcs.py --dataset-config my_datasets.json

# Use custom bucket
python batch_download_to_gcs.py --bucket-name my-custom-bucket
```

## What the Scripts Do

1. **Download** - Downloads file from URL with progress tracking
2. **Extract** - Automatically extracts zip files
3. **Upload** - Uploads to GCS using efficient `gsutil rsync`
4. **Verify** - Lists files to verify successful upload
5. **Cleanup** - Removes temporary files automatically

## Cloud VM Setup

### 1. Create a CPU VM
```bash
gcloud compute instances create dataset-downloader \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=debian-11 \
  --image-project=debian-cloud
```

### 2. SSH to VM
```bash
gcloud compute ssh dataset-downloader --zone=us-central1-a
```

### 3. Install Dependencies
```bash
# Update system
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip

# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Install Python dependencies
pip3 install requests
```

### 4. Copy Scripts and Config
```bash
# Copy your scripts and config files to the VM
gcloud compute scp download_to_gcs.py dataset-downloader:~/ --zone=us-central1-a
gcloud compute scp batch_download_to_gcs.py dataset-downloader:~/ --zone=us-central1-a
gcloud compute scp gcp_config.json dataset-downloader:~/ --zone=us-central1-a
gcloud compute scp dataset-uploader-key.json dataset-downloader:~/ --zone=us-central1-a
```

### 5. Run Download Scripts
```bash
# Single dataset
python3 download_to_gcs.py --url "YOUR_URL" --remote-path "datasets/name"

# Batch download
python3 batch_download_to_gcs.py
```

## Configuration Files

### GCP Config (`gcp_config.json`)
```json
{
  "project_id": "your-project-id",
  "key_file": "dataset-uploader-key.json",
  "bucket_name": "face-training-datasets",
  "region": "us-central1"
}
```

### Dataset URLs Config (`dataset_urls.json`)
```json
{
  "datasets": [
    {
      "name": "dataset_name",
      "url": "https://example.com/dataset.zip",
      "remote_path": "datasets/dataset_name"
    }
  ]
}
```

## Features

- ✅ **Progress tracking** - Shows download progress
- ✅ **Automatic extraction** - Handles zip files automatically
- ✅ **Efficient upload** - Uses `gsutil rsync` for speed
- ✅ **Error handling** - Continues processing on errors
- ✅ **Temporary cleanup** - Automatically removes temp files
- ✅ **Verification** - Lists files to confirm upload
- ✅ **Batch processing** - Process multiple datasets

## Supported File Types

- **ZIP files** - Automatically extracted
- **TAR.GZ files** - Automatically extracted  
- **Individual files** - Uploaded directly
- **Directories** - Uploaded as-is

## Cost Optimization

- **Use CPU VMs** - Much cheaper than GPU VMs for downloading
- **Spot instances** - Use spot instances for even lower costs
- **Regional storage** - Keep datasets in same region as your training VMs
- **Cleanup** - Scripts automatically clean up temporary files

## Troubleshooting

### Common Issues

1. **"gcloud not found"**
   ```bash
   # Install Google Cloud SDK
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   ```

2. **"Permission denied"**
   ```bash
   # Authenticate
   gcloud auth activate-service-account --key-file=dataset-uploader-key.json
   ```

3. **"Download failed"**
   - Check URL is accessible
   - Verify network connectivity
   - Check disk space on VM

4. **"Upload failed"**
   - Verify bucket exists
   - Check service account permissions
   - Ensure sufficient storage quota

### Getting Help

- **Google Cloud docs**: https://cloud.google.com/storage/docs
- **gsutil reference**: https://cloud.google.com/storage/docs/gsutil
- **VM documentation**: https://cloud.google.com/compute/docs

## Security Notes

⚠️ **Important:**
- Keep your service account key secure
- Use minimal permissions for service account
- Consider rotating keys periodically
- Monitor costs in Google Cloud Console 