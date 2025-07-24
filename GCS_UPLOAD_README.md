# Google Cloud Storage Upload Script

Quick guide for uploading datasets to Google Cloud Storage using the `upload_dataset_to_gcs.py` script.

## Quick Start

### 1. Set up Authentication (First time only)

```bash
python setup_gcp_auth.py
```

This creates:
- Service account and key file
- Configuration file
- Updated `.gitignore`

### 2. Upload Your Dataset

```bash
python upload_dataset_to_gcs.py --dataset-path /path/to/your/dataset
```

## Usage Examples

### Basic Upload
```bash
python upload_dataset_to_gcs.py --dataset-path /Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1
```

### Custom Remote Path
```bash
python upload_dataset_to_gcs.py \
  --dataset-path /path/to/dataset \
  --remote-path my_custom_dataset_name
```

### Custom Bucket
```bash
python upload_dataset_to_gcs.py \
  --dataset-path /path/to/dataset \
  --bucket-name my-custom-bucket
```

### Custom Key File
```bash
python upload_dataset_to_gcs.py \
  --dataset-path /path/to/dataset \
  --key-file /path/to/custom-key.json
```

## What the Script Does

1. **Authenticates** with Google Cloud using service account
2. **Creates bucket** if it doesn't exist
3. **Uploads dataset** using efficient `gsutil rsync`
4. **Verifies upload** by listing files
5. **Generates download script** for cloud VM use

## Output Files

- **Uploaded dataset**: `gs://face-training-datasets/YOUR_DATASET_NAME`
- **Download script**: `download_dataset_from_gcs.py` (for cloud VM use)

## Cloud VM Usage

After uploading, copy the generated `download_dataset_from_gcs.py` script to your cloud VM and run:

```bash
python3 download_dataset_from_gcs.py
```

## Configuration

The script uses `gcp_config.json` for default settings:
- Project ID
- Bucket name: `face-training-datasets`
- Key file: `dataset-uploader-key.json`
- Region: `us-central1`

## Troubleshooting

### Common Issues

1. **"gcloud not found"**
   ```bash
   # Install Google Cloud SDK
   brew install google-cloud-sdk  # macOS
   ```

2. **"Permission denied"**
   ```bash
   # Authenticate first
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **"Key file not found"**
   - Run `setup_gcp_auth.py` first to create the key file

4. **"Bucket already exists"**
   - This is normal, the script will use the existing bucket

### Getting Help

- **Full setup guide**: See `GCP_SETUP_GUIDE.md`
- **Google Cloud docs**: https://cloud.google.com/storage/docs
- **gcloud CLI**: https://cloud.google.com/sdk/gcloud

## Security Notes

⚠️ **Important:**
- The key file (`dataset-uploader-key.json`) is automatically added to `.gitignore`
- Never commit the key file to version control
- Keep the key file secure on your machine

## Cost Information

- **Storage**: ~$0.02/GB/month
- **Upload**: Free
- **Download to VM**: ~$0.12/GB (one-time)

For a 10GB dataset:
- Storage: ~$0.20/month
- Download: ~$1.20 (one-time) 