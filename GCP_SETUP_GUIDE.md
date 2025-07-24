# Google Cloud Platform Setup Guide

This guide will help you set up Google Cloud Platform authentication and upload your dataset to Google Cloud Storage for training on cloud VMs.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **gcloud CLI**: Install the Google Cloud SDK
   ```bash
   # macOS (using Homebrew)
   brew install google-cloud-sdk
   
   # Or download from Google Cloud website
   # https://cloud.google.com/sdk/docs/install
   ```
3. **Python 3.7+**: Make sure you have Python 3.7 or higher installed

## Step 1: Set up Google Cloud Project

1. **Create a new project** (or use existing):
   ```bash
   gcloud projects create YOUR_PROJECT_ID --name="Face Training Project"
   ```

2. **Set the project as default**:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable required APIs**:
   ```bash
   gcloud services enable storage.googleapis.com
   gcloud services enable iam.googleapis.com
   ```

## Step 2: Run Authentication Setup

The `setup_gcp_auth.py` script will:
- Create a service account for secure authentication
- Generate a key file for the service account
- Grant necessary permissions
- Add the key file to `.gitignore` for security
- Create a configuration file

```bash
python setup_gcp_auth.py
```

**What this creates:**
- `dataset-uploader-key.json` - Service account key file (⚠️ **KEEP SECURE**)
- `gcp_config.json` - Configuration file for upload script
- Updated `.gitignore` - Prevents key file from being committed

## Step 3: Upload Your Dataset

Use the `upload_dataset_to_gcs.py` script to upload your dataset:

```bash
python upload_dataset_to_gcs.py --dataset-path /path/to/your/dataset
```

**Options:**
- `--dataset-path`: Path to your local dataset directory (required)
- `--key-file`: Service account key file (default: `dataset-uploader-key.json`)
- `--config`: Configuration file (default: `gcp_config.json`)
- `--remote-path`: Remote path in bucket (default: dataset name)
- `--bucket-name`: GCS bucket name (overrides config)

**Example:**
```bash
python upload_dataset_to_gcs.py --dataset-path /Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1
```

## Step 4: Cloud VM Setup

Once your dataset is uploaded, you can set up a cloud VM:

1. **Create a VM instance** with GPU support
2. **Install required software**:
   ```bash
   # Install Python and pip
   sudo apt update
   sudo apt install python3 python3-pip
   
   # Install Google Cloud SDK
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   
   # Install PyTorch and other dependencies
   pip3 install torch torchvision torchaudio
   pip3 install tensorboard tqdm
   ```

3. **Download the dataset** using the generated script:
   ```bash
   # Copy the download script to your VM
   # Then run:
   python3 download_dataset_from_gcs.py
   ```

## Security Notes

⚠️ **IMPORTANT SECURITY CONSIDERATIONS:**

1. **Never commit the key file**: The `dataset-uploader-key.json` file contains sensitive credentials and is automatically added to `.gitignore`

2. **Rotate keys regularly**: Consider rotating your service account keys periodically

3. **Use minimal permissions**: The service account is granted Storage Admin permissions. For production, consider using more restrictive permissions

4. **Secure key storage**: Keep your key file secure and don't share it

## Troubleshooting

### Common Issues

1. **"gcloud command not found"**
   - Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install

2. **"Permission denied"**
   - Make sure you're authenticated: `gcloud auth login`
   - Check your project: `gcloud config get-value project`

3. **"Bucket already exists"**
   - This is normal if the bucket was created previously

4. **"Service account already exists"**
   - This is normal if you've run the setup before

### Getting Help

- **Google Cloud Documentation**: https://cloud.google.com/docs
- **gcloud CLI Reference**: https://cloud.google.com/sdk/gcloud/reference
- **GCS Documentation**: https://cloud.google.com/storage/docs

## Cost Considerations

- **Storage costs**: ~$0.02 per GB per month for Standard storage
- **Network egress**: ~$0.12 per GB (when downloading to VM)
- **Compute costs**: Varies by VM type and region

**Example costs for 10GB dataset:**
- Storage: ~$0.20/month
- Download to VM: ~$1.20 (one-time)

## Next Steps

After uploading your dataset:

1. **Set up your cloud VM** with GPU support
2. **Download the dataset** using the generated script
3. **Run your training scripts** on the cloud VM
4. **Monitor costs** in the Google Cloud Console

The uploaded dataset will be available at: `gs://face-training-datasets/YOUR_DATASET_NAME` 