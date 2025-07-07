"""
Download BLIP3o-Pretrain-Short-Caption dataset
Place this file in: src/data/download_data.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import argparse

def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__).resolve()
    # Go up from src/data/download_data.py to project root
    return current_file.parent.parent.parent

def download_blip3o_shard(shard_idx=0, data_dir=None, force_download=False):
    """
    Download a single shard of BLIP3o-Pretrain-Short-Caption dataset
    
    Args:
        shard_idx (int): Which shard to download (0-11)
        data_dir (str): Directory to save data. If None, uses project_root/data
        force_download (bool): Force re-download even if file exists
    
    Returns:
        str: Path to downloaded file, or None if failed
    """
    
    # Set up paths
    if data_dir is None:
        project_root = get_project_root()
        data_dir = project_root / "data"
    else:
        data_dir = Path(data_dir)
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset info
    repo_id = "BLIP3o/BLIP3o-Pretrain-Short-Caption"
    shard_filename = f"{shard_idx:05d}.tar"
    
    print(f"Downloading BLIP3o Short Caption Dataset")
    print(f"Repository: {repo_id}")
    print(f"Shard: {shard_filename} (shard {shard_idx})")
    print(f"Destination: {data_dir}")
    print("=" * 60)
    
    # Check if file already exists
    local_file_path = data_dir / shard_filename
    if local_file_path.exists() and not force_download:
        file_size_gb = local_file_path.stat().st_size / (1024**3)
        print(f"File already exists: {local_file_path}")
        print(f"Size: {file_size_gb:.2f} GB")
        print("Use --force to re-download")
        return str(local_file_path)
    
    try:
        print(f"Starting download...")
        
        # Download using HuggingFace Hub
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_filename,
            repo_type="dataset",
            local_dir=str(data_dir),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
        )
        
        # Verify download
        if os.path.exists(downloaded_path):
            file_size_gb = os.path.getsize(downloaded_path) / (1024**3)
            print(f" Download successful!")
            print(f" File path: {downloaded_path}")
            print(f" File size: {file_size_gb:.2f} GB")
            
            # Estimate number of samples
            estimated_samples = int(file_size_gb * 400000 / 1.0)  # Rough estimate
            print(f" Estimated samples: ~{estimated_samples:,}")
            
            return downloaded_path
        else:
            print(f" Download failed: File not found at {downloaded_path}")
            return None
            
    except Exception as e:
        print(f" Download error: {e}")
        print(f" Make sure you have internet connection and sufficient disk space")
        return None

def list_available_files(repo_id="BLIP3o/BLIP3o-Pretrain-Short-Caption"):
    """List all available files in the repository"""
    try:
        print(f" Available files in {repo_id}:")
        files = list_repo_files(repo_id, repo_type="dataset")
        
        tar_files = [f for f in files if f.endswith('.tar')]
        tar_files.sort()
        
        print(f" Found {len(tar_files)} tar files:")
        for i, filename in enumerate(tar_files):
            shard_num = filename.replace('.tar', '')
            print(f"  {i:2d}. {filename} (shard {int(shard_num)})")
        
        return tar_files
        
    except Exception as e:
        print(f" Error listing files: {e}")
        return []

def verify_download(file_path):
    """Verify the downloaded file"""
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
        print(f" File seems too small: {file_size} bytes")
        return False
    
    print(f" File verification passed")
    print(f" Size: {file_size / (1024**3):.2f} GB")
    return True

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Download BLIP3o-Pretrain-Short-Caption dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data/download_data.py                    # Download shard 0 to ./data
  python src/data/download_data.py --shard 1          # Download shard 1
  python src/data/download_data.py --data_dir /tmp    # Download to /tmp
  python src/data/download_data.py --list             # List available files
  python src/data/download_data.py --force            # Force re-download
        """
    )
    
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard index to download (default: 0)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory to save data (default: project_root/data)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files in the repository"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify a downloaded file"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_files()
        return
    
    if args.verify:
        verify_download(args.verify)
        return
    
    # Download the shard
    downloaded_path = download_blip3o_shard(
        shard_idx=args.shard,
        data_dir=args.data_dir,
        force_download=args.force
    )
    
    if downloaded_path:
        print(f"\n Success! Ready for next steps:")
        print(f"  1. Cache features: python scripts/cache_features.py --shards {args.shard}")
        print(f"  2. Test dataset: python src/data/dataset.py")
        print(f"  3. Start training pipeline development")
    else:
        print(f"\n Download failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
