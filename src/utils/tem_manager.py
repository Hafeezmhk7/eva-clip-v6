"""
Snellius Temp Directory Manager for BLIP3-o Project
Handles structured temp directory layout for embeddings, checkpoints, and other data.
Place this file as: src/utils/temp_manager.py
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SnelliusTempManager:
    """
    Manages structured temp directories for BLIP3-o project on Snellius.
    
    Directory Structure:
    /scratch-shared/<user>/blip3o_workspace/     # Persistent shared storage (14 days)
    ├── datasets/                                # Downloaded TAR files
    ├── embeddings/                              # Extracted embeddings (persistent)
    ├── checkpoints/                             # Important model checkpoints
    ├── logs/                                    # Training logs
    └── metadata/                                # Metadata and manifests
    
    $TMPDIR/blip3o_job_<job_id>/                # Job-specific temp (deleted on job end)
    ├── cache/                                   # Model cache, temporary files
    ├── working/                                 # Current processing files
    └── temp_checkpoints/                        # Temporary checkpoints during training
    """
    
    def __init__(self, project_name: str = "blip3o_workspace"):
        self.project_name = project_name
        self.user = os.environ.get("USER", "user")
        self.job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
        
        # Setup base directories
        self._setup_base_directories()
        
        # Create directory structure
        self._create_directory_structure()
        
        # Save workspace info
        self._save_workspace_info()
        
        logger.info(f"Snellius temp manager initialized for user: {self.user}")
        logger.info(f"Workspace: {self.persistent_workspace}")
        logger.info(f"Job temp: {self.job_temp}")
    
    def _setup_base_directories(self):
        """Setup base directory paths based on Snellius environment."""
        
        # Persistent shared workspace (survives 14 days on scratch-shared)
        if "SCRATCH_SHARED" in os.environ:
            scratch_shared = Path(os.environ["SCRATCH_SHARED"])
            self.persistent_workspace = scratch_shared / self.user / self.project_name
        else:
            # Fallback for local development
            self.persistent_workspace = Path.home() / f".cache/{self.project_name}"
        
        # Job-specific temp directory (deleted when job ends)
        if "TMPDIR" in os.environ:
            tmpdir = Path(os.environ["TMPDIR"])
            self.job_temp = tmpdir / f"blip3o_job_{self.job_id}"
        elif "SCRATCH_LOCAL" in os.environ:
            scratch_local = Path(os.environ["SCRATCH_LOCAL"])
            self.job_temp = scratch_local / self.user / f"blip3o_job_{self.job_id}"
        else:
            # Fallback for local development
            self.job_temp = Path("/tmp") / f"blip3o_job_{self.job_id}"
    
    def _create_directory_structure(self):
        """Create the complete directory structure."""
        
        # Persistent directories (scratch-shared, 14 days retention)
        self.dirs = {
            # Persistent storage
            'workspace': self.persistent_workspace,
            'datasets': self.persistent_workspace / "datasets",
            'embeddings': self.persistent_workspace / "embeddings", 
            'checkpoints': self.persistent_workspace / "checkpoints",
            'logs': self.persistent_workspace / "logs",
            'metadata': self.persistent_workspace / "metadata",
            
            # Job-specific temp storage
            'job_temp': self.job_temp,
            'cache': self.job_temp / "cache",
            'working': self.job_temp / "working",
            'temp_checkpoints': self.job_temp / "temp_checkpoints",
        }
        
        # Create all directories
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {name} -> {path}")
        
        # Set environment variables for easy access
        os.environ["BLIP3O_WORKSPACE"] = str(self.persistent_workspace)
        os.environ["BLIP3O_JOB_TEMP"] = str(self.job_temp)
        os.environ["BLIP3O_EMBEDDINGS"] = str(self.dirs['embeddings'])
        os.environ["BLIP3O_CHECKPOINTS"] = str(self.dirs['checkpoints'])
        os.environ["BLIP3O_DATASETS"] = str(self.dirs['datasets'])
    
    def _save_workspace_info(self):
        """Save workspace information for job tracking."""
        workspace_info = {
            'created': datetime.now().isoformat(),
            'user': self.user,
            'job_id': self.job_id,
            'project_name': self.project_name,
            'persistent_workspace': str(self.persistent_workspace),
            'job_temp': str(self.job_temp),
            'environment': {
                'SCRATCH_SHARED': os.environ.get('SCRATCH_SHARED'),
                'TMPDIR': os.environ.get('TMPDIR'),
                'SCRATCH_LOCAL': os.environ.get('SCRATCH_LOCAL'),
            },
            'directories': {name: str(path) for name, path in self.dirs.items()},
            'cleanup_policy': {
                'persistent_storage': 'scratch-shared (14 days automatic cleanup)',
                'job_temp': 'deleted when job ends',
                'recommendations': [
                    'Move important checkpoints to persistent_workspace/checkpoints',
                    'Copy final models to home directory for long-term storage',
                    'Embeddings in persistent storage can be accessed across jobs'
                ]
            }
        }
        
        # Save to both persistent and temp locations
        for location in [self.dirs['metadata'], self.dirs['job_temp']]:
            info_file = location / f"workspace_info_{self.job_id}.json"
            with open(info_file, 'w') as f:
                json.dump(workspace_info, f, indent=2)
    
    def get_dir(self, name: str) -> Path:
        """Get directory path by name."""
        if name not in self.dirs:
            raise ValueError(f"Unknown directory: {name}. Available: {list(self.dirs.keys())}")
        return self.dirs[name]
    
    def get_datasets_dir(self) -> Path:
        """Get datasets directory for downloaded TAR files."""
        return self.dirs['datasets']
    
    def get_embeddings_dir(self) -> Path:
        """Get embeddings directory (persistent across jobs)."""
        return self.dirs['embeddings']
    
    def get_checkpoints_dir(self) -> Path:
        """Get persistent checkpoints directory."""
        return self.dirs['checkpoints']
    
    def get_working_dir(self) -> Path:
        """Get working directory for current job processing."""
        return self.dirs['working']
    
    def get_cache_dir(self) -> Path:
        """Get cache directory for model downloads, etc."""
        return self.dirs['cache']
    
    def get_temp_checkpoints_dir(self) -> Path:
        """Get temporary checkpoints directory (job-specific)."""
        return self.dirs['temp_checkpoints']
    
    def get_logs_dir(self) -> Path:
        """Get logs directory."""
        return self.dirs['logs']
    
    def setup_model_cache(self):
        """Setup model cache environment variables."""
        cache_dir = self.get_cache_dir()
        
        # Redirect all model caches to job temp
        os.environ["TORCH_HOME"] = str(cache_dir / "torch")
        os.environ["HF_HOME"] = str(cache_dir / "huggingface")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
        os.environ["WANDB_DIR"] = str(self.get_logs_dir() / "wandb")
        
        # Create cache subdirectories
        for subdir in ["torch", "huggingface", "transformers"]:
            (cache_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Model cache redirected to: {cache_dir}")
    
    def create_embeddings_subdirectory(self, name: str) -> Path:
        """Create a subdirectory in embeddings folder."""
        subdir = self.dirs['embeddings'] / name
        subdir.mkdir(exist_ok=True)
        return subdir
    
    def create_checkpoint_subdirectory(self, name: str) -> Path:
        """Create a subdirectory in checkpoints folder."""
        subdir = self.dirs['checkpoints'] / name
        subdir.mkdir(exist_ok=True)
        return subdir
    
    def save_checkpoint_to_persistent(self, 
                                    temp_checkpoint_path: Path, 
                                    checkpoint_name: str) -> Path:
        """Copy checkpoint from temp to persistent storage."""
        persistent_path = self.dirs['checkpoints'] / checkpoint_name
        
        if temp_checkpoint_path.is_dir():
            if persistent_path.exists():
                shutil.rmtree(persistent_path)
            shutil.copytree(temp_checkpoint_path, persistent_path)
        else:
            shutil.copy2(temp_checkpoint_path, persistent_path)
        
        logger.info(f"Checkpoint saved to persistent storage: {persistent_path}")
        return persistent_path
    
    def cleanup_temp_files(self, keep_patterns: Optional[list] = None):
        """Clean up temporary files, optionally keeping files matching patterns."""
        keep_patterns = keep_patterns or []
        
        working_dir = self.dirs['working']
        cleaned_size = 0
        
        for item in working_dir.iterdir():
            should_keep = any(pattern in item.name for pattern in keep_patterns)
            if not should_keep:
                if item.is_file():
                    cleaned_size += item.stat().st_size
                    item.unlink()
                elif item.is_dir():
                    cleaned_size += sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
        
        logger.info(f"Cleaned up {cleaned_size / 1024**2:.1f} MB from temp directory")
    
    def get_disk_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get disk usage information for all directories."""
        usage_info = {}
        
        for name, path in self.dirs.items():
            if path.exists():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                file_count = len(list(path.rglob('*')))
                
                usage_info[name] = {
                    'path': str(path),
                    'total_size_mb': total_size / 1024**2,
                    'total_size_gb': total_size / 1024**3,
                    'file_count': file_count,
                    'exists': True
                }
            else:
                usage_info[name] = {'exists': False}
        
        # Get system disk usage for main paths
        try:
            for key, path in [('persistent_workspace', self.persistent_workspace), 
                            ('job_temp', self.job_temp)]:
                if path.exists():
                    total, used, free = shutil.disk_usage(path)
                    usage_info[f'{key}_system'] = {
                        'total_gb': total / 1024**3,
                        'used_gb': used / 1024**3,
                        'free_gb': free / 1024**3,
                        'usage_percent': (used / total) * 100
                    }
        except Exception as e:
            logger.warning(f"Could not get system disk usage: {e}")
        
        return usage_info
    
    def print_status(self):
        """Print current status and usage information."""
        print(f"\n🗂️  BLIP3-o Workspace Status (Job {self.job_id})")
        print("=" * 70)
        
        # Directory structure
        print("📁 Directory Structure:")
        print(f"   Persistent Workspace: {self.persistent_workspace}")
        print(f"   Job Temp:            {self.job_temp}")
        
        # Directory details
        print("\n📊 Directory Usage:")
        usage = self.get_disk_usage()
        for name, info in usage.items():
            if info.get('exists', False):
                size_gb = info.get('total_size_gb', 0)
                file_count = info.get('file_count', 0)
                print(f"   {name:20s}: {size_gb:8.2f} GB ({file_count:,} files)")
        
        # Storage policies
        print("\n⏰ Storage Policies:")
        print("   Persistent (scratch-shared): 14 days automatic cleanup")
        print("   Job temp (TMPDIR):           Deleted when job ends")
        print("   Home directory:              200 GiB quota, backed up")
        
        # Recommendations
        print("\n💡 Recommendations:")
        print("   • Keep embeddings in persistent workspace")
        print("   • Save final models to home directory for long-term storage")
        print("   • Use temp directories for processing and cache")
        print("   • Monitor disk usage to avoid quotas")
        print("=" * 70)
    
    def create_job_script_snippet(self) -> str:
        """Generate bash snippet for job scripts."""
        return f'''
# BLIP3-o Workspace Setup
export BLIP3O_WORKSPACE="{self.persistent_workspace}"
export BLIP3O_JOB_TEMP="{self.job_temp}"
export BLIP3O_EMBEDDINGS="{self.dirs['embeddings']}"
export BLIP3O_CHECKPOINTS="{self.dirs['checkpoints']}"
export BLIP3O_DATASETS="{self.dirs['datasets']}"

# Model cache (redirected to job temp)
export TORCH_HOME="{self.dirs['cache']}/torch"
export HF_HOME="{self.dirs['cache']}/huggingface"
export TRANSFORMERS_CACHE="{self.dirs['cache']}/transformers"
export WANDB_DIR="{self.get_logs_dir()}/wandb"

echo "🗂️  BLIP3-o workspace ready:"
echo "   Persistent: $BLIP3O_WORKSPACE"
echo "   Job temp:   $BLIP3O_JOB_TEMP"
'''


def get_temp_manager(project_name: str = "blip3o_workspace") -> SnelliusTempManager:
    """Get or create temp manager instance."""
    return SnelliusTempManager(project_name)


def setup_snellius_environment(project_name: str = "blip3o_workspace") -> SnelliusTempManager:
    """Setup complete Snellius environment for BLIP3-o project."""
    manager = get_temp_manager(project_name)
    manager.setup_model_cache()
    manager.print_status()
    return manager


if __name__ == "__main__":
    # Test the temp manager
    manager = setup_snellius_environment()
    
    # Print job script snippet
    print("\n🔧 Add this to your job scripts:")
    print(manager.create_job_script_snippet())