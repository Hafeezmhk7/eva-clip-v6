"""
FIXED Snellius Temp Directory Manager for BLIP3-o Project
Handles structured temp directory layout for embeddings, checkpoints, and other data.
Place this file as: src/modules/utils/temp_manager.py

FIXES:
- Proper Snellius environment detection
- Uses correct scratch directories
- Avoids home directory quota issues
- Better fallback mechanisms
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
    FIXED: Manages structured temp directories for BLIP3-o project on Snellius.
    
    Directory Structure:
    /scratch-shared/<user>/blip3o_workspace/     # Persistent shared storage (14 days)
    ├── datasets/                                # Downloaded TAR files
    ├── embeddings/                              # Extracted embeddings (persistent)
    ├── checkpoints/                             # Important model checkpoints
    ├── logs/                                    # Training logs
    └── metadata/                                # Metadata and manifests
    
    /scratch-local/<user>.<job_id>/blip3o_job_<job_id>/    # Job-specific temp
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
        """FIXED: Setup base directory paths based on Snellius environment."""
        
        # FIXED: First check for explicit BLIP3O environment variables
        if "BLIP3O_WORKSPACE" in os.environ:
            self.persistent_workspace = Path(os.environ["BLIP3O_WORKSPACE"])
            logger.info(f"Using explicit BLIP3O_WORKSPACE: {self.persistent_workspace}")
        
        # FIXED: Check for SCRATCH_SHARED environment variable
        elif "SCRATCH_SHARED" in os.environ:
            scratch_shared = Path(os.environ["SCRATCH_SHARED"])
            self.persistent_workspace = scratch_shared / self.user / self.project_name
            logger.info(f"Using SCRATCH_SHARED: {self.persistent_workspace}")
        
        # FIXED: Try standard Snellius scratch-shared path
        elif Path("/scratch-shared").exists():
            self.persistent_workspace = Path("/scratch-shared") / self.user / self.project_name
            logger.info(f"Using standard scratch-shared: {self.persistent_workspace}")
        
        # FIXED: Check for any scratch directory
        elif Path("/scratch").exists():
            self.persistent_workspace = Path("/scratch") / self.user / self.project_name
            logger.warning(f"Using /scratch (not ideal): {self.persistent_workspace}")
        
        # FIXED: Only use home directory as last resort with warning
        else:
            self.persistent_workspace = Path.home() / f".cache/{self.project_name}"
            logger.warning(f"FALLBACK: Using home directory (quota risk): {self.persistent_workspace}")
            logger.warning("Consider setting BLIP3O_WORKSPACE or ensure scratch-shared is available")
        
        # FIXED: Job temp directory setup
        if "BLIP3O_JOB_TEMP" in os.environ:
            self.job_temp = Path(os.environ["BLIP3O_JOB_TEMP"])
            logger.info(f"Using explicit BLIP3O_JOB_TEMP: {self.job_temp}")
        
        elif "TMPDIR" in os.environ:
            tmpdir = Path(os.environ["TMPDIR"])
            self.job_temp = tmpdir / f"blip3o_job_{self.job_id}"
            logger.info(f"Using TMPDIR: {self.job_temp}")
        
        elif "SCRATCH_LOCAL" in os.environ:
            scratch_local = Path(os.environ["SCRATCH_LOCAL"])
            self.job_temp = scratch_local / f"{self.user}.{self.job_id}" / f"blip3o_job_{self.job_id}"
            logger.info(f"Using SCRATCH_LOCAL: {self.job_temp}")
        
        # FIXED: Try standard Snellius scratch-local path
        elif Path("/scratch-local").exists():
            self.job_temp = Path("/scratch-local") / f"{self.user}.{self.job_id}" / f"blip3o_job_{self.job_id}"
            logger.info(f"Using standard scratch-local: {self.job_temp}")
        
        # FIXED: Fallback to /tmp with job isolation
        else:
            self.job_temp = Path("/tmp") / f"blip3o_job_{self.user}_{self.job_id}"
            logger.warning(f"FALLBACK: Using /tmp: {self.job_temp}")
    
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
        
        # Create all directories with error handling
        for name, path in self.dirs.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {name} -> {path}")
            except PermissionError as e:
                logger.error(f"Permission denied creating {name}: {path}")
                logger.error(f"Error: {e}")
                raise
            except OSError as e:
                if "Disk quota exceeded" in str(e):
                    logger.error(f"DISK QUOTA EXCEEDED when creating {name}: {path}")
                    logger.error("Consider using scratch directories instead of home")
                    raise
                else:
                    logger.error(f"OS Error creating {name}: {path} - {e}")
                    raise
        
        # FIXED: Set environment variables for easy access
        os.environ["BLIP3O_WORKSPACE"] = str(self.persistent_workspace)
        os.environ["BLIP3O_JOB_TEMP"] = str(self.job_temp)
        os.environ["BLIP3O_EMBEDDINGS"] = str(self.dirs['embeddings'])
        os.environ["BLIP3O_CHECKPOINTS"] = str(self.dirs['checkpoints'])
        os.environ["BLIP3O_DATASETS"] = str(self.dirs['datasets'])
        os.environ["BLIP3O_CACHE"] = str(self.dirs['cache'])
        os.environ["BLIP3O_LOGS"] = str(self.dirs['logs'])
    
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
                'BLIP3O_WORKSPACE': os.environ.get('BLIP3O_WORKSPACE'),
                'BLIP3O_JOB_TEMP': os.environ.get('BLIP3O_JOB_TEMP'),
            },
            'directories': {name: str(path) for name, path in self.dirs.items()},
            'disk_usage_at_creation': self._get_safe_disk_usage(),
            'cleanup_policy': {
                'persistent_storage': 'scratch-shared (14 days automatic cleanup)',
                'job_temp': 'deleted when job ends',
                'recommendations': [
                    'Move important checkpoints to persistent_workspace/checkpoints',
                    'Copy final models to home directory for long-term storage',
                    'Embeddings in persistent storage can be accessed across jobs',
                    'Monitor disk quotas to avoid failures'
                ]
            }
        }
        
        # Save to both persistent and temp locations
        for location in [self.dirs['metadata'], self.dirs['job_temp']]:
            try:
                info_file = location / f"workspace_info_{self.job_id}.json"
                with open(info_file, 'w') as f:
                    json.dump(workspace_info, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save workspace info to {location}: {e}")
    
    def _get_safe_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information safely (handles errors)."""
        usage_info = {}
        
        # Check key directories
        paths_to_check = [
            ('persistent_workspace', self.persistent_workspace),
            ('job_temp', self.job_temp),
            ('home', Path.home()),
        ]
        
        for name, path in paths_to_check:
            try:
                if path.exists():
                    total, used, free = shutil.disk_usage(path)
                    usage_info[name] = {
                        'total_gb': total / 1024**3,
                        'used_gb': used / 1024**3,
                        'free_gb': free / 1024**3,
                        'usage_percent': (used / total) * 100,
                        'path': str(path)
                    }
                else:
                    usage_info[name] = {'exists': False, 'path': str(path)}
            except Exception as e:
                usage_info[name] = {'error': str(e), 'path': str(path)}
        
        return usage_info
    
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
        cache_env_vars = {
            "TORCH_HOME": str(cache_dir / "torch"),
            "HF_HOME": str(cache_dir / "huggingface"),
            "TRANSFORMERS_CACHE": str(cache_dir / "transformers"),
            "WANDB_DIR": str(self.get_logs_dir() / "wandb"),
            "HUGGINGFACE_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
            "HF_DATASETS_CACHE": str(cache_dir / "datasets"),
        }
        
        for var, path in cache_env_vars.items():
            os.environ[var] = path
            Path(path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache redirected to: {cache_dir}")
        logger.info(f"Cache environment variables set: {list(cache_env_vars.keys())}")
    
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
        
        try:
            if temp_checkpoint_path.is_dir():
                if persistent_path.exists():
                    shutil.rmtree(persistent_path)
                shutil.copytree(temp_checkpoint_path, persistent_path)
            else:
                shutil.copy2(temp_checkpoint_path, persistent_path)
            
            logger.info(f"Checkpoint saved to persistent storage: {persistent_path}")
            return persistent_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint to persistent storage: {e}")
            raise
    
    def cleanup_temp_files(self, keep_patterns: Optional[list] = None):
        """Clean up temporary files, optionally keeping files matching patterns."""
        keep_patterns = keep_patterns or []
        
        working_dir = self.dirs['working']
        cleaned_size = 0
        
        try:
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
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def get_disk_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get disk usage information for all directories."""
        usage_info = {}
        
        for name, path in self.dirs.items():
            try:
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
                    usage_info[name] = {'exists': False, 'path': str(path)}
            except Exception as e:
                usage_info[name] = {'error': str(e), 'path': str(path)}
        
        # Get system disk usage for main paths
        system_usage = self._get_safe_disk_usage()
        for key, info in system_usage.items():
            usage_info[f'{key}_system'] = info
        
        return usage_info
    
    def print_status(self):
        """Print current status and usage information."""
        print(f"\n🗂️  BLIP3-o Workspace Status (Job {self.job_id})")
        print("=" * 70)
        
        # Directory structure
        print("📁 Directory Structure:")
        print(f"   Persistent Workspace: {self.persistent_workspace}")
        print(f"   Job Temp:            {self.job_temp}")
        
        # Check if we're using good storage locations
        if "scratch-shared" in str(self.persistent_workspace):
            print("   ✅ Using scratch-shared (good choice)")
        elif str(self.persistent_workspace).startswith(str(Path.home())):
            print("   ⚠️  Using home directory (quota risk)")
        
        # Directory details
        print("\n📊 Directory Usage:")
        usage = self.get_disk_usage()
        for name, info in usage.items():
            if info.get('exists', False) and 'system' not in name:
                size_gb = info.get('total_size_gb', 0)
                file_count = info.get('file_count', 0)
                print(f"   {name:20s}: {size_gb:8.2f} GB ({file_count:,} files)")
            elif info.get('error'):
                print(f"   {name:20s}: Error - {info['error']}")
        
        # System disk usage
        print("\n💾 System Disk Usage:")
        for name, info in usage.items():
            if 'system' in name and 'error' not in info:
                free_gb = info.get('free_gb', 0)
                usage_pct = info.get('usage_percent', 0)
                path = info.get('path', '')
                print(f"   {name:20s}: {free_gb:8.1f} GB free ({usage_pct:.1f}% used) - {path}")
        
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
        
        # Warnings
        if str(self.persistent_workspace).startswith(str(Path.home())):
            print("\n⚠️  WARNING: Using home directory for persistent storage")
            print("   Consider setting BLIP3O_WORKSPACE to a scratch directory")
        
        print("=" * 70)
    
    def create_job_script_snippet(self) -> str:
        """Generate bash snippet for job scripts."""
        return f'''
# BLIP3-o Workspace Setup (FIXED for Snellius)
export SCRATCH_SHARED="/scratch-shared"
export SCRATCH_LOCAL="/scratch-local"
export BLIP3O_USER=$(whoami)
export BLIP3O_JOB_ID=${{SLURM_JOB_ID}}

# Set up structured directories
export BLIP3O_WORKSPACE="/scratch-shared/${{BLIP3O_USER}}/blip3o_workspace"
export BLIP3O_EMBEDDINGS="${{BLIP3O_WORKSPACE}}/embeddings"
export BLIP3O_CHECKPOINTS="${{BLIP3O_WORKSPACE}}/checkpoints"
export BLIP3O_DATASETS="${{BLIP3O_WORKSPACE}}/datasets"
export BLIP3O_LOGS="${{BLIP3O_WORKSPACE}}/logs"

# Job temp directory
export BLIP3O_JOB_TEMP="/scratch-local/${{BLIP3O_USER}}.${{BLIP3O_JOB_ID}}/blip3o_job_${{BLIP3O_JOB_ID}}"
export BLIP3O_CACHE="${{BLIP3O_JOB_TEMP}}/cache"

# Create directories
mkdir -p "${{BLIP3O_WORKSPACE}}"{{datasets,embeddings,checkpoints,logs,metadata}}
mkdir -p "${{BLIP3O_JOB_TEMP}}"{{cache,working,temp_checkpoints}}

# Model cache (redirected to job temp to avoid home quota)
export TORCH_HOME="${{BLIP3O_CACHE}}/torch"
export HF_HOME="${{BLIP3O_CACHE}}/huggingface"
export TRANSFORMERS_CACHE="${{BLIP3O_CACHE}}/transformers"
export WANDB_DIR="${{BLIP3O_LOGS}}/wandb"

# Create cache subdirectories
mkdir -p "${{TORCH_HOME}}" "${{HF_HOME}}" "${{TRANSFORMERS_CACHE}}" "${{WANDB_DIR}}"

echo "🗂️  BLIP3-o workspace ready:"
echo "   Persistent: $BLIP3O_WORKSPACE"
echo "   Job temp:   $BLIP3O_JOB_TEMP"
'''

    def check_disk_quota_safety(self) -> Dict[str, Any]:
        """Check if we're at risk of hitting disk quotas."""
        safety_report = {
            'status': 'safe',
            'warnings': [],
            'recommendations': []
        }
        
        usage = self._get_safe_disk_usage()
        
        # Check home directory usage
        home_info = usage.get('home', {})
        if home_info.get('usage_percent', 0) > 80:
            safety_report['status'] = 'warning'
            safety_report['warnings'].append(f"Home directory {home_info['usage_percent']:.1f}% full")
            safety_report['recommendations'].append("Consider moving large files to scratch storage")
        
        # Check if using home for workspace
        if str(self.persistent_workspace).startswith(str(Path.home())):
            safety_report['status'] = 'risk'
            safety_report['warnings'].append("Using home directory for persistent workspace")
            safety_report['recommendations'].append("Set BLIP3O_WORKSPACE to use scratch-shared")
        
        # Check persistent workspace usage
        persistent_info = usage.get('persistent_workspace', {})
        if persistent_info.get('usage_percent', 0) > 90:
            safety_report['status'] = 'critical'
            safety_report['warnings'].append(f"Persistent workspace {persistent_info['usage_percent']:.1f}% full")
            safety_report['recommendations'].append("Clean up old files or use different workspace")
        
        return safety_report


def get_temp_manager(project_name: str = "blip3o_workspace") -> SnelliusTempManager:
    """Get or create temp manager instance."""
    return SnelliusTempManager(project_name)


def setup_snellius_environment(project_name: str = "blip3o_workspace") -> SnelliusTempManager:
    """FIXED: Setup complete Snellius environment for BLIP3-o project."""
    manager = get_temp_manager(project_name)
    manager.setup_model_cache()
    
    # Check disk quota safety
    safety = manager.check_disk_quota_safety()
    if safety['status'] != 'safe':
        logger.warning(f"Disk quota safety check: {safety['status']}")
        for warning in safety['warnings']:
            logger.warning(f"  Warning: {warning}")
        for rec in safety['recommendations']:
            logger.info(f"  Recommendation: {rec}")
    
    manager.print_status()
    return manager


if __name__ == "__main__":
    # Test the temp manager
    print("🧪 Testing FIXED temp manager...")
    manager = setup_snellius_environment()
    
    # Print job script snippet
    print("\n🔧 Add this to your job scripts:")
    print(manager.create_job_script_snippet())
    
    # Show safety report
    safety = manager.check_disk_quota_safety()
    print(f"\n🛡️  Disk quota safety: {safety['status']}")
    if safety['warnings']:
        print("   Warnings:")
        for warning in safety['warnings']:
            print(f"     • {warning}")
    if safety['recommendations']:
        print("   Recommendations:")
        for rec in safety['recommendations']:
            print(f"     • {rec}")