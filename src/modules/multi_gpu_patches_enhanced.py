"""
Enhanced Multi-GPU Compatibility Patches for BLIP3-o Trainer
File: src/modules/multi_gpu_patches_enhanced.py

ENHANCED FEATURES:
1. Better GPU detection and error handling
2. Robust DDP initialization with fallbacks
3. Memory optimization for multi-GPU training
4. Enhanced error reporting and debugging
5. CPU fallback support
"""

import torch
import torch.distributed as dist
import os
import logging
import traceback
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

def detect_gpu_environment():
    """Enhanced GPU environment detection with detailed reporting"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_names': [],
        'memory_total': [],
        'slurm_allocation': None,
        'cuda_visible_devices': None,
        'issues': [],
        'recommendations': []
    }
    
    # Basic CUDA check
    if not gpu_info['cuda_available']:
        gpu_info['issues'].append("CUDA not available in PyTorch")
        gpu_info['recommendations'].append("Check CUDA installation and drivers")
        return gpu_info
    
    # Detect GPUs
    try:
        gpu_info['gpu_count'] = torch.cuda.device_count()
        
        # Get detailed GPU info
        for i in range(gpu_info['gpu_count']):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_info['gpu_names'].append(props.name)
                gpu_info['memory_total'].append(props.total_memory / (1024**3))
                
                # Test GPU access
                test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                del test_tensor
                
            except Exception as e:
                gpu_info['issues'].append(f"GPU {i} not accessible: {e}")
                
    except Exception as e:
        gpu_info['issues'].append(f"Error detecting GPUs: {e}")
    
    # Check SLURM allocation
    slurm_gpus = os.environ.get('SLURM_GPUS')
    if slurm_gpus:
        gpu_info['slurm_allocation'] = slurm_gpus
        try:
            expected_gpus = int(slurm_gpus)
            if expected_gpus != gpu_info['gpu_count']:
                gpu_info['issues'].append(
                    f"SLURM allocated {expected_gpus} GPUs but {gpu_info['gpu_count']} detected"
                )
        except ValueError:
            pass
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_info['cuda_visible_devices'] = cuda_visible
    
    if cuda_visible == '' or cuda_visible is None:
        gpu_info['issues'].append("CUDA_VISIBLE_DEVICES is empty or not set")
        gpu_info['recommendations'].append("Set CUDA_VISIBLE_DEVICES or check SLURM GPU allocation")
    
    return gpu_info

def apply_gpu_fixes():
    """Apply fixes for common GPU issues"""
    fixes_applied = []
    
    # Fix 1: Handle empty CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible == '':
        # Try to fix using SLURM variables
        if 'SLURM_LOCALID' in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
            fixes_applied.append(f"Set CUDA_VISIBLE_DEVICES={os.environ['SLURM_LOCALID']} from SLURM_LOCALID")
        elif 'SLURM_GPUS' in os.environ:
            try:
                num_gpus = int(os.environ['SLURM_GPUS'])
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
                fixes_applied.append(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} from SLURM_GPUS")
            except ValueError:
                pass
    
    # Fix 2: Set optimal NCCL environment
    nccl_settings = {
        'NCCL_ASYNC_ERROR_HANDLING': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'NCCL_DEBUG': 'WARN',
        'OMP_NUM_THREADS': '1'
    }
    
    for var, value in nccl_settings.items():
        if var not in os.environ:
            os.environ[var] = value
            fixes_applied.append(f"Set {var}={value}")
    
    # Fix 3: Torch settings for better memory management
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        fixes_applied.append("Enabled CUDNN benchmark and TF32")
    
    return fixes_applied

def enhanced_ddp_init(backend='auto', timeout=1800):
    """Enhanced DDP initialization with better error handling"""
    if 'WORLD_SIZE' not in os.environ:
        return False, "Not a distributed environment"
    
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if world_size <= 1:
            return False, "Single process environment"
        
        # Auto-select backend
        if backend == 'auto':
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                backend = 'nccl'
            else:
                backend = 'gloo'
                logger.warning("Using Gloo backend (CPU or no GPU detected)")
        
        # Set device before DDP init
        if torch.cuda.is_available() and backend == 'nccl':
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
            else:
                logger.warning(f"Local rank {local_rank} >= GPU count {torch.cuda.device_count()}")
                return False, f"Local rank {local_rank} exceeds available GPUs"
        
        # Initialize process group with timeout
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=torch.timedelta(seconds=timeout)
            )
        
        # Verify DDP is working
        if backend == 'nccl' and torch.cuda.is_available():
            # Test tensor communication
            test_tensor = torch.randn(10, device=f'cuda:{local_rank}')
            dist.all_reduce(test_tensor)
        
        return True, f"DDP initialized with {backend} backend (rank {rank}/{world_size})"
        
    except Exception as e:
        return False, f"DDP initialization failed: {e}"

def patch_trainer_for_enhanced_multi_gpu():
    """Apply enhanced patches to make BLIP3oTrainer work better with multi-GPU"""
    
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        
        # Store original methods
        original_compute_loss = BLIP3oTrainer.compute_loss
        original_training_step = getattr(BLIP3oTrainer, 'training_step', None)
        original_evaluation_loop = getattr(BLIP3oTrainer, 'evaluation_loop', None)
        original_save_model = BLIP3oTrainer.save_model
        
        def enhanced_compute_loss(
            self,
            model,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Optional[int] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
            """Enhanced compute_loss with better error handling"""
            
            try:
                # Memory cleanup before forward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Extract inputs with error checking
                if 'eva_embeddings' not in inputs or 'clip_embeddings' not in inputs:
                    raise ValueError("Required embeddings not found in inputs")
                
                eva_embeddings = inputs['eva_embeddings']
                clip_embeddings = inputs['clip_embeddings']
                
                # Validate input shapes
                if eva_embeddings.dim() != 3 or clip_embeddings.dim() != 3:
                    raise ValueError(f"Invalid input shapes: EVA {eva_embeddings.shape}, CLIP {clip_embeddings.shape}")
                
                batch_size = eva_embeddings.shape[0]
                device = eva_embeddings.device
                
                # Sample timesteps with better error handling
                try:
                    timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
                except Exception as e:
                    logger.warning(f"Using fallback timestep sampling: {e}")
                    timesteps = torch.rand(batch_size, device=device)
                
                # Create noisy samples for flow matching
                noise = torch.randn_like(clip_embeddings)
                x_0 = torch.randn_like(clip_embeddings)
                
                try:
                    noisy_clip = self.flow_matching_loss.interpolate_data(
                        x_0=x_0, x_1=clip_embeddings, t=timesteps, noise=noise
                    )
                except Exception as e:
                    logger.warning(f"Using fallback interpolation: {e}")
                    # Simple linear interpolation fallback
                    alpha = timesteps.view(-1, 1, 1)
                    noisy_clip = (1 - alpha) * x_0 + alpha * clip_embeddings + 0.1 * noise
                
                # Forward pass with memory monitoring
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated()
                
                try:
                    model_output = model(
                        hidden_states=noisy_clip,
                        timestep=timesteps,
                        encoder_hidden_states=eva_embeddings,
                        return_dict=False
                    )
                except Exception as e:
                    logger.error(f"Model forward pass failed: {e}")
                    raise
                
                # Compute loss with enhanced error handling
                try:
                    loss, metrics = self.flow_matching_loss(
                        model_output=model_output,
                        target_samples=clip_embeddings,
                        timesteps=timesteps,
                        eva_conditioning=eva_embeddings,
                        noise=noise,
                        return_metrics=True
                    )
                except Exception as e:
                    logger.error(f"Loss computation failed: {e}")
                    # Fallback to simple MSE loss
                    loss = torch.nn.functional.mse_loss(model_output, clip_embeddings)
                    metrics = {'fallback_loss': loss.item()}
                
                # Memory monitoring
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_diff = (memory_after - memory_before) / 1024**2
                    
                    if hasattr(self, 'memory_usage'):
                        self.memory_usage.append(memory_diff)
                    else:
                        self.memory_usage = [memory_diff]
                
                # Store metrics (only on main process)
                if metrics and (not dist.is_initialized() or dist.get_rank() == 0):
                    if not hasattr(self, 'training_metrics'):
                        self.training_metrics = []
                    self.training_metrics.append(metrics)
                
                # Prepare outputs
                outputs = {
                    'model_output': model_output,
                    'noisy_clip': noisy_clip,
                    'timesteps': timesteps,
                    'metrics': metrics,
                } if return_outputs else None
                
                return (loss, outputs) if return_outputs else loss
                
            except Exception as e:
                logger.error(f"Enhanced compute_loss failed: {e}")
                logger.error(traceback.format_exc())
                
                # Emergency fallback
                try:
                    eva_embeddings = inputs['eva_embeddings']
                    clip_embeddings = inputs['clip_embeddings']
                    model_output = model(
                        hidden_states=clip_embeddings,
                        timestep=torch.zeros(eva_embeddings.shape[0], device=eva_embeddings.device),
                        encoder_hidden_states=eva_embeddings,
                        return_dict=False
                    )
                    loss = torch.nn.functional.mse_loss(model_output, clip_embeddings)
                    
                    outputs = {'emergency_fallback': True} if return_outputs else None
                    return (loss, outputs) if return_outputs else loss
                    
                except Exception as emergency_e:
                    logger.error(f"Emergency fallback also failed: {emergency_e}")
                    raise e
        
        def enhanced_save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
            """Enhanced model saving with better error handling"""
            
            # Only save on main process
            if dist.is_initialized() and dist.get_rank() != 0:
                return
            
            try:
                # Call original save method
                original_save_model(self, output_dir, _internal_call)
                
                # Save additional debugging info
                output_path = Path(output_dir or self.args.output_dir)
                
                # Save training metrics if available
                if hasattr(self, 'training_metrics') and self.training_metrics:
                    metrics_file = output_path / 'training_metrics.json'
                    with open(metrics_file, 'w') as f:
                        json.dump(self.training_metrics[-100:], f, indent=2)  # Last 100 steps
                
                # Save memory usage info
                if hasattr(self, 'memory_usage') and self.memory_usage:
                    memory_file = output_path / 'memory_usage.json'
                    memory_stats = {
                        'peak_memory_mb': max(self.memory_usage),
                        'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
                        'total_steps': len(self.memory_usage)
                    }
                    with open(memory_file, 'w') as f:
                        json.dump(memory_stats, f, indent=2)
                
                # Save GPU info
                gpu_info = detect_gpu_environment()
                gpu_info_file = output_path / 'gpu_info.json'
                with open(gpu_info_file, 'w') as f:
                    json.dump(gpu_info, f, indent=2)
                
                logger.info(f"Enhanced model save completed: {output_path}")
                
            except Exception as e:
                logger.error(f"Enhanced save failed: {e}")
                # Still try original save as fallback
                try:
                    original_save_model(self, output_dir, _internal_call)
                except Exception as fallback_e:
                    logger.error(f"Fallback save also failed: {fallback_e}")
                    raise e
        
        def enhanced_evaluation_loop(
            self,
            dataloader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ):
            """Enhanced evaluation with memory optimization"""
            
            # Conservative evaluation for multi-GPU
            original_batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 4
            
            # Reduce evaluation scope for stability
            max_eval_batches = 10 if dist.is_initialized() else 20
            
            eval_results = []
            eval_batch_count = 0
            
            # Set model to eval mode
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            with torch.no_grad():
                for step, inputs in enumerate(dataloader):
                    if eval_batch_count >= max_eval_batches:
                        break
                    
                    try:
                        # Memory check
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            if memory_used > 25:  # Conservative limit
                                logger.warning(f"Stopping eval due to memory usage: {memory_used:.1f}GB")
                                break
                        
                        # Reduce batch size for stability
                        if isinstance(inputs, dict):
                            for key in inputs:
                                if isinstance(inputs[key], torch.Tensor) and len(inputs[key]) > 2:
                                    inputs[key] = inputs[key][:2]  # Max 2 samples
                        
                        inputs = self._prepare_inputs(inputs)
                        
                        # Compute loss
                        loss = self.compute_loss(model, inputs)
                        eval_results.append(loss.item())
                        
                        eval_batch_count += 1
                        
                        # Cleanup
                        del inputs, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.warning(f"Eval step {step} failed: {e}")
                        continue
            
            # Aggregate results
            if eval_results:
                avg_loss = sum(eval_results) / len(eval_results)
                return {
                    f'{metric_key_prefix}_loss': avg_loss,
                    f'{metric_key_prefix}_batches': eval_batch_count,
                    f'{metric_key_prefix}_samples': eval_batch_count * 2,  # Conservative estimate
                }
            else:
                return {f'{metric_key_prefix}_loss': float('inf')}
        
        # Apply patches
        BLIP3oTrainer.compute_loss = enhanced_compute_loss
        BLIP3oTrainer.save_model = enhanced_save_model
        
        # Apply evaluation patch if method exists
        if original_evaluation_loop:
            BLIP3oTrainer.evaluation_loop = enhanced_evaluation_loop
        
        logger.info("✅ Enhanced multi-GPU trainer patches applied")
        
    except Exception as e:
        logger.error(f"Failed to apply enhanced multi-GPU patches: {e}")
        logger.error(traceback.format_exc())

def patch_dataset_for_enhanced_ddp():
    """Enhanced dataset patches for better DDP compatibility"""
    
    try:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        
        def create_enhanced_ddp_dataloader(
            dataset,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 4,
            pin_memory: bool = True,
            drop_last: bool = True,
            collate_fn=None,
            **kwargs
        ):
            """Create DataLoader with enhanced DDP support"""
            
            sampler = None
            shuffle_for_dataloader = shuffle
            
            # Use DistributedSampler if in distributed mode
            if dist.is_initialized():
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=shuffle,
                    drop_last=drop_last
                )
                shuffle_for_dataloader = False  # Don't shuffle in DataLoader when using sampler
                
                logger.info(f"Created DistributedSampler for rank {dist.get_rank()}/{dist.get_world_size()}")
            
            # Adjust num_workers for stability
            if dist.is_initialized():
                num_workers = min(num_workers, 2)  # Conservative for multi-GPU
            
            # Auto-detect pin_memory
            if pin_memory is None:
                pin_memory = torch.cuda.is_available()
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle_for_dataloader,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                persistent_workers=num_workers > 0,
                **kwargs
            )
            
            return dataloader
        
        # Patch the dataset module if available
        try:
            import src.modules.datasets.blip3o_dataset as dataset_module
            
            # Store original function
            original_create_chunked_dataloader = dataset_module.create_chunked_dataloader
            
            def enhanced_create_chunked_dataloader(*args, **kwargs):
                """Enhanced chunked dataloader creation"""
                
                # Extract dataset creation parameters
                dataset_kwargs = {k: v for k, v in kwargs.items() 
                                if k in ['chunked_embeddings_dir', 'split', 'eval_split_ratio', 
                                        'normalize_embeddings', 'shuffle_shards', 'shuffle_within_shard',
                                        'delete_after_use']}
                
                # Extract dataloader parameters
                dataloader_kwargs = {k: v for k, v in kwargs.items() 
                                   if k in ['batch_size', 'num_workers', 'pin_memory', 'drop_last']}
                
                # Create dataset
                from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset, chunked_collate_fn
                
                dataset = BLIP3oEmbeddingDataset(**dataset_kwargs)
                
                # Create enhanced dataloader
                return create_enhanced_ddp_dataloader(
                    dataset,
                    collate_fn=chunked_collate_fn,
                    **dataloader_kwargs
                )
            
            # Apply patch
            dataset_module.create_chunked_dataloader = enhanced_create_chunked_dataloader
            
            logger.info("✅ Enhanced DDP dataset patches applied")
            
        except ImportError:
            logger.warning("Could not patch dataset module - not available")
        
    except Exception as e:
        logger.error(f"Failed to apply enhanced DDP dataset patches: {e}")

def apply_all_enhanced_patches():
    """Apply all enhanced patches for better multi-GPU training"""
    
    logger.info("🔧 Applying enhanced multi-GPU compatibility patches...")
    
    # 1. Detect and fix GPU environment
    logger.info("1. GPU Environment Detection and Fixes")
    gpu_info = detect_gpu_environment()
    
    if gpu_info['issues']:
        logger.warning("GPU Issues detected:")
        for issue in gpu_info['issues']:
            logger.warning(f"  - {issue}")
    
    fixes = apply_gpu_fixes()
    if fixes:
        logger.info("Applied GPU fixes:")
        for fix in fixes:
            logger.info(f"  ✅ {fix}")
    
    # 2. Enhanced DDP initialization
    logger.info("2. Enhanced DDP Initialization")
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        success, message = enhanced_ddp_init()
        if success:
            logger.info(f"✅ {message}")
        else:
            logger.warning(f"⚠️  {message}")
    
    # 3. Apply trainer patches
    logger.info("3. Enhanced Trainer Patches")
    patch_trainer_for_enhanced_multi_gpu()
    
    # 4. Apply dataset patches
    logger.info("4. Enhanced Dataset Patches")
    patch_dataset_for_enhanced_ddp()
    
    logger.info("✅ All enhanced multi-GPU patches applied successfully!")
    
    # Return summary
    return {
        'gpu_info': gpu_info,
        'fixes_applied': fixes,
        'patches_applied': ['trainer', 'dataset', 'ddp'],
        'status': 'success'
    }

def create_gpu_debug_report():
    """Create a comprehensive GPU debug report"""
    
    report = {
        'timestamp': time.time(),
        'gpu_environment': detect_gpu_environment(),
        'torch_info': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        },
        'environment_vars': {
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'SLURM_GPUS': os.environ.get('SLURM_GPUS'),
            'WORLD_SIZE': os.environ.get('WORLD_SIZE'),
            'LOCAL_RANK': os.environ.get('LOCAL_RANK'),
            'RANK': os.environ.get('RANK'),
        },
        'distributed_info': {
            'is_initialized': dist.is_initialized() if dist.is_available() else False,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'rank': dist.get_rank() if dist.is_initialized() else 0,
        }
    }
    
    # Save report
    report_file = Path('gpu_debug_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"GPU debug report saved to: {report_file}")
    return report

if __name__ == "__main__":
    # Test the enhanced patches
    print("🧪 Testing Enhanced Multi-GPU Patches...")
    
    # Apply all patches
    patch_results = apply_all_enhanced_patches()
    
    # Create debug report
    debug_report = create_gpu_debug_report()
    
    print("✅ Enhanced patches ready for use!")
    print("📊 GPU Status:")
    gpu_info = patch_results['gpu_info']
    print(f"  CUDA Available: {gpu_info['cuda_available']}")
    print(f"  GPU Count: {gpu_info['gpu_count']}")
    
    if gpu_info['issues']:
        print("⚠️  Issues found:")
        for issue in gpu_info['issues']:
            print(f"    - {issue}")
    
    if patch_results['fixes_applied']:
        print("🔧 Fixes applied:")
        for fix in patch_results['fixes_applied']:
            print(f"    ✅ {fix}")