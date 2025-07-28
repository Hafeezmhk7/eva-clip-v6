#!/usr/bin/env python3
"""
FIXED: BLIP3-o Trainer with Consistent Training/Evaluation Data
Key fixes:
1. Validation that training and evaluation data have consistent norms
2. Detailed logging of data statistics during training and evaluation  
3. Early detection of norm mismatches
4. Consistent noise scaling between training and inference
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import numpy as np
from pathlib import Path
import json
import gc
from collections import deque
import math
import os

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class BLIP3oCLIPTrainer:
    """
    FIXED: Trainer with consistent training/evaluation data validation
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        # Evaluation
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        # Noise scaling configuration
        sync_noise_scale_every: int = 10,
        enable_generation_debug: bool = False,
        # NEW: Data consistency validation
        validate_data_consistency: bool = True,
        log_data_statistics: bool = True,
        norm_tolerance: float = 5.0,  # Tolerance for norm differences
        # Debugging
        debug_mode: bool = False,
        overfit_test_size: Optional[int] = None,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        # Output
        output_dir: str = "./checkpoints",
        # Device
        device: Optional[torch.device] = None,
        # WandB configuration
        use_wandb: bool = True,
        wandb_project: str = "blip3o-clip-reproduction",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_api_key: Optional[str] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        
        # Evaluation config
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_inference_steps = eval_inference_steps
        
        # Noise scaling config
        self.sync_noise_scale_every = sync_noise_scale_every
        self.enable_generation_debug = enable_generation_debug
        
        # NEW: Data consistency validation
        self.validate_data_consistency = validate_data_consistency
        self.log_data_statistics = log_data_statistics
        self.norm_tolerance = norm_tolerance
        
        # Debugging config
        self.debug_mode = debug_mode
        self.overfit_test_size = overfit_test_size
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self.model.to(self.device)
        
        # WandB configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.wandb_api_key = wandb_api_key
        
        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.noise_scale_history = deque(maxlen=1000)
        self.last_noise_scale_sync = 0
        
        # NEW: Data statistics tracking
        self.train_clip_norms = deque(maxlen=1000)
        self.train_eva_norms = deque(maxlen=1000)
        self.eval_clip_norms = deque(maxlen=100)
        self.eval_eva_norms = deque(maxlen=100)
        self.norm_consistency_warnings = 0
        
        # Estimate steps per epoch
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup WandB
        if self.use_wandb:
            self._setup_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
        
        # Enable generation debug if requested
        if self.enable_generation_debug and hasattr(self.model, 'enable_generation_debug'):
            self.model.enable_generation_debug()
        
        # Overfitting test data
        self.overfit_batch = None
        if self.overfit_test_size:
            self._prepare_overfit_test()
        
        # NEW: Validate data consistency on initialization
        if self.validate_data_consistency and self.eval_dataloader:
            self._validate_initial_data_consistency()
        
        logger.info("BLIP3-o CLIP Trainer initialized with data consistency validation")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Estimated steps per epoch: {self.estimated_steps_per_epoch}")
        logger.info(f"  🎯 DATA CONSISTENCY: {self.validate_data_consistency}")
        logger.info(f"  📊 STATISTICS LOGGING: {self.log_data_statistics}")
        logger.info(f"  ⚖️ NORM TOLERANCE: {self.norm_tolerance}")
        logger.info(f"  Overfit test: {self.overfit_test_size if self.overfit_test_size else 'Disabled'}")
        logger.info(f"  Mixed precision: {self.fp16}")
        logger.info(f"  WandB logging: {self.use_wandb}")

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch"""
        try:
            length = len(self.train_dataloader)
            logger.info(f"Got exact dataloader length: {length}")
            return length
        except TypeError:
            try:
                dataset_length = len(self.train_dataloader.dataset)
                batch_size = getattr(self.train_dataloader, 'batch_size', 1)
                estimated_steps = max(1, dataset_length // batch_size)
                logger.info(f"Estimated steps per epoch from dataset length: {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                logger.warning("Could not estimate steps per epoch, using default: 100")
                return 100

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = self.estimated_steps_per_epoch * self.num_epochs
        
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.01
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"  Total estimated steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

    def _setup_wandb(self):
        """Setup WandB with comprehensive configuration"""
        try:
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key
            elif "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_API_KEY"] = "0d9895af249ee18e4fa141e8a2350e0f4adb920f"
            
            # Get model config
            model_config = {}
            if hasattr(self.model, 'config'):
                model_config = {
                    'model_type': getattr(self.model.config, 'model_type', 'blip3o_clip_dit'),
                    'hidden_size': getattr(self.model.config, 'hidden_size', 768),
                    'num_hidden_layers': getattr(self.model.config, 'num_hidden_layers', 12),
                    'num_attention_heads': getattr(self.model.config, 'num_attention_heads', 12),
                    'use_3d_rope': getattr(self.model.config, 'use_3d_rope', True),
                    'use_sandwich_norm': getattr(self.model.config, 'use_sandwich_norm', True),
                    'training_mode': getattr(self.model.config, 'training_mode', 'patch_only'),
                    'eva_embedding_size': getattr(self.model.config, 'eva_embedding_size', 4096),
                    'clip_embedding_size': getattr(self.model.config, 'clip_embedding_size', 1024),
                }
            
            # Get loss function config
            loss_config = {}
            if hasattr(self.loss_fn, 'use_adaptive_noise_scaling'):
                loss_config = {
                    'adaptive_noise_scaling': getattr(self.loss_fn, 'use_adaptive_noise_scaling', False),
                    'noise_scale_momentum': getattr(self.loss_fn, 'noise_scale_momentum', 0.99),
                    'prediction_type': getattr(self.loss_fn, 'prediction_type', 'velocity'),
                    'flow_type': getattr(self.loss_fn, 'flow_type', 'rectified'),
                }
            
            # Create comprehensive WandB config
            wandb_config = {
                # Training hyperparameters
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'fp16': self.fp16,
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                
                # Evaluation config
                'eval_every_n_steps': self.eval_every_n_steps,
                'eval_num_samples': self.eval_num_samples,
                'eval_inference_steps': self.eval_inference_steps,
                
                # Noise scaling config
                'sync_noise_scale_every': self.sync_noise_scale_every,
                'enable_generation_debug': self.enable_generation_debug,
                
                # NEW: Data consistency config
                'validate_data_consistency': self.validate_data_consistency,
                'log_data_statistics': self.log_data_statistics,
                'norm_tolerance': self.norm_tolerance,
                
                # Model architecture
                **model_config,
                
                # Loss function config
                **loss_config,
                
                # Experiment details
                'experiment_type': 'clip_reproduction_with_consistent_evaluation',
                'task': 'EVA_to_CLIP_embedding_reproduction',
                'method': 'BLIP3o_DiT_with_rectified_flow_and_consistent_data',
                'normalization_approach': 'minimal_with_data_consistency_validation',
                'overfit_test_size': self.overfit_test_size,
                'debug_mode': self.debug_mode,
                
                # Data consistency features
                'consistent_evaluation': True,
                'norm_validation': True,
                'statistics_tracking': True,
                
                **self.wandb_config,
            }
            
            # Safely get dataset info
            try:
                if hasattr(self.train_dataloader, 'batch_size'):
                    wandb_config['batch_size'] = self.train_dataloader.batch_size
                if hasattr(self.train_dataloader, 'dataset'):
                    dataset = self.train_dataloader.dataset
                    if hasattr(dataset, 'training_mode'):
                        wandb_config['training_mode'] = dataset.training_mode
                    if hasattr(dataset, 'normalize_embeddings'):
                        wandb_config['normalize_embeddings'] = dataset.normalize_embeddings
                    if hasattr(dataset, 'expected_tokens'):
                        wandb_config['expected_tokens'] = dataset.expected_tokens
            except Exception as e:
                logger.warning(f"Could not extract dataset info for WandB: {e}")
            
            # Initialize WandB run
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "clip_reproduction", "eva_conditioning", "data_consistency", "norm_validation"]
            )
            
            # Log model architecture
            if hasattr(self.model, 'get_num_parameters'):
                wandb.log({"model/total_parameters": self.model.get_num_parameters()})
            
            # Watch model
            wandb.watch(self.model, log="all", log_freq=self.log_every_n_steps)
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            logger.info(f"   Run ID: {self.wandb_run.id}")
            logger.info(f"   Run URL: {self.wandb_run.url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            logger.error("Continuing without WandB logging...")
            self.use_wandb = False

    def _prepare_overfit_test(self):
        """Prepare overfitting test batch"""
        logger.info(f"Preparing overfitting test with {self.overfit_test_size} samples...")
        
        try:
            first_batch = next(iter(self.train_dataloader))
            actual_size = min(self.overfit_test_size, first_batch['batch_size'])
            
            self.overfit_batch = {}
            for key, value in first_batch.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    self.overfit_batch[key] = value[:actual_size].clone().detach()
                elif isinstance(value, list):
                    self.overfit_batch[key] = value[:actual_size]
                else:
                    self.overfit_batch[key] = value
            
            self.overfit_batch['batch_size'] = actual_size
            
            logger.info(f"Overfitting test prepared with {actual_size} samples")
            
            if self.use_wandb:
                wandb.log({
                    "overfit_test/enabled": True,
                    "overfit_test/size": actual_size,
                    "overfit_test/step": 0
                })
            
        except Exception as e:
            logger.error(f"Failed to prepare overfitting test: {e}")
            self.overfit_batch = None

    def _validate_initial_data_consistency(self):
        """Validate that training and evaluation data have consistent statistics"""
        logger.info("🔍 Validating initial data consistency...")
        
        try:
            # Get a few batches from training data
            train_clip_norms = []
            train_eva_norms = []
            train_batches_checked = 0
            
            for batch in self.train_dataloader:
                if train_batches_checked >= 3:  # Check first 3 batches
                    break
                
                clip_norms = torch.norm(batch['clip_embeddings'], dim=-1).mean(dim=1)
                eva_norms = torch.norm(batch['encoder_hidden_states'], dim=-1).mean(dim=1)
                
                train_clip_norms.extend(clip_norms.cpu().numpy())
                train_eva_norms.extend(eva_norms.cpu().numpy())
                train_batches_checked += 1
            
            # Get a few batches from evaluation data
            eval_clip_norms = []
            eval_eva_norms = []
            eval_batches_checked = 0
            
            for batch in self.eval_dataloader:
                if eval_batches_checked >= 3:  # Check first 3 batches
                    break
                
                clip_norms = torch.norm(batch['clip_embeddings'], dim=-1).mean(dim=1)
                eva_norms = torch.norm(batch['encoder_hidden_states'], dim=-1).mean(dim=1)
                
                eval_clip_norms.extend(clip_norms.cpu().numpy())
                eval_eva_norms.extend(eva_norms.cpu().numpy())
                eval_batches_checked += 1
            
            # Compute statistics
            train_clip_mean = np.mean(train_clip_norms)
            train_eva_mean = np.mean(train_eva_norms)
            eval_clip_mean = np.mean(eval_clip_norms)
            eval_eva_mean = np.mean(eval_eva_norms)
            
            clip_diff = abs(train_clip_mean - eval_clip_mean)
            eva_diff = abs(train_eva_mean - eval_eva_mean)
            
            logger.info(f"📊 INITIAL DATA CONSISTENCY CHECK:")
            logger.info(f"   Training CLIP norm: {train_clip_mean:.2f} ± {np.std(train_clip_norms):.2f}")
            logger.info(f"   Evaluation CLIP norm: {eval_clip_mean:.2f} ± {np.std(eval_clip_norms):.2f}")
            logger.info(f"   Training EVA norm: {train_eva_mean:.2f} ± {np.std(train_eva_norms):.2f}")
            logger.info(f"   Evaluation EVA norm: {eval_eva_mean:.2f} ± {np.std(eval_eva_norms):.2f}")
            logger.info(f"   CLIP difference: {clip_diff:.2f}")
            logger.info(f"   EVA difference: {eva_diff:.2f}")
            
            # Check for consistency
            if clip_diff <= self.norm_tolerance and eva_diff <= self.norm_tolerance:
                logger.info(f"✅ DATA CONSISTENCY: Differences within tolerance ({self.norm_tolerance})")
                consistency_status = "good"
            elif clip_diff <= self.norm_tolerance * 2 and eva_diff <= self.norm_tolerance * 2:
                logger.warning(f"⚠️ DATA CONSISTENCY: Moderate differences detected")
                consistency_status = "moderate"
            else:
                logger.error(f"🚨 DATA CONSISTENCY: LARGE differences detected!")
                logger.error(f"   This may cause the training/evaluation norm mismatch issue!")
                consistency_status = "poor"
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "data_consistency/initial_check": True,
                    "data_consistency/train_clip_norm": train_clip_mean,
                    "data_consistency/eval_clip_norm": eval_clip_mean,
                    "data_consistency/train_eva_norm": train_eva_mean,
                    "data_consistency/eval_eva_norm": eva_eva_mean,
                    "data_consistency/clip_difference": clip_diff,
                    "data_consistency/eva_difference": eva_diff,
                    "data_consistency/status": consistency_status,
                    "data_consistency/within_tolerance": (clip_diff <= self.norm_tolerance and eva_diff <= self.norm_tolerance),
                }, step=0)
            
            # Store for tracking
            self.initial_train_clip_norm = train_clip_mean
            self.initial_eval_clip_norm = eval_clip_mean
            self.initial_train_eva_norm = train_eva_mean
            self.initial_eval_eva_norm = eval_eva_mean
            
        except Exception as e:
            logger.error(f"Failed to validate initial data consistency: {e}")

    def _sync_noise_scale(self):
        """Synchronize noise scale from loss function to model"""
        if hasattr(self.loss_fn, 'get_noise_scale') and hasattr(self.model, 'set_noise_scale'):
            noise_scale = self.loss_fn.get_noise_scale()
            if noise_scale > 0:
                self.model.set_noise_scale(noise_scale)
                self.noise_scale_history.append(noise_scale)
                
                if self.debug_mode and self.global_step % 100 == 0:
                    logger.debug(f"Synced noise scale: {noise_scale:.3f}")
                
                return noise_scale
        return None

    def _compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with data statistics logging"""
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        # Use overfit batch if specified
        if self.overfit_batch is not None:
            for key, value in self.overfit_batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
                else:
                    batch[key] = value
        
        # NEW: Log batch statistics
        if self.log_data_statistics:
            clip_norm_mean = batch.get('clip_norm_mean', 0)
            eva_norm_mean = batch.get('eva_norm_mean', 0)
            
            self.train_clip_norms.append(clip_norm_mean)
            self.train_eva_norms.append(eva_norm_mean)
            
            # Log detailed statistics
            if self.global_step % 50 == 0:
                logger.debug(f"Training batch statistics:")
                logger.debug(f"  CLIP norm: {clip_norm_mean:.2f}")
                logger.debug(f"  EVA norm: {eva_norm_mean:.2f}")
                logger.debug(f"  Batch size: {batch.get('batch_size', 0)}")
        
        # Extract inputs
        hidden_states = batch['hidden_states']
        timestep = batch['timestep']
        encoder_hidden_states = batch['encoder_hidden_states']
        clip_embeddings = batch['clip_embeddings']
        noise = batch.get('noise')
        
        # Forward pass
        if self.fp16:
            with torch.cuda.amp.autocast():
                model_output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )
                
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=clip_embeddings,
                    timesteps=timestep,
                    eva_conditioning=encoder_hidden_states,
                    noise=noise,
                    return_metrics=True
                )
        else:
            model_output = self.model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
            
            loss, metrics = self.loss_fn(
                model_output=model_output,
                target_samples=clip_embeddings,
                timesteps=timestep,
                eva_conditioning=encoder_hidden_states,
                noise=noise,
                return_metrics=True
            )
        
        # Sync noise scale periodically
        if (self.global_step - self.last_noise_scale_sync) >= self.sync_noise_scale_every:
            synced_scale = self._sync_noise_scale()
            if synced_scale is not None:
                self.last_noise_scale_sync = self.global_step
                if metrics:
                    metrics['noise_scale_synced'] = synced_scale
        
        # Add batch statistics to metrics
        if metrics and self.log_data_statistics:
            metrics.update({
                'batch_clip_norm': batch.get('clip_norm_mean', 0),
                'batch_eva_norm': batch.get('eva_norm_mean', 0),
                'batch_clip_std': batch.get('clip_std', 0),
                'batch_eva_std': batch.get('eva_std', 0),
            })
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Backward pass and optimizer step"""
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            if self.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return grad_norm

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation with data consistency validation"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        self.model.eval()
        
        all_similarities = []
        all_mse_losses = []
        all_generated_norms = []
        all_target_norms = []
        eval_clip_norms = []
        eval_eva_norms = []
        samples_processed = 0
        
        eval_start_time = time.time()
        
        # Get current noise scale
        noise_scale = None
        if hasattr(self.loss_fn, 'get_noise_scale'):
            noise_scale = self.loss_fn.get_noise_scale()
            if noise_scale <= 0:
                noise_scale = None
        
        if self.debug_mode and noise_scale:
            logger.debug(f"Using noise scale {noise_scale:.3f} for evaluation")
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                # Move to device
                eva_features = batch['encoder_hidden_states'].to(self.device)
                target_clip = batch['clip_embeddings'].to(self.device)
                
                # NEW: Log evaluation data statistics
                if self.log_data_statistics:
                    eval_clip_norm = torch.norm(target_clip, dim=-1).mean().item()
                    eval_eva_norm = torch.norm(eva_features, dim=-1).mean().item()
                    eval_clip_norms.append(eval_clip_norm)
                    eval_eva_norms.append(eval_eva_norm)
                    
                    # Store for consistency checking
                    self.eval_clip_norms.append(eval_clip_norm)
                    self.eval_eva_norms.append(eval_eva_norm)
                
                # Generate CLIP embeddings
                generated_clip = self.model.generate(
                    eva_features=eva_features,
                    num_inference_steps=self.eval_inference_steps,
                    normalize_output=False,
                    noise_scale=noise_scale,
                )
                
                # Compute similarity (normalize ONLY for similarity computation)
                target_norm = F.normalize(target_clip, p=2, dim=-1)
                generated_norm = F.normalize(generated_clip, p=2, dim=-1)
                similarity = F.cosine_similarity(generated_norm, target_norm, dim=-1)
                per_image_similarity = similarity.mean(dim=1)
                
                # Compute MSE loss in raw space
                mse_loss = F.mse_loss(generated_clip, target_clip, reduction='none').mean(dim=(1, 2))
                
                # Track norms for analysis
                generated_norms = torch.norm(generated_clip, dim=-1).mean(dim=1)
                target_norms = torch.norm(target_clip, dim=-1).mean(dim=1)
                
                all_similarities.append(per_image_similarity.cpu())
                all_mse_losses.append(mse_loss.cpu())
                all_generated_norms.append(generated_norms.cpu())
                all_target_norms.append(target_norms.cpu())
                samples_processed += eva_features.shape[0]
        
        self.model.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        all_mse = torch.cat(all_mse_losses)
        all_gen_norms = torch.cat(all_generated_norms)
        all_tgt_norms = torch.cat(all_target_norms)
        
        eval_time = time.time() - eval_start_time
        
        eval_metrics = {
            'eval_clip_similarity': all_sims.mean().item(),
            'eval_clip_similarity_std': all_sims.std().item(),
            'eval_clip_similarity_min': all_sims.min().item(),
            'eval_clip_similarity_max': all_sims.max().item(),
            'eval_mse_loss': all_mse.mean().item(),
            'eval_mse_loss_std': all_mse.std().item(),
            'eval_high_quality': (all_sims > 0.7).float().mean().item(),
            'eval_very_high_quality': (all_sims > 0.8).float().mean().item(),
            'eval_excellent_quality': (all_sims > 0.9).float().mean().item(),
            'eval_samples': samples_processed,
            'eval_time_seconds': eval_time,
            'eval_samples_per_second': samples_processed / eval_time if eval_time > 0 else 0,
            
            # Norm analysis
            'eval_generated_norm_mean': all_gen_norms.mean().item(),
            'eval_generated_norm_std': all_gen_norms.std().item(),
            'eval_target_norm_mean': all_tgt_norms.mean().item(),
            'eval_target_norm_std': all_tgt_norms.std().item(),
            'eval_norm_ratio': all_gen_norms.mean().item() / (all_tgt_norms.mean().item() + 1e-8),
            'eval_noise_scale_used': noise_scale if noise_scale else 0.0,
        }
        
        # NEW: Data consistency validation during evaluation
        if self.validate_data_consistency and eval_clip_norms and len(self.train_clip_norms) > 0:
            train_clip_mean = np.mean(list(self.train_clip_norms)[-100:])  # Recent training norms
            eval_clip_mean = np.mean(eval_clip_norms)
            clip_diff = abs(train_clip_mean - eval_clip_mean)
            
            eval_metrics.update({
                'eval_data_clip_norm': eval_clip_mean,
                'train_data_clip_norm': train_clip_mean,
                'train_eval_clip_diff': clip_diff,
                'data_consistency_good': clip_diff <= self.norm_tolerance,
            })
            
            if clip_diff > self.norm_tolerance:
                self.norm_consistency_warnings += 1
                logger.warning(f"🚨 NORM MISMATCH DETECTED!")
                logger.warning(f"   Training CLIP norm: {train_clip_mean:.2f}")
                logger.warning(f"   Evaluation CLIP norm: {eval_clip_mean:.2f}")
                logger.warning(f"   Difference: {clip_diff:.2f} (tolerance: {self.norm_tolerance})")
                logger.warning(f"   This may explain low CLIP similarity!")
                
                if self.use_wandb:
                    wandb.log({
                        "consistency_warning/detected": True,
                        "consistency_warning/step": self.global_step,
                        "consistency_warning/train_norm": train_clip_mean,
                        "consistency_warning/eval_norm": eval_clip_mean,
                        "consistency_warning/difference": clip_diff,
                        "consistency_warning/count": self.norm_consistency_warnings,
                    }, step=self.global_step)
        
        return eval_metrics

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log training metrics with data statistics"""
        # Store metrics
        self.loss_history.append(loss)
        if 'velocity_similarity' in metrics:
            self.similarity_history.append(metrics['velocity_similarity'])
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        if 'velocity_similarity' in metrics:
            if metrics['velocity_similarity'] > self.best_eval_similarity:
                self.best_eval_similarity = metrics['velocity_similarity']
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Prepare WandB metrics
        wandb_metrics = {}
        if self.use_wandb:
            # Training metrics
            wandb_metrics.update({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                "train/epoch": self.current_epoch,
                "train/step": self.global_step,
            })
            
            # Loss function specific metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        if key.startswith('eval_'):
                            wandb_metrics[f"eval/{key[5:]}"] = value
                        else:
                            wandb_metrics[f"train/{key}"] = value
            
            # Data statistics
            if self.log_data_statistics:
                if len(self.train_clip_norms) > 0:
                    recent_train_clip = np.mean(list(self.train_clip_norms)[-10:])
                    wandb_metrics["data/train_clip_norm"] = recent_train_clip
                
                if len(self.train_eva_norms) > 0:
                    recent_train_eva = np.mean(list(self.train_eva_norms)[-10:])
                    wandb_metrics["data/train_eva_norm"] = recent_train_eva
                
                if len(self.eval_clip_norms) > 0:
                    recent_eval_clip = np.mean(list(self.eval_clip_norms)[-10:])
                    wandb_metrics["data/eval_clip_norm"] = recent_eval_clip
                    
                    # Consistency metrics
                    if len(self.train_clip_norms) > 0:
                        recent_train_clip = np.mean(list(self.train_clip_norms)[-10:])
                        norm_diff = abs(recent_train_clip - recent_eval_clip)
                        wandb_metrics["data/train_eval_norm_diff"] = norm_diff
                        wandb_metrics["data/consistency_good"] = norm_diff <= self.norm_tolerance
            
            # Moving averages
            if len(self.loss_history) > 0:
                wandb_metrics["train/loss_ma"] = np.mean(list(self.loss_history))
            if len(self.similarity_history) > 0:
                wandb_metrics["train/similarity_ma"] = np.mean(list(self.similarity_history))
            if len(self.grad_norm_history) > 0:
                wandb_metrics["train/grad_norm_ma"] = np.mean(list(self.grad_norm_history))
            if len(self.noise_scale_history) > 0:
                wandb_metrics["train/noise_scale_ma"] = np.mean(list(self.noise_scale_history))
            
            # Best metrics
            wandb_metrics["train/best_loss"] = self.best_loss
            wandb_metrics["train/best_similarity"] = self.best_eval_similarity
            
            # Consistency warnings
            wandb_metrics["data/consistency_warnings"] = self.norm_consistency_warnings
            
            # System metrics
            if torch.cuda.is_available():
                wandb_metrics["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                wandb_metrics["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
            
            # Log to WandB
            wandb.log(wandb_metrics, step=self.global_step)
        
        # Log to console
        if self.global_step % self.log_every_n_steps == 0:
            log_msg = f"Step {self.global_step}: Loss={loss:.6f}"
            
            if 'velocity_similarity' in metrics:
                sim = metrics['velocity_similarity']
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", VelSim={sim:.4f} ({quality})"
            
            log_msg += f", GradNorm={grad_norm:.3f}"
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            
            # Show noise scaling info
            if 'noise_scale' in metrics:
                noise_scale = metrics['noise_scale']
                log_msg += f", NoiseScale={noise_scale:.3f}"
            
            # Show norm statistics
            if self.log_data_statistics and 'batch_clip_norm' in metrics:
                clip_norm = metrics['batch_clip_norm']
                eva_norm = metrics['batch_eva_norm']
                log_msg += f", CLIP={clip_norm:.2f}, EVA={eva_norm:.2f}"
            
            # Show consistency warnings
            if self.norm_consistency_warnings > 0:
                log_msg += f", ConsistencyWarnings={self.norm_consistency_warnings}"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT TEST]"
            
            logger.info(log_msg)

    def _save_checkpoint(self):
        """Save model checkpoint with data statistics"""
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_similarity': self.best_eval_similarity,
            'best_loss': self.best_loss,
            'loss_history': list(self.loss_history),
            'similarity_history': list(self.similarity_history),
            'noise_scale_history': list(self.noise_scale_history),
            
            # NEW: Data consistency information
            'train_clip_norms': list(self.train_clip_norms),
            'train_eva_norms': list(self.train_eva_norms),
            'eval_clip_norms': list(self.eval_clip_norms),
            'eval_eva_norms': list(self.eval_eva_norms),
            'norm_consistency_warnings': self.norm_consistency_warnings,
            'current_noise_scale': self.loss_fn.get_noise_scale() if hasattr(self.loss_fn, 'get_noise_scale') else 1.0,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if self.use_wandb:
            wandb.log({
                "checkpoint/saved": True,
                "checkpoint/step": self.global_step,
                "checkpoint/path": str(checkpoint_path),
                "checkpoint/consistency_warnings": self.norm_consistency_warnings,
            }, step=self.global_step)

    def train(self) -> Dict[str, Any]:
        """Main training loop with data consistency validation"""
        logger.info("Starting CLIP reproduction training with data consistency validation...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Estimated training steps per epoch: {self.estimated_steps_per_epoch}")
        logger.info(f"  Total estimated training steps: {self.estimated_steps_per_epoch * self.num_epochs}")
        logger.info(f"  🎯 DATA CONSISTENCY VALIDATION: {self.validate_data_consistency}")
        logger.info(f"  📊 DATA STATISTICS LOGGING: {self.log_data_statistics}")
        
        if self.overfit_batch is not None:
            logger.info(f"  OVERFITTING TEST MODE: Using {self.overfit_batch['batch_size']} samples")
        
        # Log initial setup to WandB
        if self.use_wandb:
            wandb.log({
                "setup/total_parameters": sum(p.numel() for p in self.model.parameters()),
                "setup/estimated_steps_per_epoch": self.estimated_steps_per_epoch,
                "setup/total_estimated_steps": self.estimated_steps_per_epoch * self.num_epochs,
                "setup/training_started": True,
                "setup/data_consistency_validation": self.validate_data_consistency,
                "setup/statistics_logging": self.log_data_statistics,
            }, step=0)
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                # Log epoch start to WandB
                if self.use_wandb:
                    wandb.log({
                        "epoch/started": epoch + 1,
                        "epoch/total_epochs": self.num_epochs,
                    }, step=self.global_step)
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_start_time = time.time()
                
                try:
                    dataloader_iter = iter(self.train_dataloader)
                    batch_count = 0
                    
                    while True:
                        try:
                            batch = next(dataloader_iter)
                            batch_count += 1
                        except StopIteration:
                            logger.info(f"Epoch {epoch + 1} completed: {batch_count} batches processed")
                            break
                        
                        step_start_time = time.time()
                        
                        # Compute loss
                        try:
                            loss, metrics = self._compute_loss(batch)
                        except Exception as e:
                            logger.error(f"Error computing loss at step {self.global_step}: {e}")
                            continue
                        
                        # Backward pass
                        try:
                            grad_norm = self._backward_and_step(loss)
                        except Exception as e:
                            logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                            continue
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        epoch_steps += 1
                        self.global_step += 1
                        
                        # Log metrics
                        self._log_metrics(loss.item(), metrics or {}, grad_norm)
                        
                        # Run evaluation
                        if self.global_step % self.eval_every_n_steps == 0:
                            logger.info(f"Running evaluation at step {self.global_step}...")
                            eval_metrics = self._evaluate()
                            
                            if eval_metrics:
                                logger.info(f"Evaluation results:")
                                logger.info(f"  CLIP similarity: {eval_metrics.get('eval_clip_similarity', 0):.4f}")
                                logger.info(f"  Generated norm: {eval_metrics.get('eval_generated_norm_mean', 0):.3f}")
                                logger.info(f"  Target norm: {eval_metrics.get('eval_target_norm_mean', 0):.3f}")
                                logger.info(f"  Norm ratio: {eval_metrics.get('eval_norm_ratio', 0):.3f}")
                                
                                # NEW: Log data consistency info
                                if 'train_eval_clip_diff' in eval_metrics:
                                    diff = eval_metrics['train_eval_clip_diff']
                                    consistency = eval_metrics.get('data_consistency_good', False)
                                    logger.info(f"  🎯 Data consistency: {'✅ Good' if consistency else '⚠️ Poor'} (diff: {diff:.2f})")
                                
                                # Log evaluation metrics to WandB
                                if self.use_wandb:
                                    wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                                    wandb.log(wandb_eval_metrics, step=self.global_step)
                                
                                # Update best eval similarity
                                if eval_metrics.get('eval_clip_similarity', 0) > self.best_eval_similarity:
                                    self.best_eval_similarity = eval_metrics['eval_clip_similarity']
                                    logger.info(f"🎉 New best CLIP similarity: {self.best_eval_similarity:.4f}")
                                    
                                    if self.use_wandb:
                                        wandb.log({
                                            "eval/new_best_similarity": self.best_eval_similarity,
                                            "eval/best_similarity_step": self.global_step,
                                        }, step=self.global_step)
                        
                        # Save checkpoint
                        if self.global_step % self.save_every_n_steps == 0:
                            self._save_checkpoint()
                        
                        # # Check for early success in overfitting test
                        # if (self.overfit_batch is not None and 
                        #     metrics and 
                        #     metrics.get('velocity_similarity', 0) > 0.9):
                        #     logger.info("🎉 OVERFITTING TEST PASSED! Model can learn effectively.")
                        #     if self.use_wandb:
                        #         wandb.log({
                        #             "overfit_test/passed": True,
                        #             "overfit_test/final_similarity": metrics['velocity_similarity'],
                        #             "overfit_test/steps_to_pass": self.global_step,
                        #         }, step=self.global_step)
                        #     break
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {e}")
                    continue
                
                # End of epoch logging
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                
                logger.info(f"Epoch {epoch + 1} completed:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Steps in epoch: {epoch_steps}")
                logger.info(f"  Epoch time: {epoch_time:.1f}s")
                logger.info(f"  Consistency warnings: {self.norm_consistency_warnings}")
                
                # Log epoch summary to WandB
                if self.use_wandb:
                    wandb_epoch_metrics = {
                        "epoch/completed": epoch + 1,
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/steps": epoch_steps,
                        "epoch/time_seconds": epoch_time,
                        "epoch/consistency_warnings": self.norm_consistency_warnings,
                    }
                    wandb.log(wandb_epoch_metrics, step=self.global_step)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.use_wandb:
                wandb.log({"training/interrupted": True}, step=self.global_step)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            if self.use_wandb:
                wandb.log({"training/failed": True, "training/error": str(e)}, step=self.global_step)
            raise
        
        finally:
            # Final checkpoint
            self._save_checkpoint()
            
            # Final evaluation
            logger.info("Running final evaluation...")
            final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'training_completed': True,
                'total_time_seconds': total_time,
                'total_steps': self.global_step,
                'final_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'best_eval_similarity': self.best_eval_similarity,
                'final_eval': final_eval,
                'overfit_test': self.overfit_batch is not None,
                'overfit_success': (self.overfit_batch is not None and 
                                  len(self.similarity_history) > 0 and 
                                  max(self.similarity_history) > 0.8),
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'lr_history': list(self.lr_history),
                'grad_norm_history': list(self.grad_norm_history),
                'noise_scale_history': list(self.noise_scale_history),
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                'final_noise_scale': self.noise_scale_history[-1] if self.noise_scale_history else 1.0,
                'wandb_enabled': self.use_wandb,
                
                # NEW: Data consistency summary
                'data_consistency_enabled': self.validate_data_consistency,
                'norm_consistency_warnings': self.norm_consistency_warnings,
                'train_clip_norms': list(self.train_clip_norms),
                'eval_clip_norms': list(self.eval_clip_norms),
                'data_consistency_good': self.norm_consistency_warnings == 0,
            }
            
            # Log final summary to WandB
            if self.use_wandb:
                final_wandb_metrics = {
                    "final/training_completed": True,
                    "final/total_time_seconds": total_time,
                    "final/total_steps": self.global_step,
                    "final/best_loss": self.best_loss,
                    "final/best_eval_similarity": self.best_eval_similarity,
                    "final/consistency_warnings": self.norm_consistency_warnings,
                    "final/data_consistency_good": summary['data_consistency_good'],
                }
                
                if final_eval:
                    for key, value in final_eval.items():
                        final_wandb_metrics[f"final/{key}"] = value
                
                wandb.log(final_wandb_metrics, step=self.global_step)
                wandb.finish()
            
            # Save training summary
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Training completed!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best CLIP similarity: {self.best_eval_similarity:.4f}")
            logger.info(f"  🎯 Data consistency warnings: {self.norm_consistency_warnings}")
            
            if self.norm_consistency_warnings == 0:
                logger.info(f"  ✅ DATA CONSISTENCY: No warnings - training/eval data is consistent!")
            else:
                logger.warning(f"  ⚠️ DATA CONSISTENCY: {self.norm_consistency_warnings} warnings - check data preprocessing!")
            
            if final_eval:
                logger.info(f"  Final evaluation:")
                logger.info(f"    CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
                logger.info(f"    Generated norm: {final_eval.get('eval_generated_norm_mean', 0):.3f}")
                logger.info(f"    Target norm: {final_eval.get('eval_target_norm_mean', 0):.3f}")
                logger.info(f"    Norm ratio: {final_eval.get('eval_norm_ratio', 0):.3f}")
            
            return summary


def create_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    overfit_test_size: Optional[int] = None,
    debug_mode: bool = False,
    # Data consistency parameters
    validate_data_consistency: bool = True,
    log_data_statistics: bool = True,
    norm_tolerance: float = 5.0,
    # Noise scaling parameters
    sync_noise_scale_every: int = 10,
    enable_generation_debug: bool = False,
    # WandB parameters
    use_wandb: bool = True,
    wandb_project: str = "blip3o-clip-reproduction",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    **kwargs
) -> BLIP3oCLIPTrainer:
    """Factory function to create CLIP trainer with data consistency validation"""
    
    return BLIP3oCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        overfit_test_size=overfit_test_size,
        debug_mode=debug_mode,
        validate_data_consistency=validate_data_consistency,
        log_data_statistics=log_data_statistics,
        norm_tolerance=norm_tolerance,
        sync_noise_scale_every=sync_noise_scale_every,
        enable_generation_debug=enable_generation_debug,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
        **kwargs
    )