"""
Enhanced BLIP3-o Trainer with Better Multi-GPU Support
File: src/modules/trainers/blip3o_trainer_enhanced.py

ENHANCED FEATURES:
1. Robust GPU detection and error handling
2. Better memory management for multi-GPU
3. Enhanced error reporting and recovery
4. Automatic fallback mechanisms
5. Improved debugging capabilities
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import numpy as np
from collections import defaultdict
import json
import time
from pathlib import Path
import traceback
import gc

logger = logging.getLogger(__name__)

class EnhancedBLIP3oTrainer(Trainer):
    """
    Enhanced BLIP3-o Trainer with better multi-GPU support and error handling
    """
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        flow_matching_loss,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        
        self.flow_matching_loss = flow_matching_loss
        self.training_step_count = 0
        
        # Enhanced tracking
        self.loss_components = defaultdict(list)
        self.memory_usage = []
        self.training_metrics = []
        self.error_log = []
        
        # Multi-GPU settings
        self.is_distributed = dist.is_initialized()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Enhanced debugging
        self.debug_mode = getattr(args, 'debug', False)
        
        if self.is_main_process:
            logger.info("✅ Enhanced BLIP3-o trainer initialized")
            if self.is_distributed:
                logger.info(f"Distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
            logger.info(f"Debug mode: {self.debug_mode}")
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            
            self.memory_usage.append({
                'stage': stage,
                'step': self.training_step_count,
                'allocated_gb': memory_allocated,
                'cached_gb': memory_cached
            })
            
            if self.debug_mode and self.is_main_process:
                logger.debug(f"{stage}: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
    
    def _handle_oom_error(self, error: Exception, inputs: Dict[str, Any]):
        """Handle out of memory errors with recovery attempts"""
        self.error_log.append({
            'type': 'OOM',
            'step': self.training_step_count,
            'error': str(error),
            'batch_size': inputs['eva_embeddings'].shape[0] if 'eva_embeddings' in inputs else 'unknown'
        })
        
        if self.is_main_process:
            logger.warning(f"OOM error at step {self.training_step_count}: {error}")
        
        # Recovery attempts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Reduce batch size for this step
        if isinstance(inputs, dict):
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and len(inputs[key]) > 1:
                    inputs[key] = inputs[key][:len(inputs[key])//2]  # Half the batch
        
        return inputs
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Enhanced compute_loss with better error handling and memory management
        """
        self._log_memory_usage("compute_loss_start")
        
        try:
            # Validate inputs
            required_keys = ['eva_embeddings', 'clip_embeddings']
            for key in required_keys:
                if key not in inputs:
                    raise ValueError(f"Missing required input: {key}")
            
            eva_embeddings = inputs['eva_embeddings']
            clip_embeddings = inputs['clip_embeddings']
            
            # Input validation
            if eva_embeddings.dim() != 3 or clip_embeddings.dim() != 3:
                raise ValueError(f"Invalid input dimensions: EVA {eva_embeddings.shape}, CLIP {clip_embeddings.shape}")
            
            batch_size = eva_embeddings.shape[0]
            device = eva_embeddings.device
            
            self._log_memory_usage("inputs_loaded")
            
            # Sample timesteps with error handling
            try:
                timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
            except Exception as e:
                logger.warning(f"Flow matching timestep sampling failed: {e}")
                timesteps = torch.rand(batch_size, device=device)
            
            # Sample noise for flow matching
            noise = torch.randn_like(clip_embeddings)
            
            # Create noisy samples
            try:
                x_0 = torch.randn_like(clip_embeddings)
                noisy_clip = self.flow_matching_loss.interpolate_data(
                    x_0=x_0, x_1=clip_embeddings, t=timesteps, noise=noise
                )
            except Exception as e:
                logger.warning(f"Flow matching interpolation failed: {e}")
                # Simple fallback interpolation
                alpha = timesteps.view(-1, 1, 1)
                x_0 = torch.randn_like(clip_embeddings)
                noisy_clip = (1 - alpha) * x_0 + alpha * clip_embeddings + 0.1 * noise
            
            self._log_memory_usage("flow_matching_setup")
            
            # Forward pass with error handling
            try:
                model_output = model(
                    hidden_states=noisy_clip,
                    timestep=timesteps,
                    encoder_hidden_states=eva_embeddings,
                    return_dict=False
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    inputs = self._handle_oom_error(e, inputs)
                    # Retry with reduced inputs
                    eva_embeddings = inputs['eva_embeddings']
                    clip_embeddings = inputs['clip_embeddings']
                    batch_size = eva_embeddings.shape[0]
                    
                    # Redo preprocessing with smaller batch
                    timesteps = timesteps[:batch_size]
                    noise = noise[:batch_size]
                    noisy_clip = noisy_clip[:batch_size]
                    
                    model_output = model(
                        hidden_states=noisy_clip,
                        timestep=timesteps,
                        encoder_hidden_states=eva_embeddings,
                        return_dict=False
                    )
                else:
                    raise e
            
            self._log_memory_usage("model_forward")
            
            # Compute loss with error handling
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
                logger.warning(f"Flow matching loss computation failed: {e}")
                # Fallback to simple MSE loss
                loss = F.mse_loss(model_output, clip_embeddings)
                metrics = {'fallback_mse_loss': loss.item(), 'loss_computation_failed': True}
            
            self._log_memory_usage("loss_computed")
            
            # Store metrics (only on main process)
            if metrics and self.is_main_process:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.loss_components[key].append(value)
                
                # Store full metrics
                metrics['step'] = self.training_step_count
                metrics['timestamp'] = time.time()
                self.training_metrics.append(metrics)
            
            # Periodic logging (only on main process)
            if self.is_main_process and self.training_step_count % self.args.logging_steps == 0:
                self._log_training_progress(metrics, timesteps, model_output, clip_embeddings)
            
            self.training_step_count += 1
            
            # Memory cleanup
            del noise, x_0, noisy_clip
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._log_memory_usage("compute_loss_end")
            
            # Prepare outputs
            outputs = {
                'model_output': model_output,
                'timesteps': timesteps,
                'metrics': metrics,
                'eva_embeddings': eva_embeddings,
                'clip_embeddings': clip_embeddings,
            } if return_outputs else None
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            # Enhanced error logging
            error_info = {
                'step': self.training_step_count,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'input_shapes': {k: v.shape if isinstance(v, torch.Tensor) else type(v) 
                               for k, v in inputs.items()},
                'memory_usage': self.memory_usage[-1] if self.memory_usage else None
            }
            self.error_log.append(error_info)
            
            if self.is_main_process:
                logger.error(f"Enhanced compute_loss failed at step {self.training_step_count}: {e}")
                if self.debug_mode:
                    logger.error(traceback.format_exc())
            
            # Emergency fallback
            try:
                eva_embeddings = inputs['eva_embeddings']
                clip_embeddings = inputs['clip_embeddings']
                
                # Simple forward pass without flow matching
                model_output = model(
                    hidden_states=clip_embeddings,
                    timestep=torch.zeros(eva_embeddings.shape[0], device=eva_embeddings.device),
                    encoder_hidden_states=eva_embeddings,
                    return_dict=False
                )
                
                loss = F.mse_loss(model_output, clip_embeddings)
                
                if self.is_main_process:
                    logger.warning("Using emergency fallback loss computation")
                
                outputs = {'emergency_fallback': True, 'original_error': str(e)} if return_outputs else None
                return (loss, outputs) if return_outputs else loss
                
            except Exception as fallback_error:
                if self.is_main_process:
                    logger.error(f"Emergency fallback also failed: {fallback_error}")
                raise e
    
    def _log_training_progress(
        self,
        metrics: Optional[Dict[str, float]],
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
        target_clip: torch.Tensor
    ):
        """Enhanced training progress logging"""
        
        if not self.is_main_process or metrics is None:
            return
        
        # Calculate additional metrics
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(
                F.normalize(model_output.mean(dim=1), dim=-1),
                F.normalize(target_clip.mean(dim=1), dim=-1),
                dim=-1
            ).mean().item()
            
            output_norm = torch.norm(model_output, dim=-1).mean().item()
            target_norm = torch.norm(target_clip, dim=-1).mean().item()
        
        # Create progress message
        loss_value = metrics.get('flow_loss', metrics.get('total_loss', metrics.get('fallback_mse_loss', 0)))
        
        progress_msg = (
            f"Step {self.training_step_count}: "
            f"Loss={loss_value:.4f}, "
            f"Cosine={cosine_sim:.4f}, "
            f"OutNorm={output_norm:.3f}, "
            f"TgtNorm={target_norm:.3f}"
        )
        
        # Add memory info if available
        if self.memory_usage:
            latest_memory = self.memory_usage[-1]
            progress_msg += f", Mem={latest_memory['allocated_gb']:.1f}GB"
        
        # Add error info if any
        if self.error_log:
            recent_errors = [e for e in self.error_log if e['step'] > self.training_step_count - self.args.logging_steps]
            if recent_errors:
                progress_msg += f", Errors={len(recent_errors)}"
        
        logger.info(progress_msg)
        
        # Detailed debug logging
        if self.debug_mode:
            debug_info = {
                'timestep_stats': {
                    'mean': timesteps.mean().item(),
                    'std': timesteps.std().item(),
                    'min': timesteps.min().item(),
                    'max': timesteps.max().item()
                },
                'output_stats': {
                    'mean': model_output.mean().item(),
                    'std': model_output.std().item(),
                    'norm_mean': output_norm
                },
                'target_stats': {
                    'mean': target_clip.mean().item(),
                    'std': target_clip.std().item(),
                    'norm_mean': target_norm
                }
            }
            logger.debug(f"Debug info: {debug_info}")
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Enhanced evaluation with better memory management"""
        
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            if self.is_main_process:
                logger.warning("No evaluation dataset provided")
            return {}
        
        if self.is_main_process:
            logger.info("Starting enhanced evaluation...")
        
        # Memory cleanup before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._log_memory_usage("eval_start")
        
        # Set model to evaluation mode
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Enhanced evaluation settings
        max_eval_batches = 15 if not self.is_distributed else 10
        eval_losses = []
        all_metrics = defaultdict(list)
        eval_errors = []
        
        batch_count = 0
        successful_batches = 0
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                if batch_count >= max_eval_batches:
                    break
                
                try:
                    # Memory check
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        if memory_used > 20:  # Conservative limit for eval
                            if self.is_main_process:
                                logger.warning(f"Stopping eval due to memory: {memory_used:.1f}GB")
                            break
                    
                    # Prepare inputs
                    inputs = self._prepare_inputs(inputs)
                    
                    # Reduce batch size for stability
                    if isinstance(inputs, dict):
                        for key in inputs:
                            if isinstance(inputs[key], torch.Tensor) and len(inputs[key]) > 3:
                                inputs[key] = inputs[key][:3]  # Max 3 samples for eval
                    
                    # Compute loss
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    
                    eval_losses.append(loss.item())
                    successful_batches += 1
                    
                    # Collect detailed metrics
                    if outputs and outputs.get('metrics'):
                        for key, value in outputs['metrics'].items():
                            if isinstance(value, (int, float)):
                                all_metrics[key].append(value)
                    
                    # Cleanup
                    del inputs, loss, outputs
                    
                except Exception as e:
                    eval_errors.append({
                        'step': step,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    
                    if self.is_main_process:
                        logger.warning(f"Eval step {step} failed: {e}")
                    
                    # OOM recovery
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        break
                
                batch_count += 1
                
                # Periodic cleanup
                if batch_count % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self._log_memory_usage("eval_end")
        
        # Aggregate results (only on main process)
        eval_results = {}
        if self.is_main_process:
            if eval_losses:
                eval_results = {
                    f'{metric_key_prefix}_loss': np.mean(eval_losses),
                    f'{metric_key_prefix}_loss_std': np.std(eval_losses),
                    f'{metric_key_prefix}_successful_batches': successful_batches,
                    f'{metric_key_prefix}_total_batches': batch_count,
                    f'{metric_key_prefix}_error_rate': len(eval_errors) / max(batch_count, 1),
                }
                
                # Aggregate detailed metrics
                for key, values in all_metrics.items():
                    if values:
                        eval_results[f'{metric_key_prefix}_{key}'] = np.mean(values)
                        eval_results[f'{metric_key_prefix}_{key}_std'] = np.std(values)
                
                logger.info(f"Evaluation completed: {successful_batches}/{batch_count} successful batches")
                logger.info(f"Average eval loss: {eval_results[f'{metric_key_prefix}_loss']:.4f}")
                
                if eval_errors:
                    logger.warning(f"Evaluation errors: {len(eval_errors)}")
                    # Log first few errors for debugging
                    for error in eval_errors[:3]:
                        logger.warning(f"  {error['error_type']}: {error['error']}")
            else:
                eval_results = {f'{metric_key_prefix}_loss': float('inf')}
                logger.warning("No successful evaluation batches")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return eval_results
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Enhanced model saving with debugging info"""
        
        # Only save on main process
        if not self.is_main_process:
            return
        
        output_dir = output_dir or self.args.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model using parent class
            super().save_model(output_dir, _internal_call)
            
            # Save enhanced debugging information
            self._save_enhanced_debug_info(output_path)
            
            logger.info(f"✅ Enhanced model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Enhanced model saving failed: {e}")
            logger.error(traceback.format_exc())
            
            # Try basic save as fallback
            try:
                torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
                logger.info("Fallback model save completed")
            except Exception as fallback_e:
                logger.error(f"Fallback save also failed: {fallback_e}")
                raise e
    
    def _save_enhanced_debug_info(self, output_path: Path):
        """Save comprehensive debugging information"""
        
        # Training summary
        summary = {
            'training_completed': True,
            'total_steps': self.training_step_count,
            'total_errors': len(self.error_log),
            'final_step': self.training_step_count,
            'distributed_training': self.is_distributed,
            'world_size': dist.get_world_size() if self.is_distributed else 1,
            'timestamp': time.time()
        }
        
        # Latest metrics
        if self.training_metrics:
            latest_metrics = self.training_metrics[-1]
            summary.update({
                'final_loss': latest_metrics.get('total_loss', latest_metrics.get('flow_loss')),
                'final_metrics': latest_metrics
            })
        
        with open(output_path / 'enhanced_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save recent training metrics
        if self.training_metrics:
            with open(output_path / 'training_metrics_history.json', 'w') as f:
                json.dump(self.training_metrics[-100:], f, indent=2)  # Last 100 steps
        
        # Save memory usage
        if self.memory_usage:
            with open(output_path / 'memory_usage.json', 'w') as f:
                json.dump(self.memory_usage, f, indent=2)
        
        # Save error log
        if self.error_log:
            with open(output_path / 'error_log.json', 'w') as f:
                json.dump(self.error_log, f, indent=2)
        
        # Loss components summary
        if self.loss_components:
            loss_summary = {}
            for key, values in self.loss_components.items():
                if values:
                    loss_summary[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            with open(output_path / 'loss_components_summary.json', 'w') as f:
                json.dump(loss_summary, f, indent=2)
        
        # System info
        system_info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'distributed_available': dist.is_available(),
            'distributed_initialized': dist.is_initialized(),
        }
        
        if torch.cuda.is_available():
            system_info['gpu_info'] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                system_info['gpu_info'].append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        
        with open(output_path / 'system_info.json', 'w') as f:
            json.dump(system_info, f, indent=2)
        
        logger.info("Enhanced debugging information saved")


def create_enhanced_training_args(
    output_dir: str,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 20,
    save_steps: int = 500,
    eval_steps: int = 250,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    dataloader_num_workers: int = 4,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    debug: bool = False,
    **kwargs
) -> TrainingArguments:
    # Remove ddp_find_unused_parameters from kwargs if present
    ddp_find_unused_parameters = kwargs.pop('ddp_find_unused_parameters', False)
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if eval_steps > 0 else "no",
        eval_steps=eval_steps if eval_steps > 0 else None,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=3,
        prediction_loss_only=False,
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
        
        # Enhanced multi-GPU settings
        ddp_find_unused_parameters=ddp_find_unused_parameters,  # Use the extracted value
        # save_on_each_node=False,
        dataloader_persistent_workers=True,
        
        # Enhanced error handling
        ignore_data_skip=True,
        
        # Debug mode
        debug=debug,
        
        **kwargs
    )


# Legacy compatibility
BLIP3oTrainer = EnhancedBLIP3oTrainer
create_blip3o_training_args = create_enhanced_training_args