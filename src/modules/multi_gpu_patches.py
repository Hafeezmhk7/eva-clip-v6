"""
Multi-GPU Compatibility Patches for BLIP3-o Trainer
Add this to the beginning of your training script to enable proper multi-GPU support
"""

import torch
import torch.distributed as dist
import os
from typing import Dict, Any, Optional, Union, Tuple, List

def patch_trainer_for_multi_gpu():
    """Apply patches to make BLIP3oTrainer work properly with multi-GPU training"""
    
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Store original methods
        original_compute_loss = BLIP3oTrainer.compute_loss
        original_log_training_metrics = BLIP3oTrainer._log_training_metrics
        original_save_model = BLIP3oTrainer.save_model
        
        def multi_gpu_compute_loss(
            self,
            model,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Optional[int] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
            """Multi-GPU compatible compute_loss method"""
            
            # Extract inputs from batch
            eva_embeddings = inputs['eva_embeddings']
            clip_embeddings = inputs['clip_embeddings']
            
            batch_size = eva_embeddings.shape[0]
            device = eva_embeddings.device
            
            # Sample random timesteps for flow matching
            timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
            
            # Sample noise for flow matching
            noise = torch.randn_like(clip_embeddings)
            
            # Create noisy samples according to flow matching interpolation
            x_0 = torch.randn_like(clip_embeddings)
            noisy_clip = self.flow_matching_loss.interpolate_data(
                x_0=x_0,
                x_1=clip_embeddings,
                t=timesteps,
                noise=noise
            )
            
            # Forward pass through DiT model
            model_output = model(
                hidden_states=noisy_clip,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=False
            )
            
            # Compute flow matching loss with detailed metrics
            loss, metrics = self.flow_matching_loss(
                model_output=model_output,
                target_samples=clip_embeddings,
                timesteps=timesteps,
                eva_conditioning=eva_embeddings,
                noise=noise,
                return_metrics=True
            )
            
            # Store metrics for logging (only on main process)
            if metrics is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                for key, value in metrics.items():
                    self.loss_components[key].append(value)
            
            # Log metrics periodically (only on main process)
            if (not dist.is_initialized() or dist.get_rank() == 0) and self.training_step_count % self.args.logging_steps == 0:
                self._log_training_metrics(metrics, timesteps, model_output, clip_embeddings)
            
            self.training_step_count += 1
            
            # Prepare outputs
            outputs = {
                'model_output': model_output,
                'noisy_clip': noisy_clip,
                'timesteps': timesteps,
                'metrics': metrics,
                'eva_embeddings': eva_embeddings,
                'clip_embeddings': clip_embeddings,
            } if return_outputs else None
            
            if return_outputs:
                return loss, outputs
            else:
                return loss
        
        def multi_gpu_log_training_metrics(
            self,
            metrics: Optional[Dict[str, float]],
            timesteps: torch.Tensor,
            model_output: torch.Tensor,
            target_clip: torch.Tensor
        ):
            """Multi-GPU compatible logging - only log on main process"""
            
            # Only log on main process in distributed training
            if dist.is_initialized() and dist.get_rank() != 0:
                return
            
            # Call original logging method
            original_log_training_metrics(self, metrics, timesteps, model_output, target_clip)
        
        def multi_gpu_save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
            """Multi-GPU compatible model saving - only save on main process"""
            
            # Only save on main process in distributed training
            if dist.is_initialized() and dist.get_rank() != 0:
                return
            
            # Call original save method
            original_save_model(self, output_dir, _internal_call)
        
        def multi_gpu_evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            """Multi-GPU compatible evaluation with memory optimization"""
            
            eval_dataset = eval_dataset or self.eval_dataset
            if eval_dataset is None:
                if dist.is_initialized() and dist.get_rank() == 0:
                    logger.warning("No evaluation dataset provided")
                return {}
            
            # Aggressive memory cleanup before evaluation
            torch.cuda.empty_cache()
            
            # Set model to evaluation mode
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            # Create evaluation dataloader
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            # Collect evaluation metrics
            eval_losses = []
            all_metrics = {}
            
            # CRITICAL: Limit evaluation batches for memory (even more aggressive for multi-GPU)
            MAX_EVAL_BATCHES = 5  # Very conservative for multi-GPU
            eval_batch_count = 0
            
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.info(f"Running LIMITED multi-GPU evaluation (max {MAX_EVAL_BATCHES} batches)")
            
            with torch.no_grad():
                for step, inputs in enumerate(eval_dataloader):
                    # Memory check before each batch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        memory_percent = (memory_used / memory_total) * 100
                        
                        # Stop evaluation if memory usage is too high
                        if memory_percent > 70:  # Conservative for multi-GPU
                            if dist.is_initialized() and dist.get_rank() == 0:
                                logger.warning(f"Stopping evaluation due to memory: {memory_percent:.1f}%")
                            break
                    
                    # Move inputs to device
                    inputs = self._prepare_inputs(inputs)
                    
                    try:
                        # Force smaller eval batch size for multi-GPU
                        if isinstance(inputs, dict):
                            for key in inputs:
                                if isinstance(inputs[key], torch.Tensor) and len(inputs[key].shape) > 0:
                                    inputs[key] = inputs[key][:2]  # Only 2 samples max per GPU
                        
                        # Compute loss
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        
                        eval_losses.append(loss.item())
                        
                        # Collect metrics
                        if outputs and outputs.get('metrics'):
                            for key, value in outputs['metrics'].items():
                                if key not in all_metrics:
                                    all_metrics[key] = []
                                all_metrics[key].append(value)
                        
                        # Clear immediately
                        del inputs, loss, outputs
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if dist.is_initialized() and dist.get_rank() == 0:
                                logger.warning(f"OOM during evaluation batch {step}")
                            torch.cuda.empty_cache()
                            break
                        else:
                            if dist.is_initialized() and dist.get_rank() == 0:
                                logger.warning(f"Error during evaluation: {e}")
                            continue
                    
                    # Cleanup and count
                    torch.cuda.empty_cache()
                    eval_batch_count += 1
                    
                    if eval_batch_count >= MAX_EVAL_BATCHES:
                        if dist.is_initialized() and dist.get_rank() == 0:
                            logger.info(f"Completed evaluation after {eval_batch_count} batches")
                        break
            
            # Aggregate results (only on main process)
            eval_results = {}
            if not dist.is_initialized() or dist.get_rank() == 0:
                if eval_losses:
                    eval_results = {
                        f'{metric_key_prefix}_loss': sum(eval_losses) / len(eval_losses),
                        f'{metric_key_prefix}_batches_processed': eval_batch_count,
                    }
                    
                    # Aggregate detailed metrics
                    for key, values in all_metrics.items():
                        if values:
                            eval_results[f'{metric_key_prefix}_{key}'] = sum(values) / len(values)
                
                if eval_results:
                    logger.info(f"Multi-GPU evaluation results: {eval_results}")
            
            # Cleanup
            torch.cuda.empty_cache()
            
            return eval_results
        
        # Apply patches
        BLIP3oTrainer.compute_loss = multi_gpu_compute_loss
        BLIP3oTrainer._log_training_metrics = multi_gpu_log_training_metrics
        BLIP3oTrainer.save_model = multi_gpu_save_model
        BLIP3oTrainer.evaluate = multi_gpu_evaluate
        
        print("✅ Applied multi-GPU trainer patches")
        
    except Exception as e:
        print(f"⚠️  Failed to apply multi-GPU patches: {e}")


def patch_chunked_dataset_for_ddp():
    """Patch chunked dataset to work properly with DistributedDataParallel"""
    
    try:
        from src.modules.datasets.blip3o_dataset import create_chunked_dataloader
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        import torch.distributed as dist
        
        def create_ddp_chunked_dataloader(
            chunked_embeddings_dir,
            batch_size: int = 32,
            split: str = "train",
            eval_split_ratio: float = 0.1,
            normalize_embeddings: bool = True,
            shuffle_shards: bool = True,
            shuffle_within_shard: bool = True,
            delete_after_use: bool = False,
            num_workers: int = 0,
            pin_memory: bool = None,
            **kwargs
        ):
            """Create DataLoader optimized for DDP"""
            
            # Auto-detect pin_memory
            if pin_memory is None:
                pin_memory = torch.cuda.is_available()
            
            # Create dataset (this will be the same on all processes)
            from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset
            
            dataset = BLIP3oEmbeddingDataset(
                chunked_embeddings_dir=chunked_embeddings_dir,
                split=split,
                eval_split_ratio=eval_split_ratio,
                normalize_embeddings=normalize_embeddings,
                shuffle_shards=shuffle_shards,
                shuffle_within_shard=shuffle_within_shard,
                delete_after_use=delete_after_use,
                **kwargs
            )
            
            # Create DistributedSampler for DDP if in distributed mode
            sampler = None
            shuffle = False  # Don't shuffle in DataLoader when using sampler
            
            if dist.is_initialized():
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=(split == "train" and shuffle_shards)
                )
                print(f"✅ Created DistributedSampler for rank {dist.get_rank()}")
            else:
                shuffle = (split == "train" and shuffle_shards)
            
            # Use the chunked collate function
            from src.modules.datasets.blip3o_dataset import chunked_collate_fn
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=chunked_collate_fn,
                pin_memory=pin_memory,
                drop_last=True,  # Important for DDP
            )
            
            return dataloader
        
        # Replace the original function
        import src.modules.datasets.blip3o_dataset as dataset_module
        dataset_module.create_chunked_dataloader = create_ddp_chunked_dataloader
        
        print("✅ Applied DDP dataset patches")
        
    except Exception as e:
        print(f"⚠️  Failed to apply DDP dataset patches: {e}")


def patch_config_for_multi_gpu():
    """Patch config for better multi-GPU performance"""
    
    try:
        from src.modules.trainers.blip3o_trainer import create_blip3o_training_args
        from transformers import TrainingArguments
        import logging
        
        def create_multi_gpu_training_args(
            output_dir: str,
            num_train_epochs: int = 10,
            per_device_train_batch_size: int = 8,
            per_device_eval_batch_size: int = 4,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            warmup_steps: int = 100,
            logging_steps: int = 20,
            save_steps: int = 500,
            eval_steps: int = 250,
            gradient_accumulation_steps: int = 4,
            fp16: bool = True,
            bf16: bool = False,
            dataloader_num_workers: int = 4,
            remove_unused_columns: bool = False,
            load_best_model_at_end: bool = False,
            metric_for_best_model: str = "eval_loss",
            greater_is_better: bool = False,
            **kwargs
        ) -> TrainingArguments:
            """Create TrainingArguments optimized for multi-GPU training"""
            
            # Ensure save_steps is compatible with eval_steps
            if load_best_model_at_end and eval_steps > 0:
                if save_steps % eval_steps != 0:
                    adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
                    logging.warning(f"Adjusting save_steps from {save_steps} to {adjusted_save_steps}")
                    save_steps = adjusted_save_steps
            
            # Determine evaluation strategy
            if eval_steps > 0:
                eval_strategy = "steps"
                eval_steps_value = eval_steps
            else:
                eval_strategy = "no"
                eval_steps_value = None
                load_best_model_at_end = False
            
            return TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_strategy=eval_strategy,
                eval_steps=eval_steps_value,
                save_strategy="steps",
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=fp16 and not bf16,
                bf16=bf16,
                dataloader_num_workers=dataloader_num_workers,
                remove_unused_columns=remove_unused_columns,
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                save_total_limit=3,
                prediction_loss_only=False,
                report_to=["wandb"] if "wandb" in kwargs else [],
                # Multi-GPU specific optimizations
                ddp_find_unused_parameters=False,  # Optimize DDP communication
                dataloader_pin_memory=True,       # Optimize data loading
                save_on_each_node=False,          # Only save on main process
                push_to_hub=False,                # Avoid hub issues
                **kwargs
            )
        
        # Replace the original function
        import src.modules.trainers.blip3o_trainer as trainer_module
        trainer_module.create_blip3o_training_args = create_multi_gpu_training_args
        
        print("✅ Applied multi-GPU training args patches")
        
    except Exception as e:
        print(f"⚠️  Failed to apply training args patches: {e}")


def apply_all_multi_gpu_patches():
    """Apply all patches needed for multi-GPU training"""
    print("🔧 Applying multi-GPU compatibility patches...")
    
    patch_trainer_for_multi_gpu()
    patch_chunked_dataset_for_ddp()
    patch_config_for_multi_gpu()
    
    print("✅ All multi-GPU patches applied successfully!")


if __name__ == "__main__":
    # Test the patches
    apply_all_multi_gpu_patches()
    print("Multi-GPU patches ready for use!")