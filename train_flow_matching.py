"""
Enhanced Training Script for EVA-CLIP to CLIP Flow Matching
Fixed autocast implementation for PyTorch 2.3+
"""

import torch
from torch.utils.data import DataLoader
from src.data_hand.embedding_dataset import EmbeddingDataset
from src.modules.flow_matching import FlowMatchingLoss
import argparse
import os
import time
import datetime
from tqdm import tqdm
import wandb
import json
import traceback

def setup_wandb(args):
    """Setup Weights & Biases with authentication and project configuration"""
    if not args.use_wandb:
        return None
    
    # Set wandb API key
    wandb_token = "0d9895af249ee18e4fa141e8a2350e0f4adb920f"
    os.environ["WANDB_API_KEY"] = wandb_token
    
    # Create experiment name with timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"alignment_of_eva_clip_{timestamp}"
    
    # Initialize wandb
    run = wandb.init(
        project="unified_3D",
        name=experiment_name,
        config=vars(args),
        tags=["flow_matching", "eva_clip", "lumina_dit", "clip_alignment", "blip3o_inspired"],
        notes="EVA-CLIP to CLIP alignment using flow matching with BLIP3-o inspired improvements"
    )
    
    print(f" Weights & Biases initialized:")
    print(f"    Project: unified_3D")
    print(f"    Experiment: {experiment_name}")
    print(f"    URL: {wandb.run.url}")
    
    return run

def load_model(args, input_dim, cond_dim, device):
    print(f" Initializing LuminaDiT for Embedding Translation...")
    
    # Set default num_kv_heads if not specified
    if args.num_kv_heads is None:
        for kv_heads in range(args.num_heads // 2, 0, -1):
            if args.num_heads % kv_heads == 0:
                args.num_kv_heads = kv_heads
                break
        if args.num_kv_heads is None:
            args.num_kv_heads = 1
    
    print(f"  Using num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}")
    
    # Check if we wants to force original model or if enhanced should be used
    if args.force_original:
        print(f"    Force Original mode: Using Original LuminaDiT")
        print(f"    Recommended for embedding translation tasks")
        use_enhanced = False
    else:
        # Auto-select based on dimension and task
        use_enhanced = args.dim % 6 == 0 and not args.embedding_task
        if use_enhanced:
            print(f"    Using Enhanced LuminaDiT (dim={args.dim} compatible with Time RoPE)")
            print(f"     Note: Enhanced model may not be optimal for embedding translation")
        else:
            if args.dim % 6 != 0:
                print(f"    Dimension {args.dim} not compatible with 3D RoPE")
            print(f"    Using Original LuminaDiT (better for embedding tasks)")
    
    if use_enhanced:
        # Try Enhanced LuminaDiT with improved architecture
        from src.modules.lumina_next import EnhancedLuminaDiT
        
        model = EnhancedLuminaDiT(
            input_dim=input_dim,
            cond_dim=cond_dim,
            dim=args.dim,
            depth=args.depth,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads
        ).to(device)
        
        model_type = "Enhanced"
        print(f"    Using Enhanced LuminaDiT with Time RoPE")
        print(f"    Grouped Query Attention: {args.num_heads}/{args.num_kv_heads} heads")
        print(f"    Sandwich normalization enabled")
        print(f"    Time RoPE for temporal modeling")
    
    if not use_enhanced:
        # Use Original LuminaDiT
        
        from src.modules.lumina_next import LuminaDiT
        model = LuminaDiT(
            input_dim=input_dim,
            cond_dim=cond_dim,
            dim=args.dim,
            depth=args.depth,
            num_heads=args.num_heads
        ).to(device)
        
        model_type = "Original"
        print(f"    Using Original LuminaDiT")
        print(f"    Optimized for embedding-to-embedding translation")
        print(f"    No spatial assumptions, clean semantic mapping")
    
    return model, model_type

def safe_save_model(state, path):
    """Robust model saving with error handling and retry"""
    try:
        torch.save(state, path)
        print(f" Saved model to {path}")
        return True
    except Exception as e:
        print(f" Error saving model: {e}")
        print(" Attempting minimal save...")
        
        try:
            # Save only essential components
            minimal_state = {
                'model_state_dict': state['model_state_dict'],
                'args': state['args'],
                'model_type': state['model_type'],
                'val_loss': state.get('val_loss', 0),
                'val_alignment': state.get('val_alignment', 0)
            }
            torch.save(minimal_state, path)
            print(f" Minimal model saved to {path}")
            return True
        except Exception as e2:
            print(f" Critical error saving model: {e2}")
            traceback.print_exc()
            return False

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize mixed precision scaler
    scaler = None
    if device.type == 'cuda':
        try:
            # New PyTorch API
            scaler = torch.amp.GradScaler(device_type='cuda')
        except (AttributeError, TypeError):
            try:
                # Fallback to old API
                scaler = torch.cuda.amp.GradScaler()
            except:
                pass
    
    print(f"  Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
    
    # Initialize autocast context
    autocast = None
    if device.type == 'cuda' and scaler:
        try:
            # New autocast API
            autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        except AttributeError:
            try:
                # Fallback to old API
                autocast = torch.cuda.amp.autocast
            except:
                pass
    
    # Initialize wandb for experiment tracking
    wandb_run = setup_wandb(args)
    
    # Load dataset
    print(f" Loading dataset from {args.embedding_path}")
    dataset = EmbeddingDataset(args.embedding_path)
    
    # Split dataset into train/val (90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")
    
    # Initialize model with smart selection
    model, model_type = load_model(
        args, 
        input_dim=dataset[0]['clip_embedding'].shape[0],
        cond_dim=dataset[0]['eva_embedding'].shape[0],
        device=device
    )
    
    # Print model info and log to wandb
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1e6
    
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Model size: {model_size_mb:.1f} MB")
    print(f"    Model type: {model_type}")
    
    # Enhanced loss function with stability improvements
    criterion = FlowMatchingLoss(
        loss_type=args.loss_type,
        stability_weight=args.stability_weight,
        embedding_task=True
    )
    
    # Optimizer with embedding-specific settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # Better for transformers
        eps=1e-8
    )
    
    # BLIP3-o inspired learning rate schedule
    if args.scheduler == 'cosine_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=15,
            eta_min=args.lr * 0.05
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            min_lr=args.lr * 0.01
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.lr * 0.1
        )
    
    # Log model info to wandb
    if args.use_wandb:
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/size_mb": model_size_mb,
            "model/depth": args.depth,
            "model/dim": args.dim,
            "model/num_heads": args.num_heads,
            "model/type": model_type,
            "model/num_kv_heads": getattr(args, 'num_kv_heads', None),
            "model/enhanced_features": model_type == "Enhanced",
            "data/train_samples": len(train_dataset),
            "data/val_samples": len(val_dataset),
            "training/loss_type": args.loss_type,
            "training/scheduler": args.scheduler,
            "training/mixed_precision": scaler is not None
        })
        
        # Watch model for gradient tracking
        wandb.watch(model, log="all", log_freq=100)
    
    print(f" Training Configuration:")
    print(f"    Learning rate: {args.lr}")
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Model: {model_type} LuminaDiT")
    print(f"    Loss type: {args.loss_type}")
    print(f"    Scheduler: {args.scheduler}")
    print(f"    W&B Logging: {'Enabled' if args.use_wandb else 'Disabled'}")
    
    # Training loop with BLIP3-o improvements
    best_val_loss = float('inf')
    best_alignment = -1.0
    global_step = 0
    training_start_time = time.time()
    stagnation_counter = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_alignment = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Training loop with enhanced monitoring
        for batch_idx, batch in enumerate(pbar):
            clip_emb = batch['clip_embedding'].to(device)
            eva_emb = batch['eva_embedding'].to(device)
            
            # Convert to model's dtype
            dtype = next(model.parameters()).dtype
            clip_emb = clip_emb.to(dtype)
            eva_emb = eva_emb.to(dtype)
            
            # Forward pass with mixed precision
            if autocast:
                with autocast:  # Correct usage without parentheses
                    loss, diagnostics = criterion(model, clip_emb, eva_emb)
            else:
                loss, diagnostics = criterion(model, clip_emb, eva_emb)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            if scaler:
                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError as e:
                    print(f"Mixed precision error: {e}")
                    # Fallback to regular training
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            epoch_alignment += diagnostics.get('pred_target_cosine', 0.0)
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'align': f"{diagnostics.get('pred_target_cosine', 0.0):.3f}",
                'lr': f"{current_lr:.2e}",
                'model': model_type[:3]
            })
            
            # Enhanced logging with flow matching diagnostics
            if args.use_wandb and global_step % 100 == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/avg_loss': epoch_loss/(batch_idx+1),
                    'train/lr': current_lr,
                    'train/epoch': epoch,
                    'train/step': global_step,
                    'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'train/samples_processed': global_step * args.batch_size,
                    'system/gpu_memory_mb': torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
                }
                
                # Add flow matching diagnostics
                for key, value in diagnostics.items():
                    log_dict[f'train/{key}'] = value
                    
                wandb.log(log_dict)
        
        # Update learning rate
        if args.scheduler == 'plateau':
            # For plateau scheduler, we'll update after validation
            pass
        else:
            scheduler.step()
        
        # Enhanced validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"\n🧪 Running validation...")
            model.eval()
            val_loss = 0.0
            val_alignment = 0.0
            val_batches = 0
            val_diagnostics_sum = {}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= 10:  # Limit validation to 10 batches for speed
                        break
                        
                    clip_emb = batch['clip_embedding'].to(device)
                    eva_emb = batch['eva_embedding'].to(device)
                    
                    # Convert to model's dtype
                    dtype = next(model.parameters()).dtype
                    clip_emb = clip_emb.to(dtype)
                    eva_emb = eva_emb.to(dtype)
                    
                    if autocast:
                        with autocast:  # Correct usage without parentheses
                            loss, diagnostics = criterion(model, clip_emb, eva_emb)
                    else:
                        loss, diagnostics = criterion(model, clip_emb, eva_emb)
                    
                    val_loss += loss.item()
                    val_alignment += diagnostics.get('pred_target_cosine', 0.0)
                    
                    # Accumulate diagnostics
                    for key, value in diagnostics.items():
                        val_diagnostics_sum[key] = val_diagnostics_sum.get(key, 0) + value
                    
                    val_batches += 1
            
            val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            val_alignment = val_alignment / val_batches if val_batches > 0 else 0.0
            val_diagnostics_avg = {k: v / val_batches for k, v in val_diagnostics_sum.items()}
            
            model.train()
            
            print(f"    Val loss: {val_loss:.4f}")
            print(f"    Val alignment: {val_alignment:.4f}")
            
            # Update plateau scheduler with validation loss
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            
            # Enhanced wandb logging for validation
            if args.use_wandb:
                # Calculate improvement metrics
                val_improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0.0
                epoch_time = time.time() - epoch_start_time
                
                log_dict = {
                    'val/loss': val_loss,
                    'val/epoch': epoch,
                    'val/best_loss': min(best_val_loss, val_loss),
                    'val/improvement_pct': val_improvement,
                    'val/alignment': val_alignment,
                    'val/best_alignment': max(best_alignment, val_alignment),
                    'system/epoch_time_minutes': epoch_time / 60,
                    'train/avg_loss_epoch': epoch_loss / len(train_dataloader),
                    'train/avg_alignment_epoch': epoch_alignment / len(train_dataloader)
                }
                
                # Add validation diagnostics
                for key, value in val_diagnostics_avg.items():
                    log_dict[f'val/{key}'] = value
                    
                wandb.log(log_dict)
            
            # Save best model based on alignment (more important for embeddings)
            if val_alignment > best_alignment:
                best_alignment = val_alignment
                stagnation_counter = 0
                print(f"    New best alignment: {best_alignment:.4f}")
                
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_alignment': val_alignment,
                    'model_type': model_type,
                    'args': vars(args)
                }
                
                # Use safe saving method
                save_path = os.path.join(args.save_dir, 'best_alignment_model.pt')
                if not safe_save_model(state, save_path):
                    print(" Could not save best alignment model!")
            
            # Also save best loss model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"    New best validation loss: {best_val_loss:.4f}")
                
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_alignment': val_alignment,
                    'model_type': model_type,
                    'args': vars(args)
                }
                
                save_path = os.path.join(args.save_dir, 'best_loss_model.pt')
                if not safe_save_model(state, save_path):
                    print(" Could not save best loss model!")
            else:
                stagnation_counter += 1
                if stagnation_counter >= args.early_stopping_patience:
                    print(f" Early stopping: No improvement for {stagnation_counter} validation cycles")
                    break
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            state = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss': epoch_loss / len(train_dataloader),
                'alignment': epoch_alignment / len(train_dataloader),
                'model_type': model_type,
                'args': vars(args)
            }
            
            ckpt_path = os.path.join(args.save_dir, f"lumina_epoch{epoch+1}.pt")
            if safe_save_model(state, ckpt_path):
                print(f" Saved checkpoint to {ckpt_path}")
    
    # Training completion
    total_training_time = time.time() - training_start_time
    
    print(f"\n Training completed!")
    print(f" Best validation loss: {best_val_loss:.4f}")
    print(f" Best alignment: {best_alignment:.4f}")
    print(f"⏱ Total training time: {total_training_time/3600:.2f} hours")
    print(f" Model type used: {model_type}")
    
    # Final wandb logging and cleanup
    if args.use_wandb:
        # Log final summary metrics
        wandb.log({
            "summary/best_val_loss": best_val_loss,
            "summary/best_alignment": best_alignment,
            "summary/total_training_time_hours": total_training_time / 3600,
            "summary/final_epoch": epoch + 1,
            "summary/total_steps": global_step,
            "summary/samples_seen": global_step * args.batch_size,
            "summary/model_type": model_type
        })
        
        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Best Validation Loss", f"{best_val_loss:.4f}"],
                ["Best Alignment", f"{best_alignment:.4f}"],
                ["Total Parameters", f"{total_params:,}"],
                ["Model Size (MB)", f"{model_size_mb:.1f}"],
                ["Model Type", model_type],
                ["Enhanced Features", "Yes" if model_type == "Enhanced" else "No"],
                ["Training Time (hours)", f"{total_training_time/3600:.2f}"],
                ["Final Learning Rate", f"{optimizer.param_groups[0]['lr']:.2e}"],
                ["Total Epochs", epoch + 1],
                ["Batch Size", args.batch_size],
                ["Architecture", f"{args.depth}L-{args.dim}D-{args.num_heads}H"],
                ["Loss Type", args.loss_type],
                ["Scheduler", args.scheduler]
            ]
        )
        wandb.log({"summary/training_summary": summary_table})
        
        print(f" Experiment logged to: {wandb.run.url}")
        wandb.finish()
        print(f" Weights & Biases session closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVA-CLIP to CLIP Flow Matching Training")
    
    # Data arguments
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to precomputed embeddings pickle')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Model architecture
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--dim', type=int, default=1024,
                        help='Transformer hidden dimension')
    parser.add_argument('--depth', type=int, default=24,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--num_kv_heads', type=int, default=None,
                        help='Number of KV heads for grouped query attention (enhanced model only)')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (lower for embeddings)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Enhanced training options
    parser.add_argument('--force_original', action='store_true',
                        help='Force original LuminaDiT (recommended for embedding translation)')
    parser.add_argument('--embedding_task', action='store_true', default=True,
                        help='Flag indicating this is an embedding translation task')
    parser.add_argument('--loss_type', type=str, default='huber',
                        choices=['mse', 'l1', 'huber'],
                        help='Loss function type (huber recommended for stability)')
    parser.add_argument('--stability_weight', type=float, default=0.1,
                        help='Weight for stability loss component')
    parser.add_argument('--scheduler', type=str, default='cosine_restarts',
                        choices=['cosine', 'cosine_restarts', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience (validation cycles)')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging (enabled by default)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Handle wandb flags
    if args.no_wandb:
        args.use_wandb = False
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save args to file
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f" Starting EVA-CLIP to CLIP Flow Matching Training")
    print(f"=" * 70)
    print(f" Save directory: {args.save_dir}")
    print(f" W&B Logging: {'Enabled' if args.use_wandb else 'Disabled'}")
    print(f" Force Original: {'Yes' if args.force_original else 'Auto-select'}")
    print(f" Loss Type: {args.loss_type}")
    print(f" Scheduler: {args.scheduler}")
    if args.use_wandb:
        print(f"    Project: unified_3D")
        print(f"    Experiment: alignment_of_eva_clip")
    print(f"=" * 70)
    
    main(args)