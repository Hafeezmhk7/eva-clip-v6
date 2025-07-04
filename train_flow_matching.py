#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
from src.data_hand.embedding_dataset import EmbeddingDataset
from src.modules.lumina_next import LuminaDiT
from src.modules.flow_matching import FlowMatchingLoss
import argparse
import os
from tqdm import tqdm

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = EmbeddingDataset(args.embedding_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(4, os.cpu_count())
    )
    
    # Initialize model
    model = LuminaDiT(
        input_dim=dataset[0]['clip_embedding'].shape[0],
        cond_dim=dataset[0]['eva_embedding'].shape[0],
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    # Loss function
    criterion = FlowMatchingLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        # In training loop:
        for batch in pbar:
            clip_emb = batch['clip_embedding'].to(device)
            eva_emb = batch['eva_embedding'].to(device)
            
            # Convert to model's dtype
            dtype = next(model.parameters()).dtype
            clip_emb = clip_emb.to(dtype)
            eva_emb = eva_emb.to(dtype)
            
            loss = criterion(model, clip_emb, eva_emb)
    
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), avg_loss=epoch_loss/(pbar.n+1))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"lumina_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to precomputed embeddings pickle')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dim', type=int, default=1024,
                        help='Transformer hidden dimension')
    parser.add_argument('--depth', type=int, default=24,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)