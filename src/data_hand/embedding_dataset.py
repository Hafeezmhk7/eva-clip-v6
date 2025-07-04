import torch
from torch.utils.data import Dataset
import pickle

class EmbeddingDataset(Dataset):
    """Loads precomputed EVA/CLIP embeddings"""
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.eva_emb = data['eva_embeddings']
        self.clip_emb = data['clip_embeddings']
        
    def __len__(self):
        return len(self.eva_emb)
    
    def __getitem__(self, idx):
        return {
            'eva_embedding': self.eva_emb[idx],
            'clip_embedding': self.clip_emb[idx]
        }