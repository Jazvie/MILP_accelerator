from __future__ import annotations
import argparse
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gnn_model import BipartiteGNN

class BranchingDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "variable_features": torch.tensor(sample["variable_features"], dtype=torch.float32),
            "constraint_features": torch.tensor(sample["constraint_features"], dtype=torch.float32),
            "edge_indices": torch.tensor(sample["edge_indices"], dtype=torch.long),
            "edge_values": torch.tensor(sample["edge_values"], dtype=torch.float32),
            "expert_action": torch.tensor(sample["expert_action"], dtype=torch.long),
            "action_mask": torch.tensor(sample["action_mask"], dtype=torch.bool),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    return batch

def load_data(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


def train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    val_size = int(len(data) * val_ratio)
    val_indices = set(indices[:val_size])
    
    train_data = [data[i] for i in range(len(data)) if i not in val_indices]
    val_data = [data[i] for i in range(len(data)) if i in val_indices]
    
    return train_data, val_data


def compute_topk_accuracy(
    scores: torch.Tensor,
    expert_action: int,
    action_mask: torch.Tensor,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    valid_indices = torch.where(action_mask)[0]
    valid_scores = scores[action_mask]
    
    k_max = min(max(k_values), valid_scores.numel())
    _, topk_local = valid_scores.topk(k_max)
    topk_global = valid_indices[topk_local]
    
    results = {}
    for k in k_values:
        topk_set = set(topk_global[:k].tolist())
        results[f"top{k}"] = 1.0 if expert_action in topk_set else 0.0
    
    return results


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    topk_accs = {f"top{k}": 0.0 for k in [1, 3, 5]}
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        batch_loss = 0.0
        
        for sample in batch:
            var_feats = sample["variable_features"].to(device)
            con_feats = sample["constraint_features"].to(device)
            edge_index = sample["edge_indices"].to(device)
            edge_attr = sample["edge_values"].to(device)
            action_mask = sample["action_mask"].to(device)
            expert_action = sample["expert_action"].item()
            
            scores = model(var_feats, con_feats, edge_index, edge_attr, action_mask)
            
            valid_scores = scores[action_mask]
            
            valid_indices = torch.where(action_mask)[0]
            expert_local_idx = (valid_indices == expert_action).nonzero(as_tuple=True)[0]
            
            if len(expert_local_idx) == 0:
                continue
            
            expert_local_idx = expert_local_idx[0]
            
            loss = F.cross_entropy(valid_scores.unsqueeze(0), expert_local_idx.unsqueeze(0))
            batch_loss += loss
            
            with torch.no_grad():
                accs = compute_topk_accuracy(scores, expert_action, action_mask)
                for k, v in accs.items():
                    topk_accs[k] += v
            
            total_samples += 1
        
        if total_samples > 0:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            pbar.set_postfix({
                "loss": f"{total_loss/total_samples:.4f}",
                "top1": f"{topk_accs['top1']/total_samples:.2%}",
            })
    
    if total_samples > 0:
        metrics = {
            "loss": total_loss / total_samples,
            **{k: v / total_samples for k, v in topk_accs.items()}
        }
    else:
        metrics = {"loss": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}
    
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    topk_accs = {f"top{k}": 0.0 for k in [1, 3, 5]}
    
    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        for sample in batch:
            var_feats = sample["variable_features"].to(device)
            con_feats = sample["constraint_features"].to(device)
            edge_index = sample["edge_indices"].to(device)
            edge_attr = sample["edge_values"].to(device)
            action_mask = sample["action_mask"].to(device)
            expert_action = sample["expert_action"].item()
            
            scores = model(var_feats, con_feats, edge_index, edge_attr, action_mask)
            
            valid_scores = scores[action_mask]
            valid_indices = torch.where(action_mask)[0]
            expert_local_idx = (valid_indices == expert_action).nonzero(as_tuple=True)[0]
            
            if len(expert_local_idx) == 0:
                continue
            
            expert_local_idx = expert_local_idx[0]
            loss = F.cross_entropy(valid_scores.unsqueeze(0), expert_local_idx.unsqueeze(0))
            total_loss += loss.item()
            
            accs = compute_topk_accuracy(scores, expert_action, action_mask)
            for k, v in accs.items():
                topk_accs[k] += v
            
            total_samples += 1
    
    if total_samples > 0:
        metrics = {
            "loss": total_loss / total_samples,
            **{k: v / total_samples for k, v in topk_accs.items()}
        }
    else:
        metrics = {"loss": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}
    
    return metrics


def train(
    model: nn.Module,
    train_data: List[Dict],
    val_data: List[Dict],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = None,
    save_path: Optional[str] = None,
    patience: int = 10,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = BranchingDataset(train_data)
    val_dataset = BranchingDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_top1": [], "val_top1": [],
        "train_top3": [], "val_top3": [],
        "train_top5": [], "val_top5": [],
    }
    
    best_val_top1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\nTraining on {device}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        val_metrics = evaluate(model, val_loader, device)
        
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        for k in [1, 3, 5]:
            history[f"train_top{k}"].append(train_metrics[f"top{k}"])
            history[f"val_top{k}"].append(val_metrics[f"top{k}"])
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Top-1: {val_metrics['top1']:.2%} | "
              f"Val Top-3: {val_metrics['top3']:.2%} | "
              f"Val Top-5: {val_metrics['top5']:.2%}")
        
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            patience_counter = 0
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_top1": best_val_top1,
                    "history": history,
                }, save_path)
                print(f"  -> Saved best model (Top-1: {best_val_top1:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break
    
    print("-" * 60)
    print(f"Training complete. Best Val Top-1: {best_val_top1:.2%} at epoch {best_epoch}")
    
    # Load best model if saved
    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from {save_path}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Bipartite GNN for MILP branching")
    parser.add_argument("--data", type=str, default="data/setcover_samples.pkl",
                        help="Path to training data")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--save_path", type=str, default="data/model_checkpoint.pt",
                        help="Path to save model checkpoint")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = load_data(args.data)
    
    # get feature dimensions from first sample
    var_dim = data[0]['variable_features'].shape[1]
    con_dim = data[0]['constraint_features'].shape[1]
    print(f"Feature dimensions: var={var_dim}, con={con_dim}")
    
    # Split data
    train_data, val_data = train_val_split(data, val_ratio=0.1, seed=args.seed)
    
    model = BipartiteGNN(var_dim=var_dim, con_dim=con_dim, emb_dim=args.emb_dim)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_path=args.save_path,
        patience=args.patience,
    )
    
    # Print final results
    print("\nFinal Results:")
    print(f"  Best Val Top-1: {max(history['val_top1']):.2%}")
    print(f"  Best Val Top-3: {max(history['val_top3']):.2%}")
    print(f"  Best Val Top-5: {max(history['val_top5']):.2%}")


if __name__ == "__main__":
    main()
