from __future__ import annotations

"""Export model weights + one graph to C headers."""

import argparse
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from gnn_model import BipartiteGNN


def format_c_array(
    array: np.ndarray,
    name: str,
    dtype: str = "float",
    max_per_line: int = 8,
) -> str:
    flat = array.flatten()
    size = len(flat)
    
    lines = [f"const {dtype} {name}[{size}] = {{"]
    
    for i in range(0, size, max_per_line):
        chunk = flat[i:i + max_per_line]
        if dtype == "float":
            values = ", ".join(f"{v:.8f}f" for v in chunk)
        elif dtype == "int":
            values = ", ".join(str(int(v)) for v in chunk)
        else:
            values = ", ".join(str(v) for v in chunk)
        
        if i + max_per_line < size:
            values += ","
        
        lines.append(f"    {values}")
    
    lines.append("};")
    
    return "\n".join(lines)


def export_weights_to_header(
    model: torch.nn.Module,
    filepath: str,
    include_shapes: bool = True,
) -> Dict[str, Tuple[int, ...]]:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    shapes = {}
    
    with open(filepath, 'w') as f:
        guard = os.path.basename(filepath).upper().replace('.', '_')
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"// Generated {datetime.now().isoformat()}\n")
        f.write(f"// {model.__class__.__name__}\n\n")
        
        params = {}
        for name, param in model.named_parameters():
            clean_name = name.replace('.', '_')
            params[clean_name] = param.detach().cpu().numpy()
            shapes[clean_name] = param.shape
        
        if include_shapes:
            f.write("// Layer shapes\n")
            for name, shape in shapes.items():
                shape_str = "x".join(str(s) for s in shape)
                f.write(f"// {name}: ({shape_str})\n")
            f.write("\n")
            
            f.write("// Dimension defines\n")
            if hasattr(model, 'var_dim'):
                f.write(f"#define VAR_DIM {model.var_dim}\n")
            if hasattr(model, 'con_dim'):
                f.write(f"#define CON_DIM {model.con_dim}\n")
            if hasattr(model, 'emb_dim'):
                f.write(f"#define EMB_DIM {model.emb_dim}\n")
            f.write("\n")
        
        for name, array in params.items():
            f.write(f"// Shape: {array.shape}\n")
            f.write(format_c_array(array, f"w_{name}", "float"))
            f.write("\n\n")
        
        f.write(f"#endif // {guard}\n")
    
    print(f"Exported {len(params)} weight arrays to {filepath}")
    return shapes


def coo_to_csr(
    edge_indices: np.ndarray,
    edge_values: np.ndarray,
    num_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_indices = edge_indices[0]
    col_indices = edge_indices[1]
    
    sorted_indices = np.lexsort((col_indices, row_indices))
    row_indices = row_indices[sorted_indices]
    col_indices = col_indices[sorted_indices]
    values = edge_values[sorted_indices]
    
    row_pointers = np.zeros(num_rows + 1, dtype=np.int32)
    for row in row_indices:
        row_pointers[row + 1] += 1
    row_pointers = np.cumsum(row_pointers)
    
    return row_pointers, col_indices.astype(np.int32), values.astype(np.float32)


def export_graph_to_header(
    sample: Dict[str, Any],
    filepath: str,
    include_features: bool = True,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    var_feats = sample['variable_features']
    con_feats = sample['constraint_features']
    edge_indices = sample['edge_indices']
    edge_values = sample['edge_values']
    
    n_vars = var_feats.shape[0]
    n_cons = con_feats.shape[0]
    n_edges = edge_indices.shape[1]
    var_dim = var_feats.shape[1]
    con_dim = con_feats.shape[1]
    
    row_pointers, col_indices, csr_values = coo_to_csr(
        edge_indices, edge_values, n_cons
    )
    
    stats = {
        'n_vars': n_vars,
        'n_cons': n_cons,
        'n_edges': n_edges,
        'var_dim': var_dim,
        'con_dim': con_dim,
        'density': n_edges / (n_vars * n_cons),
    }
    
    with open(filepath, 'w') as f:
        guard = os.path.basename(filepath).upper().replace('.', '_')
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"// Generated {datetime.now().isoformat()}\n\n")
        
        f.write("// Graph dimensions\n")
        f.write(f"#define NUM_VARS {n_vars}\n")
        f.write(f"#define NUM_CONS {n_cons}\n")
        f.write(f"#define NUM_EDGES {n_edges}\n")
        f.write(f"#define VAR_FEATURE_DIM {var_dim}\n")
        f.write(f"#define CON_FEATURE_DIM {con_dim}\n")
        f.write(f"// Density: {stats['density']:.4f}\n\n")
        
        f.write("// CSR format: Constraint-Variable adjacency matrix\n")
        f.write(format_c_array(row_pointers, "row_pointers", "int"))
        f.write("\n\n")
        
        f.write(format_c_array(col_indices, "col_indices", "int"))
        f.write("\n\n")
        
        f.write(format_c_array(csr_values, "edge_values", "float"))
        f.write("\n\n")
        
        if include_features:
            f.write(format_c_array(var_feats, "var_features", "float"))
            f.write("\n\n")
            
            f.write(format_c_array(con_feats, "con_features", "float"))
            f.write("\n\n")
        
        if 'expert_action' in sample:
            f.write(f"#define EXPERT_ACTION {sample['expert_action']}\n\n")
        
        if 'action_mask' in sample:
            mask = sample['action_mask'].astype(np.int32)
            f.write(format_c_array(mask, "action_mask", "int"))
            f.write("\n\n")
        
        f.write(f"#endif // {guard}\n")
    
    print(f"Exported graph to {filepath} (vars={n_vars}, cons={n_cons}, edges={n_edges})")
    
    return stats


def export_multiple_graphs(
    samples: List[Dict[str, Any]],
    output_dir: str,
    num_samples: int = 5,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = []
    for i, sample in enumerate(samples[:num_samples]):
        filepath = os.path.join(output_dir, f"sample_graph_{i}.h")
        stats = export_graph_to_header(sample, filepath)
        all_stats.append(stats)
    
    return all_stats


def verify_csr_format(
    edge_indices: np.ndarray,
    edge_values: np.ndarray,
    num_rows: int,
) -> bool:
    row_pointers, col_indices, csr_values = coo_to_csr(edge_indices, edge_values, num_rows)
    
    # Reconstruct COO
    reconstructed_rows = []
    reconstructed_cols = []
    reconstructed_vals = []
    
    for row in range(num_rows):
        start = row_pointers[row]
        end = row_pointers[row + 1]
        for j in range(start, end):
            reconstructed_rows.append(row)
            reconstructed_cols.append(col_indices[j])
            reconstructed_vals.append(csr_values[j])
    
    reconstructed_rows = np.array(reconstructed_rows)
    reconstructed_cols = np.array(reconstructed_cols)
    reconstructed_vals = np.array(reconstructed_vals)
    
    # Sort original for comparison
    original_rows = edge_indices[0]
    original_cols = edge_indices[1]
    original_vals = edge_values
    
    sorted_idx = np.lexsort((original_cols, original_rows))
    original_rows = original_rows[sorted_idx]
    original_cols = original_cols[sorted_idx]
    original_vals = original_vals[sorted_idx]
    
    rows_match = np.array_equal(reconstructed_rows, original_rows)
    cols_match = np.array_equal(reconstructed_cols, original_cols)
    vals_match = np.allclose(reconstructed_vals, original_vals)
    
    if rows_match and cols_match and vals_match:
        print("CSR verification passed!")
        return True
    print("CSR verification failed!")
    return False


def main():
    parser = argparse.ArgumentParser(description="Export GNN model and data for hardware")
    parser.add_argument("--model", type=str, default="data/model_checkpoint.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/setcover_samples.pkl",
                        help="Path to training data (for sample graphs)")
    parser.add_argument("--output", type=str, default="exports/",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of sample graphs to export")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="Embedding dimension (if model not found)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, weights_only=False, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        var_dim = state_dict["var_embedding.weight"].shape[1]
        con_dim = state_dict["con_embedding.weight"].shape[1]
        emb_dim = state_dict["var_embedding.weight"].shape[0]
        model = BipartiteGNN(var_dim=var_dim, con_dim=con_dim, emb_dim=emb_dim)
        model.load_state_dict(state_dict)
    else:
        model = BipartiteGNN(var_dim=19, con_dim=5, emb_dim=args.emb_dim)
    
    model.eval()
    
    weights_path = os.path.join(args.output, "weights.h")
    export_weights_to_header(model, weights_path)
    
    if os.path.exists(args.data):
        with open(args.data, 'rb') as f:
            samples = pickle.load(f)
        export_multiple_graphs(samples, args.output, args.num_samples)
        sample = samples[0]
        verify_csr_format(
            sample['edge_indices'],
            sample['edge_values'],
            sample['constraint_features'].shape[0]
        )
    print(f"Export complete: {args.output}")


if __name__ == "__main__":
    main()
