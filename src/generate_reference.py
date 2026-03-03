from __future__ import annotations

"""Generate C headers with reference policy scores from a trained checkpoint."""

import argparse
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

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


def load_model(checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    var_dim = state_dict["var_embedding.weight"].shape[1]
    con_dim = state_dict["con_embedding.weight"].shape[1]
    emb_dim = state_dict["var_embedding.weight"].shape[0]

    model = BipartiteGNN(var_dim=var_dim, con_dim=con_dim, emb_dim=emb_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_sample(model: torch.nn.Module, sample: Dict[str, Any]) -> np.ndarray:
    with torch.no_grad():
        var_feats = torch.tensor(sample["variable_features"], dtype=torch.float32)
        con_feats = torch.tensor(sample["constraint_features"], dtype=torch.float32)
        edge_indices = torch.tensor(sample["edge_indices"], dtype=torch.long)
        edge_values = torch.tensor(sample["edge_values"], dtype=torch.float32)

        scores = model(var_feats, con_feats, edge_indices, edge_values, action_mask=None)
        return scores.detach().cpu().numpy().astype(np.float32)


def write_reference_header(
    filepath: str,
    sample_index: int,
    scores: np.ndarray,
    checkpoint_path: str,
    samples_path: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    guard = os.path.basename(filepath).upper().replace(".", "_")

    with open(filepath, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"// Generated {datetime.now().isoformat()}\n")
        f.write(f"// Checkpoint: {checkpoint_path}\n")
        f.write(f"// Samples: {samples_path}\n")
        f.write(f"// Sample index: {sample_index}\n\n")
        f.write(f"#define REFERENCE_NUM_VARS {scores.shape[0]}\n\n")
        f.write(format_c_array(scores, f"reference_scores_{sample_index}", "float"))
        f.write("\n\n")
        f.write(f"#endif // {guard}\n")


def generate_references(
    checkpoint_path: str,
    samples_path: str,
    output_dir: str,
    num_samples: int,
) -> None:
    with open(samples_path, "rb") as f:
        samples: List[Dict[str, Any]] = pickle.load(f)

    model = load_model(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    count = min(num_samples, len(samples))
    for i in range(count):
        scores = score_sample(model, samples[i])
        out_path = os.path.join(output_dir, f"reference_scores_{i}.h")
        write_reference_header(
            filepath=out_path,
            sample_index=i,
            scores=scores,
            checkpoint_path=checkpoint_path,
            samples_path=samples_path,
        )
        print(f"Exported {out_path} (num_scores={scores.shape[0]})")

    print(f"Reference generation complete: {output_dir} ({count} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reference score headers")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Path to samples pickle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for reference headers",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample graphs to score",
    )
    args = parser.parse_args()

    generate_references(
        checkpoint_path=args.checkpoint,
        samples_path=args.samples,
        output_dir=args.output,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
