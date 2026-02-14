from __future__ import annotations

"""Collect {NodeBipartite, StrongBranchingScores} samples from Ecole."""

import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import ecole


def create_instance_generator(
    n_rows: int = 500,
    n_cols: int = 1000,
    density: float = 0.05,
    seed: int = 42
) -> ecole.instance.SetCoverGenerator:
    generator = ecole.instance.SetCoverGenerator(
        n_rows=n_rows,
        n_cols=n_cols,
        density=density,
    )
    generator.seed(seed)
    return generator


def create_branching_environment(use_strong_branching: bool = True) -> ecole.environment.Branching:
    if use_strong_branching:
        # Combined observation: NodeBipartite + StrongBranchingScores
        obs_func = (
            ecole.observation.NodeBipartite(),
            ecole.observation.StrongBranchingScores(),
        )
    else:
        obs_func = ecole.observation.NodeBipartite()
    
    env = ecole.environment.Branching(
        observation_function=obs_func,
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "lp_iterations": ecole.reward.LpIterations(),
        },
    )
    return env


def get_strong_branching_action(
    action_set: np.ndarray,
    strong_branching_scores: np.ndarray,
) -> int:
    valid_scores = strong_branching_scores[action_set]
    best_local_idx = np.argmax(valid_scores)
    return int(action_set[best_local_idx])


def _ensure_finite(x: np.ndarray, *, name: str, sanitize: bool) -> np.ndarray:
    mask = ~np.isfinite(x)
    if not mask.any():
        return x
    if not sanitize:
        raise ValueError(f"{name} contains {int(mask.sum())} non-finite values")
    x = x.copy()
    x[mask] = 0.0
    return x


def extract_sample_from_observation(
    obs: Any,
    action_set: np.ndarray,
    expert_action: int,
    instance_id: int,
    step_id: int,
    sanitize_features: bool,
) -> Dict[str, Any]:
    var_feats = _ensure_finite(obs.variable_features, name="variable_features", sanitize=sanitize_features)
    con_feats = _ensure_finite(obs.row_features, name="constraint_features", sanitize=sanitize_features)
    
    edge_indices = obs.edge_features.indices.astype(np.int64)
    edge_values = obs.edge_features.values.flatten().astype(np.float32)
    
    n_vars = var_feats.shape[0]
    action_mask = np.zeros(n_vars, dtype=bool)
    action_mask[action_set] = True
    
    return {
        "variable_features": var_feats.astype(np.float32),
        "constraint_features": con_feats.astype(np.float32),
        "edge_indices": edge_indices,
        "edge_values": edge_values,
        "expert_action": int(expert_action),
        "action_mask": action_mask,
        "instance_id": int(instance_id),
        "step_id": int(step_id),
    }


def collect_samples(
    num_samples: int = 10000,
    n_rows: int = 500,
    n_cols: int = 1000,
    density: float = 0.05,
    seed: int = 42,
    max_steps_per_instance: int = 100,
    verbose: bool = True,
    use_strong_branching: bool = True,
    sanitize_features: bool = False,
) -> List[Dict[str, Any]]:
    generator = create_instance_generator(n_rows, n_cols, density, seed)
    env = create_branching_environment(use_strong_branching=use_strong_branching)
    env.seed(seed)
    
    samples: List[Dict[str, Any]] = []
    instance_id = 0
    
    var_dim = None
    con_dim = None
    
    if verbose:
        print(f"Collecting samples={num_samples} (rows={n_rows}, cols={n_cols}, density={density})")
    
    if not use_strong_branching:
        raise ValueError("collect_samples requires strong branching labels; set use_strong_branching=True")

    while len(samples) < num_samples:
        instance = next(generator)
        instance_id += 1
        
        obs, action_set, _, done, _ = env.reset(instance)
        step_id = 0
        
        while not done and len(samples) < num_samples and step_id < max_steps_per_instance:
            node_obs, sb_scores = obs
            if sb_scores is None:
                raise RuntimeError("Expected StrongBranchingScores but got None")
            
            if var_dim is None:
                var_dim = node_obs.variable_features.shape[1]
                con_dim = node_obs.row_features.shape[1]
                if verbose:
                    print(f"Feature dimensions: var={var_dim}, con={con_dim}")
            
            expert_action = get_strong_branching_action(action_set, sb_scores)
            
            sample = extract_sample_from_observation(
                node_obs,
                action_set,
                expert_action,
                instance_id,
                step_id,
                sanitize_features=sanitize_features,
            )
            samples.append(sample)
            
            obs, action_set, _, done, _ = env.step(expert_action)
            step_id += 1
            
            if verbose and len(samples) % 500 == 0:
                print(f"Collected {len(samples)}/{num_samples} samples "
                      f"(instance {instance_id}, step {step_id})")
    
    if verbose:
        avg = (len(samples) / instance_id) if instance_id else 0.0
        print(f"Done. samples={len(samples)} instances={instance_id} avg_samples_per_instance={avg:.1f}")
    
    return samples


def verify_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic consistency checks and dataset stats."""
    if not samples:
        return {"error": "No samples collected"}
    
    # Get statistics
    var_dims = [s["variable_features"].shape[1] for s in samples]
    con_dims = [s["constraint_features"].shape[1] for s in samples]
    n_vars = [s["variable_features"].shape[0] for s in samples]
    n_cons = [s["constraint_features"].shape[0] for s in samples]
    n_edges = [s["edge_indices"].shape[1] for s in samples]
    
    stats = {
        "num_samples": len(samples),
        "var_feature_dim": var_dims[0],
        "con_feature_dim": con_dims[0],
        "var_dim_consistent": all(d == var_dims[0] for d in var_dims),
        "con_dim_consistent": all(d == con_dims[0] for d in con_dims),
        "avg_num_vars": np.mean(n_vars),
        "avg_num_cons": np.mean(n_cons),
        "avg_num_edges": np.mean(n_edges),
        "min_num_edges": min(n_edges),
        "max_num_edges": max(n_edges),
    }
    
    return stats


def save_samples(samples: List[Dict[str, Any]], filepath: str):
    """Save samples to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(samples, f)
    print(f"Wrote {len(samples)} samples to {filepath}")


def load_samples(filepath: str) -> List[Dict[str, Any]]:
    """Load samples from a pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Collect MILP branching samples")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to collect")
    parser.add_argument("--n_rows", type=int, default=500,
                        help="Number of constraints")
    parser.add_argument("--n_cols", type=int, default=1000,
                        help="Number of variables")
    parser.add_argument("--density", type=float, default=0.05,
                        help="Constraint matrix density")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="data/setcover_samples.pkl",
                        help="Output file path")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max branching steps per instance")
    parser.add_argument("--sanitize_features", action="store_true",
                        help="Replace non-finite features with 0 instead of raising")
    args = parser.parse_args()
    
    # Collect samples
    samples = collect_samples(
        num_samples=args.num_samples,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        density=args.density,
        seed=args.seed,
        max_steps_per_instance=args.max_steps,
        verbose=True,
        sanitize_features=args.sanitize_features,
    )
    
    # Verify samples
    stats = verify_samples(samples)
    print("\nSample Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save samples
    save_samples(samples, args.output)


if __name__ == "__main__":
    main()
