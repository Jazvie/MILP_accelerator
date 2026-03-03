#ifndef MILP_ACCELERATOR_H
#define MILP_ACCELERATOR_H

constexpr int MAX_VARS = 2048;
constexpr int MAX_CONS = 2048;
constexpr int MAX_EDGES = 30000;
constexpr int EMB_DIM_CONST = 64;

using data_t = float;

void inference_kernel(
    const float* var_feats,
    const float* con_feats,
    const int* row_ptr,
    const int* col_idx,
    const float* edge_vals,
    const int* t_row_ptr,
    const int* t_col_idx,
    const int* t_edge_map,
    float* scores,
    int n_vars,
    int n_cons,
    int n_edges
);

#endif  // MILP_ACCELERATOR_H
