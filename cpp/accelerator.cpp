#include "accelerator.h"

#include "weights.h"

namespace {

static_assert(EMB_DIM == EMB_DIM_CONST, "weights.h EMB_DIM must match EMB_DIM_CONST");

data_t g_var_emb[MAX_VARS * EMB_DIM_CONST];
data_t g_con_emb[MAX_CONS * EMB_DIM_CONST];
data_t g_var_agg[MAX_VARS * EMB_DIM_CONST];
data_t g_con_agg[MAX_CONS * EMB_DIM_CONST];
data_t g_var_next[MAX_VARS * EMB_DIM_CONST];
data_t g_con_next[MAX_CONS * EMB_DIM_CONST];
data_t g_policy_hidden[MAX_VARS * EMB_DIM_CONST];

inline data_t relu(data_t x) {
    return x > 0.0f ? x : 0.0f;
}

void zero_matrix(data_t* buffer, int rows) {
    const int total = rows * EMB_DIM_CONST;
    for (int i = 0; i < total; ++i) {
        buffer[i] = 0.0f;
    }
}

void dense_relu(
    const data_t* in,
    int rows,
    int in_dim,
    const float* weight,
    const float* bias,
    data_t* out,
    int out_dim
) {
    for (int r = 0; r < rows; ++r) {
        const int in_base = r * in_dim;
        const int out_base = r * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            data_t acc = bias[o];
            const int w_base = o * in_dim;
            for (int i = 0; i < in_dim; ++i) {
                acc += in[in_base + i] * weight[w_base + i];
            }
            out[out_base + o] = relu(acc);
        }
    }
}

void dense_relu_concat_64(
    const data_t* left,
    const data_t* right,
    int rows,
    const float* weight,
    const float* bias,
    data_t* out
) {
    for (int r = 0; r < rows; ++r) {
        const int base = r * EMB_DIM_CONST;
        for (int o = 0; o < EMB_DIM_CONST; ++o) {
            data_t acc = bias[o];
            const int w_base = o * (2 * EMB_DIM_CONST);
            for (int i = 0; i < EMB_DIM_CONST; ++i) {
                acc += left[base + i] * weight[w_base + i];
            }
            for (int i = 0; i < EMB_DIM_CONST; ++i) {
                acc += right[base + i] * weight[w_base + EMB_DIM_CONST + i];
            }
            out[base + o] = relu(acc);
        }
    }
}

void gather_constraints(
    const int* row_ptr,
    const int* col_idx,
    const float* edge_vals,
    const data_t* var_emb,
    data_t* con_agg,
    int n_cons
) {
    zero_matrix(con_agg, n_cons);

    for (int c = 0; c < n_cons; ++c) {
#if defined(HLS) && HLS
#pragma HLS PIPELINE II = 1
#endif
        const int row_start = row_ptr[c];
        const int row_end = row_ptr[c + 1];
        const int con_base = c * EMB_DIM_CONST;
        for (int e = row_start; e < row_end; ++e) {
            const int v = col_idx[e];
            const float edge_val = edge_vals[e];
            const int var_base = v * EMB_DIM_CONST;
            for (int d = 0; d < EMB_DIM_CONST; ++d) {
                con_agg[con_base + d] +=
                    var_emb[var_base + d] * edge_val * w_edge_weight_weight[d];
            }
        }
    }
}

void scatter_variables(
    const int* t_row_ptr,
    const int* t_col_idx,
    const int* t_edge_map,
    const float* edge_vals,
    const data_t* con_emb,
    data_t* var_agg,
    int n_vars
) {
    zero_matrix(var_agg, n_vars);

    for (int v = 0; v < n_vars; ++v) {
#if defined(HLS) && HLS
#pragma HLS PIPELINE II = 1
#endif
        const int row_start = t_row_ptr[v];
        const int row_end = t_row_ptr[v + 1];
        const int var_base = v * EMB_DIM_CONST;
        for (int t_idx = row_start; t_idx < row_end; ++t_idx) {
            const int c = t_col_idx[t_idx];
            const int edge_idx = t_edge_map[t_idx];
            const float edge_val = edge_vals[edge_idx];
            const int con_base = c * EMB_DIM_CONST;
            for (int d = 0; d < EMB_DIM_CONST; ++d) {
                var_agg[var_base + d] +=
                    con_emb[con_base + d] * edge_val * w_edge_weight_weight[d];
            }
        }
    }
}

void policy_head(const data_t* var_emb, float* scores, int n_vars) {
    dense_relu(
        var_emb,
        n_vars,
        EMB_DIM_CONST,
        w_policy_0_weight,
        w_policy_0_bias,
        g_policy_hidden,
        EMB_DIM_CONST
    );

    for (int v = 0; v < n_vars; ++v) {
        const int base = v * EMB_DIM_CONST;
        data_t acc = w_policy_2_bias[0];
        for (int d = 0; d < EMB_DIM_CONST; ++d) {
            acc += g_policy_hidden[base + d] * w_policy_2_weight[d];
        }
        scores[v] = acc;
    }
}

}  // namespace

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
) {
    if (
        var_feats == nullptr || con_feats == nullptr || row_ptr == nullptr || col_idx == nullptr
        || edge_vals == nullptr || t_row_ptr == nullptr || t_col_idx == nullptr
        || t_edge_map == nullptr || scores == nullptr || n_vars <= 0 || n_cons <= 0 || n_edges <= 0
        || n_vars > MAX_VARS || n_cons > MAX_CONS || n_edges > MAX_EDGES
    ) {
        if (scores != nullptr && n_vars > 0) {
            for (int i = 0; i < n_vars; ++i) {
                scores[i] = 0.0f;
            }
        }
        return;
    }

    dense_relu(
        var_feats,
        n_vars,
        VAR_DIM,
        w_var_embedding_weight,
        w_var_embedding_bias,
        g_var_emb,
        EMB_DIM_CONST
    );
    dense_relu(
        con_feats,
        n_cons,
        CON_DIM,
        w_con_embedding_weight,
        w_con_embedding_bias,
        g_con_emb,
        EMB_DIM_CONST
    );

    gather_constraints(row_ptr, col_idx, edge_vals, g_var_emb, g_con_agg, n_cons);
    dense_relu_concat_64(
        g_con_emb,
        g_con_agg,
        n_cons,
        w_con_update_weight,
        w_con_update_bias,
        g_con_next
    );

    scatter_variables(
        t_row_ptr,
        t_col_idx,
        t_edge_map,
        edge_vals,
        g_con_next,
        g_var_agg,
        n_vars
    );
    dense_relu_concat_64(
        g_var_emb,
        g_var_agg,
        n_vars,
        w_var_update_weight,
        w_var_update_bias,
        g_var_next
    );

    policy_head(g_var_next, scores, n_vars);
}
