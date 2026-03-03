#include "accelerator.h"

#include <cmath>
#include <cstdio>

namespace sample0 {
#include "sample_graph_0.h"
#include "reference_scores_0.h"
static constexpr int kNumVars = NUM_VARS;
static constexpr int kNumCons = NUM_CONS;
static constexpr int kNumEdges = NUM_EDGES;
}  // namespace sample0
#undef NUM_VARS
#undef NUM_CONS
#undef NUM_EDGES
#undef VAR_FEATURE_DIM
#undef CON_FEATURE_DIM
#undef EXPERT_ACTION
#undef REFERENCE_NUM_VARS

namespace sample1 {
#include "sample_graph_1.h"
#include "reference_scores_1.h"
static constexpr int kNumVars = NUM_VARS;
static constexpr int kNumCons = NUM_CONS;
static constexpr int kNumEdges = NUM_EDGES;
}  // namespace sample1
#undef NUM_VARS
#undef NUM_CONS
#undef NUM_EDGES
#undef VAR_FEATURE_DIM
#undef CON_FEATURE_DIM
#undef EXPERT_ACTION
#undef REFERENCE_NUM_VARS

namespace sample2 {
#include "sample_graph_2.h"
#include "reference_scores_2.h"
static constexpr int kNumVars = NUM_VARS;
static constexpr int kNumCons = NUM_CONS;
static constexpr int kNumEdges = NUM_EDGES;
}  // namespace sample2
#undef NUM_VARS
#undef NUM_CONS
#undef NUM_EDGES
#undef VAR_FEATURE_DIM
#undef CON_FEATURE_DIM
#undef EXPERT_ACTION
#undef REFERENCE_NUM_VARS

namespace sample3 {
#include "sample_graph_3.h"
#include "reference_scores_3.h"
static constexpr int kNumVars = NUM_VARS;
static constexpr int kNumCons = NUM_CONS;
static constexpr int kNumEdges = NUM_EDGES;
}  // namespace sample3
#undef NUM_VARS
#undef NUM_CONS
#undef NUM_EDGES
#undef VAR_FEATURE_DIM
#undef CON_FEATURE_DIM
#undef EXPERT_ACTION
#undef REFERENCE_NUM_VARS

namespace sample4 {
#include "sample_graph_4.h"
#include "reference_scores_4.h"
static constexpr int kNumVars = NUM_VARS;
static constexpr int kNumCons = NUM_CONS;
static constexpr int kNumEdges = NUM_EDGES;
}  // namespace sample4
#undef NUM_VARS
#undef NUM_CONS
#undef NUM_EDGES
#undef VAR_FEATURE_DIM
#undef CON_FEATURE_DIM
#undef EXPERT_ACTION
#undef REFERENCE_NUM_VARS

struct GraphCase {
    const char* name;
    int n_vars;
    int n_cons;
    int n_edges;
    const float* var_features;
    const float* con_features;
    const int* row_ptr;
    const int* col_idx;
    const float* edge_vals;
    const int* t_row_ptr;
    const int* t_col_idx;
    const int* t_edge_map;
    const float* reference_scores;
};

bool run_sparse_sanity_test() {
    const int row_ptr[3] = {0, 2, 3};
    const int col_idx[3] = {0, 1, 1};
    const float edge_vals[3] = {2.0f, 3.0f, 5.0f};

    const int t_row_ptr[3] = {0, 1, 3};
    const int t_col_idx[3] = {0, 0, 1};
    const int t_edge_map[3] = {0, 1, 2};

    const float var_scalar[2] = {1.0f, 10.0f};
    float con_scalar[2] = {0.0f, 0.0f};
    float var_back[2] = {0.0f, 0.0f};

    for (int c = 0; c < 2; ++c) {
        for (int e = row_ptr[c]; e < row_ptr[c + 1]; ++e) {
            const int v = col_idx[e];
            con_scalar[c] += var_scalar[v] * edge_vals[e];
        }
    }
    for (int v = 0; v < 2; ++v) {
        for (int t = t_row_ptr[v]; t < t_row_ptr[v + 1]; ++t) {
            const int c = t_col_idx[t];
            const int e = t_edge_map[t];
            var_back[v] += con_scalar[c] * edge_vals[e];
        }
    }

    const bool pass =
        std::fabs(con_scalar[0] - 32.0f) < 1e-6f &&
        std::fabs(con_scalar[1] - 50.0f) < 1e-6f &&
        std::fabs(var_back[0] - 64.0f) < 1e-6f &&
        std::fabs(var_back[1] - 346.0f) < 1e-6f;

    std::printf(
        "Sparse sanity: con=[%.3f, %.3f] var_back=[%.3f, %.3f] -> %s\n",
        con_scalar[0],
        con_scalar[1],
        var_back[0],
        var_back[1],
        pass ? "PASS" : "FAIL"
    );
    return pass;
}

void print_ops(const GraphCase& c) {
    const long long dense_var_emb = static_cast<long long>(c.n_vars) * EMB_DIM_CONST * 19LL;
    const long long dense_con_emb = static_cast<long long>(c.n_cons) * EMB_DIM_CONST * 5LL;
    const long long dense_con_update =
        static_cast<long long>(c.n_cons) * EMB_DIM_CONST * (2LL * EMB_DIM_CONST);
    const long long dense_var_update =
        static_cast<long long>(c.n_vars) * EMB_DIM_CONST * (2LL * EMB_DIM_CONST);
    const long long dense_policy0 =
        static_cast<long long>(c.n_vars) * EMB_DIM_CONST * EMB_DIM_CONST;
    const long long dense_policy2 = static_cast<long long>(c.n_vars) * EMB_DIM_CONST;
    const long long dense_total = dense_var_emb + dense_con_emb + dense_con_update + dense_var_update
                                  + dense_policy0 + dense_policy2;

    const long long sparse_gather = static_cast<long long>(c.n_edges) * EMB_DIM_CONST;
    const long long sparse_scatter = static_cast<long long>(c.n_edges) * EMB_DIM_CONST;
    const long long sparse_total = sparse_gather + sparse_scatter;

    std::printf(
        "  Ops estimate: dense_mac=%lld sparse_mac=%lld (gather=%lld scatter=%lld)\n",
        dense_total,
        sparse_total,
        sparse_gather,
        sparse_scatter
    );
}

bool run_graph_case(const GraphCase& c, float max_tol, float mean_tol) {
    if (c.n_vars > MAX_VARS || c.n_cons > MAX_CONS || c.n_edges > MAX_EDGES) {
        std::printf(
            "%s: exceeds static caps (vars=%d/%d cons=%d/%d edges=%d/%d) -> FAIL\n",
            c.name,
            c.n_vars,
            MAX_VARS,
            c.n_cons,
            MAX_CONS,
            c.n_edges,
            MAX_EDGES
        );
        return false;
    }

    static float scores[MAX_VARS];
    for (int i = 0; i < c.n_vars; ++i) {
        scores[i] = 0.0f;
    }

    inference_kernel(
        c.var_features,
        c.con_features,
        c.row_ptr,
        c.col_idx,
        c.edge_vals,
        c.t_row_ptr,
        c.t_col_idx,
        c.t_edge_map,
        scores,
        c.n_vars,
        c.n_cons,
        c.n_edges
    );

    float max_abs_err = 0.0f;
    double mean_abs_err = 0.0;
    for (int i = 0; i < c.n_vars; ++i) {
        const float err = std::fabs(scores[i] - c.reference_scores[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        mean_abs_err += static_cast<double>(err);
    }
    mean_abs_err /= static_cast<double>(c.n_vars);

    const bool pass = (max_abs_err < max_tol) && (mean_abs_err < mean_tol);
    std::printf(
        "%s: max_abs_err=%.8f mean_abs_err=%.8f (tol max=%.1e mean=%.1e) -> %s\n",
        c.name,
        max_abs_err,
        static_cast<float>(mean_abs_err),
        max_tol,
        mean_tol,
        pass ? "PASS" : "FAIL"
    );
    print_ops(c);
    return pass;
}

int main() {
    const GraphCase cases[5] = {
        {
            "sample_graph_0",
            sample0::kNumVars,
            sample0::kNumCons,
            sample0::kNumEdges,
            sample0::var_features,
            sample0::con_features,
            sample0::row_pointers,
            sample0::col_indices,
            sample0::edge_values,
            sample0::t_row_pointers,
            sample0::t_col_indices,
            sample0::t_edge_map,
            sample0::reference_scores_0,
        },
        {
            "sample_graph_1",
            sample1::kNumVars,
            sample1::kNumCons,
            sample1::kNumEdges,
            sample1::var_features,
            sample1::con_features,
            sample1::row_pointers,
            sample1::col_indices,
            sample1::edge_values,
            sample1::t_row_pointers,
            sample1::t_col_indices,
            sample1::t_edge_map,
            sample1::reference_scores_1,
        },
        {
            "sample_graph_2",
            sample2::kNumVars,
            sample2::kNumCons,
            sample2::kNumEdges,
            sample2::var_features,
            sample2::con_features,
            sample2::row_pointers,
            sample2::col_indices,
            sample2::edge_values,
            sample2::t_row_pointers,
            sample2::t_col_indices,
            sample2::t_edge_map,
            sample2::reference_scores_2,
        },
        {
            "sample_graph_3",
            sample3::kNumVars,
            sample3::kNumCons,
            sample3::kNumEdges,
            sample3::var_features,
            sample3::con_features,
            sample3::row_pointers,
            sample3::col_indices,
            sample3::edge_values,
            sample3::t_row_pointers,
            sample3::t_col_indices,
            sample3::t_edge_map,
            sample3::reference_scores_3,
        },
        {
            "sample_graph_4",
            sample4::kNumVars,
            sample4::kNumCons,
            sample4::kNumEdges,
            sample4::var_features,
            sample4::con_features,
            sample4::row_pointers,
            sample4::col_indices,
            sample4::edge_values,
            sample4::t_row_pointers,
            sample4::t_col_indices,
            sample4::t_edge_map,
            sample4::reference_scores_4,
        },
    };

    bool all_pass = true;
    all_pass &= run_sparse_sanity_test();

    const float max_tol = 1e-4f;
    const float mean_tol = 1e-5f;
    for (const GraphCase& c : cases) {
        all_pass &= run_graph_case(c, max_tol, mean_tol);
    }

    std::printf("\nOverall: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
