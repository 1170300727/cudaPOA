/* #include <__clang_cuda_builtin_vars.h> */
/* #include <__clang_cuda_builtin_vars.h> */
/* #include <__clang_cuda_builtin_vars.h> */
#include <bits/stdint-uintn.h>
#include <climits>
#include <cmath>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>
#include <stdio.h>
#include "abpoa.h"
#include "helper_cuda.h"
#include "helper_string.h"
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <vector_types.h>
#include "abpoa_align.h"
#include "simd_instruction.h"
#include "simd_abpoa_align.h"
#include "utils.h"

typedef struct {
    const int reg_n, bits_n, log_num, num_of_value, size;
    int inf_min; // based on penalty of mismatch and GAP_OE1
} SIMD_para_t;

#define __DEBUG__\

/* no need SIMD_para_t */\
int INF_MIN;
__device__ const int d_float_min = INT_MIN;
#define SIMDShiftOneNi8  1
#define SIMDShiftOneNi16 2
#define SIMDShiftOneNi32 4
#define SIMDShiftOneNi64 8

#ifdef __AVX512F__
SIMD_para_t _simd_p8  = {512,  8, 6, 64, 64, -1};
SIMD_para_t _simd_p16 = {512, 16, 5, 32, 64, -1};
SIMD_para_t _simd_p32 = {512, 32, 4, 16, 64, -1};
SIMD_para_t _simd_p64 = {512, 64, 3,  8, 64, -1};
#define SIMDTotalBytes 64
#elif defined(__AVX2__)
SIMD_para_t _simd_p8  = {256,  8, 5, 32, 32, -1};
SIMD_para_t _simd_p16 = {256, 16, 4, 16, 32, -1};
SIMD_para_t _simd_p32 = {256, 32, 3,  8, 32, -1};
SIMD_para_t _simd_p64 = {256, 64, 2,  4, 32, -1};
#define SIMDTotalBytes 32
#else
SIMD_para_t _simd_p8  = {128,  8, 4, 16, 16, -1};
SIMD_para_t _simd_p16 = {128, 16, 3,  8, 16, -1};
SIMD_para_t _simd_p32 = {128, 32, 2,  4, 16, -1};
SIMD_para_t _simd_p64 = {128, 64, 1,  2, 16, -1};
#define SIMDTotalBytes 16
#endif

#define print_simd(s, str, score_t) {                                   \
    int _i; score_t *_a = (score_t*)(s);                                \
    fprintf(stderr, "%s\t", str);                                       \
    for (_i = 0; _i < SIMDTotalBytes / (int)sizeof(score_t); ++_i) {    \
        fprintf(stderr, "%d\t", _a[_i]);                                \
    } fprintf(stderr, "\n");                                            \
}

#define simd_abpoa_print_ag_matrix(score_t) {                   \
    for (j = 0; j <= matrix_row_n-2; ++j) {	                    \
        printf("index: %d\t", j);	                            \
        dp_h = DP_HEF + j * 3 * dp_sn; dp_e1 = dp_h + dp_sn;	\
        _dp_h = (score_t*)dp_h, _dp_e1 = (score_t*)dp_e1;	    \
        for (i = dp_beg[j]; i <= dp_end[j]; ++i) {	            \
            printf("%d:(%d,%d)\t", i, _dp_h[i], _dp_e1[i]);	    \
        } printf("\n");	                                        \
    }	                                                        \
}

#define debug_simd_abpoa_print_cg_matrix_row(str, score_t, index_i) {                                   \
    score_t *_dp_h = (score_t*)dp_h, *_dp_e1 = (score_t*)dp_e1;                                         \
    score_t *_dp_e2 = (score_t*)dp_e2, *_dp_f1 = (score_t*)dp_f1, *_dp_f2 = (score_t*)dp_f2;            \
    fprintf(stderr, "%s\tindex: %d\t", str, index_i);	                                                \
    for (i = dp_beg[index_i]; i <= (dp_end[index_i]/16+1)*16-1; ++i) {	                                \
        fprintf(stderr, "%d:(%d,%d,%d,%d,%d)\t", i, _dp_h[i], _dp_e1[i],_dp_e2[i], _dp_f1[i],_dp_f2[i]);\
    } fprintf(stderr, "\n");	                                                                        \
}

#define simd_abpoa_print_cg_matrix(score_t) {                                                               \
    FILE *file = fopen("simd_matrix.fa", "w"); \
    for (j = 0; j <= matrix_row_n-2; ++j) {	                                                                \
        fprintf(file, "index:%d\t\n", j);	                                                                \
        dp_h=DP_H2E2F+j*5*dp_sn; dp_e1=dp_h+dp_sn; dp_e2=dp_e1+dp_sn; dp_f1=dp_e2+dp_sn; dp_f2=dp_f1+dp_sn; \
        score_t *_dp_h=(score_t*)dp_h, *_dp_e1=(score_t*)dp_e1, *_dp_e2=(score_t*)dp_e2;                    \
        score_t *_dp_f1=(score_t*)dp_f1, *_dp_f2=(score_t*)dp_f2;	                                        \
        for (i = dp_beg[j]; i <= dp_end[j]; ++i) {                                                          \
		  fprintf(file, "%d:(%d,%d,%d,%d,%d)\t", i, _dp_h[i], _dp_e1[i],_dp_e2[i], _dp_f1[i],_dp_f2[i]);\
          fprintf(file, "\n");	                                                                        \
        } \
    }	                                                                                                \
	fclose(file);\
}

#define simd_abpoa_print_lg_matrix(score_t) {       \
    for (j = 0; j <= graph->node_n-2; ++j) {	    \
        printf("index: %d\t", j);	                \
        dp_h = DP_H + j * dp_sn;	                \
        _dp_h = (score_t*)dp_h;	                    \
        for (i = dp_beg[j]; i <= dp_end[j]; ++i) {	\
            printf("%d:(%d)\t", i, _dp_h[i]);	    \
        } printf("\n");	                            \
    }	                                            \
}

/* max_pos_left/right: left/right boundary of max column index for each row, based on the pre_nodes' DP score */
/* === workflow of alignment === */
/* a. global:
 * (1) alloc mem 
 * (2) init for first row
 * (3) DP for each row
 * (3.2) if use_ada, update max_pos_left/right
 * (4) find best_i/j, backtrack
 * b. extend:
 * (1) alloc mem
 * (3) init for first row
 * (4) DP for each row
 * (3.2) find max of current row
 * (3.3) z-drop, set_max_score
 * (3.4) if use_ada, update max_pos_left/right
 */

// backtrack order:
// Match/Mismatch, Deletion, Insertion
#define simd_abpoa_lg_backtrack(score_t) {                                                                  \
    int i, j, k, pre_i, n_c = 0, s, m_c = 0, hit, id, _start_i, _start_j;                                   \
    SIMDi *dp_h; score_t *_dp_h=NULL, *_pre_dp_h; abpoa_cigar_t *cigar = 0;                                 \
    i = best_i, j = best_j, _start_i = best_i, _start_j = best_j;                                           \
    id = abpoa_graph_index_to_node_id(graph, i);                                                            \
    if (best_j < qlen) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, qlen-j, -1, qlen-1);   \
    dp_h = DP_H + i * dp_sn; _dp_h = (score_t*)dp_h;                                                        \
    while (i > 0 && j > 0) {                                                                                \
        if (abpt->align_mode == ABPOA_LOCAL_MODE && _dp_h[j] == 0) break;                                   \
        _start_i = i, _start_j = j;                                                                         \
        int *pre_index_i = pre_index[i];                                                                    \
        s = mat[m * graph->node[id].base + query[j-1]]; hit = 0;                                            \
        for (k = 0; k < pre_n[i]; ++k) {                                                                    \
            pre_i = pre_index_i[k];                                                                         \
            if (j-1 < dp_beg[pre_i] || j-1 > dp_end[pre_i]) continue;                                       \
            _pre_dp_h = (score_t*)(DP_H + pre_i * dp_sn);                                                   \
            if (_pre_dp_h[j-1] + s == _dp_h[j]) { /* match/mismatch */                                      \
                cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CMATCH, 1, id, j-1);                      \
                i = pre_i; --j; hit = 1; id = abpoa_graph_index_to_node_id(graph, i);                       \
                dp_h = DP_H + i * dp_sn; _dp_h = (score_t*)dp_h;                                            \
                ++res->n_aln_bases; res->n_matched_bases += s == mat[0] ? 1 : 0;                            \
                break;                                                                                      \
            }                                                                                               \
        }                                                                                                   \
        if (hit == 0) {                                                                                     \
            for (k = 0; k < pre_n[i]; ++k) {                                                                \
                pre_i = pre_index_i[k];                                                                     \
                if (j < dp_beg[pre_i] || j > dp_end[pre_i]) continue;                                       \
                _pre_dp_h = (score_t*)( DP_H + pre_i * dp_sn);                                              \
                if (_pre_dp_h[j] - gap_ext1 == _dp_h[j]) { /* deletion */                                   \
                    cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);                    \
                    i = pre_i; hit = 1; id = abpoa_graph_index_to_node_id(graph, i);                        \
                    dp_h = DP_H + i * dp_sn; _dp_h = (score_t*)dp_h;                                        \
                    break;                                                                                  \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        if (hit == 0) { /* insertion */                                                                     \
            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1); j--;                       \
            ++res->n_aln_bases;                                                                             \
        }                                                                                                   \
    }                                                                                                       \
    if (j > 0) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, j, -1, j-1);                   \
    /* reverse cigar */                                                                                     \
    res->graph_cigar = abpt->rev_cigar ? cigar : abpoa_reverse_cigar(n_c, cigar);                           \
    res->n_cigar = n_c;                                                                                     \
    res->node_e = best_i, res->query_e = best_j-1; /* 0-based */                                            \
    res->node_s = _start_i, res->query_s = _start_j-1;                                                      \
    /*abpoa_print_cigar(n_c, *graph_cigar, graph);*/                                                        \
}

#define simd_abpoa_ag_backtrack(score_t) {                                                                  \
    int i, j, k, pre_i, n_c = 0, s, m_c = 0, id, hit, cur_op = ABPOA_ALL_OP, _start_i, _start_j;            \
    score_t *_dp_h, *_dp_e1, *_dp_f1, *_pre_dp_h, *_pre_dp_e1; abpoa_cigar_t *cigar = 0;                    \
    i = best_i, j = best_j; _start_i = best_i, _start_j = best_j;                                           \
    id = abpoa_graph_index_to_node_id(graph, i);                                                            \
    if (best_j < qlen) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, qlen-j, -1, qlen-1);   \
    SIMDi *dp_h = DP_HEF + dp_sn * i * 3; _dp_h = (score_t*)dp_h;                                           \
    while (i > 0 && j > 0) {                                                                                \
        if (_dp_h[j] == 0) break;                                                                           \
        _start_i = i, _start_j = j;                                                                         \
        int *pre_index_i = pre_index[i];                                                                    \
        s = mat[m * graph->node[id].base + query[j-1]]; hit = 0;                                            \
        if (cur_op & ABPOA_M_OP) {                                                                          \
            for (k = 0; k < pre_n[i]; ++k) {                                                                \
                pre_i = pre_index_i[k];                                                                     \
                if (j-1 < dp_beg[pre_i] || j-1 > dp_end[pre_i]) continue;                                   \
                /* match/mismatch */                                                                        \
                _pre_dp_h = (score_t*)(DP_HEF + dp_sn * pre_i * 3);                                         \
                if (_pre_dp_h[j-1] + s == _dp_h[j]) {                                                       \
                    cur_op = ABPOA_ALL_OP; hit = 1;                                                         \
                    cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CMATCH, 1, id, j-1);                  \
                    i = pre_i; --j; id = abpoa_graph_index_to_node_id(graph, i);                            \
                    dp_h = DP_HEF + dp_sn * i * 3; _dp_h = (score_t*)dp_h;                                  \
                    ++res->n_aln_bases; res->n_matched_bases += s == mat[0] ? 1 : 0;                        \
                    break;                                                                                  \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        if (hit == 0 && cur_op & ABPOA_E1_OP) {                                                             \
            for (k = 0; k < pre_n[i]; ++k) {                                                                \
                pre_i = pre_index_i[k];                                                                     \
                if (j < dp_beg[pre_i] || j > dp_end[pre_i]) continue;                                       \
                _pre_dp_e1 = (score_t*)(DP_HEF + dp_sn * (pre_i * 3 + 1));                                  \
                if (cur_op & ABPOA_M_OP) {                                                                  \
                    if (_dp_h[j] == _pre_dp_e1[j]) { /* deletion */                                         \
                        _pre_dp_h = (score_t*)(DP_HEF + dp_sn * pre_i * 3);                                 \
                        if (_pre_dp_h[j] - gap_oe1 == _pre_dp_e1[j]) cur_op = ABPOA_M_OP;                   \
                        else cur_op = ABPOA_E1_OP;                                                          \
                        hit = 1;                                                                            \
                        cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);                \
                        i = pre_i; id = abpoa_graph_index_to_node_id(graph, i);                             \
                        dp_h = DP_HEF + dp_sn * i * 3; _dp_h = (score_t*)dp_h;                              \
                        break;                                                                              \
                    }                                                                                       \
                } else {                                                                                    \
                    _dp_e1 = (score_t*)(dp_h + dp_sn);                                                      \
                    if (_dp_e1[j] == _pre_dp_e1[j] - gap_ext1) {                                            \
                        _pre_dp_h = (score_t*)(DP_HEF + dp_sn * pre_i * 3);                                 \
                        if (_pre_dp_h[j] - gap_oe1 == _pre_dp_e1[j]) cur_op = ABPOA_M_OP;                   \
                        else cur_op = ABPOA_E1_OP;                                                          \
                        hit = 1;                                                                            \
                        cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);                \
                        i = pre_i; id = abpoa_graph_index_to_node_id(graph, i);                             \
                        dp_h = DP_HEF + dp_sn * i * 3; _dp_h = (score_t*)dp_h;                              \
                        break;                                                                              \
                    }                                                                                       \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        if (hit == 0 && cur_op & ABPOA_F_OP) { /* insertion */                                              \
            _dp_f1 = (score_t*)(dp_h + dp_sn * 2);                                                          \
            if (cur_op & ABPOA_M_OP) {                                                                      \
                if (_dp_h[j] == _dp_f1[j]) {                                                                \
                    if (_dp_h[j-1] - gap_oe1 == _dp_f1[j]) cur_op = ABPOA_M_OP, hit = 1;                    \
                    else if (_dp_f1[j-1] - gap_ext1 == _dp_f1[j]) cur_op = ABPOA_F1_OP, hit = 1;            \
                    else err_fatal_simple("Error in ag_backtrack. (1)");                                    \
                }                                                                                           \
            } else {                                                                                        \
                if (_dp_f1[j-1] - gap_ext1 == _dp_f1[j]) cur_op = ABPOA_F1_OP, hit = 1;                     \
                else if (_dp_h[j-1] - gap_oe1 == _dp_f1[j]) cur_op = ABPOA_M_OP, hit = 1;                   \
                else err_fatal_simple("Error in ag_backtrack. (2)");                                        \
            }                                                                                               \
            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1); --j;                       \
            ++res->n_aln_bases;                                                                             \
        }                                                                                                   \
        if (hit == 0) err_fatal_simple("Error in ag_backtrack. (3)");                                       \
    }                                                                                                       \
    if (j > 0) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, j, -1, j-1);                   \
    /* reverse cigar */                                                                                     \
    res->graph_cigar = abpt->rev_cigar ? cigar : abpoa_reverse_cigar(n_c, cigar);                           \
    res->n_cigar = n_c;                                                                                     \
    res->node_e = best_i, res->query_e = best_j-1; /* 0-based */                                            \
    res->node_s = _start_i, res->query_s = _start_j-1;                                                      \
    /*abpoa_print_cigar(n_c, *graph_cigar, graph);*/                                                        \
}

#define simd_abpoa_cg_backtrack(score_t) {                                                                  \
    int i, j, k, pre_i, n_c = 0, s, m_c = 0, id, hit, cur_op = ABPOA_ALL_OP, _start_i, _start_j;            \
    score_t *_dp_h, *_dp_e1, *_dp_e2, *_dp_f1, *_dp_f2, *_pre_dp_h, *_pre_dp_e1, *_pre_dp_e2;               \
    abpoa_cigar_t *cigar = 0;                                                                               \
    i = best_i, j = best_j, _start_i = best_i, _start_j = best_j;                                           \
    id = abpoa_graph_index_to_node_id(graph, i);                                                            \
    if (best_j < qlen) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, qlen-j, -1, qlen-1);   \
    SIMDi *dp_h = DP_H2E2F + dp_sn * i * 5; _dp_h = (score_t*)dp_h;                                         \
    while (i > 0 && j > 0) {                                                                                \
        if (abpt->align_mode == ABPOA_LOCAL_MODE && _dp_h[j] == 0) break;                                   \
        _start_i = i, _start_j = j;                                                                         \
        int *pre_index_i = pre_index[i];                                                                    \
        s = mat[m * graph->node[id].base + query[j-1]]; hit = 0;                                            \
        if (cur_op & ABPOA_M_OP) {                                                                          \
            for (k = 0; k < pre_n[i]; ++k) {                                                                \
                pre_i = pre_index_i[k];                                                                     \
                if (j-1 < dp_beg[pre_i] || j-1 > dp_end[pre_i]) continue;                                   \
                _pre_dp_h = (score_t*)(DP_H2E2F + dp_sn * pre_i * 5);                                       \
                if (_pre_dp_h[j-1] + s == _dp_h[j]) { /* match/mismatch */                                  \
                    cur_op = ABPOA_ALL_OP; hit = 1;                                                         \
                    cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CMATCH, 1, id, j-1);                  \
                    i = pre_i; --j; id = abpoa_graph_index_to_node_id(graph, i);                            \
                    dp_h = DP_H2E2F + dp_sn * i * 5; _dp_h = (score_t*)dp_h;                                \
                    ++res->n_aln_bases; res->n_matched_bases += s == mat[0] ? 1 : 0;                        \
                    break;                                                                                  \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        if (hit == 0 && cur_op & ABPOA_E_OP) {                                                              \
            for (k = 0; k < pre_n[i]; ++k) {                                                                \
                pre_i = pre_index_i[k];                                                                     \
                if (j < dp_beg[pre_i] || j > dp_end[pre_i]) continue;                                       \
                if (cur_op & ABPOA_E1_OP) {                                                                 \
                    _pre_dp_e1 = (score_t*)(DP_H2E2F + dp_sn * (pre_i * 5 + 1));                            \
                    if (cur_op & ABPOA_M_OP) {                                                              \
                        if (_dp_h[j] == _pre_dp_e1[j]) {                                                    \
                            _pre_dp_h = (score_t*)(DP_H2E2F + dp_sn * pre_i * 5);                           \
                            if (_pre_dp_h[j] - gap_oe1 == _pre_dp_e1[j]) cur_op = ABPOA_M_OP;               \
                            else cur_op = ABPOA_E1_OP;                                                      \
                            hit = 1; cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);   \
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i);                         \
                            dp_h = DP_H2E2F + dp_sn * i * 5; _dp_h = (score_t*)dp_h;                        \
                            break;                                                                          \
                        }                                                                                   \
                    } else {                                                                                \
                        _dp_e1 = (score_t*)(dp_h + dp_sn);                                                  \
                        if (_dp_e1[j] == _pre_dp_e1[j] - gap_ext1) {                                        \
                            _pre_dp_h = (score_t*)(DP_H2E2F + dp_sn * pre_i * 5);                           \
                            if (_pre_dp_h[j] - gap_oe1 == _pre_dp_e1[j]) cur_op = ABPOA_M_OP;               \
                            else cur_op = ABPOA_E1_OP;                                                      \
                            hit = 1; cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);   \
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i);                         \
                            dp_h = DP_H2E2F + dp_sn * i * 5; _dp_h = (score_t*)dp_h;                        \
                            break;                                                                          \
                        }                                                                                   \
                    }                                                                                       \
                }                                                                                           \
                if (cur_op & ABPOA_E2_OP) {                                                                 \
                    _pre_dp_e2 = (score_t*)(DP_H2E2F + dp_sn * (pre_i * 5 + 2));                            \
                    if (cur_op & ABPOA_M_OP) {                                                              \
                        if (_dp_h[j] == _pre_dp_e2[j]) {                                                    \
                            _pre_dp_h = (score_t*)(DP_H2E2F + dp_sn * pre_i * 5);                           \
                            if (_pre_dp_h[j] - gap_oe2 == _pre_dp_e2[j]) cur_op = ABPOA_M_OP;               \
                            else cur_op = ABPOA_E2_OP;                                                      \
                            hit = 1; cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);   \
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i);                         \
                            dp_h = DP_H2E2F + dp_sn * i * 5; _dp_h = (score_t*)dp_h;                        \
                            break;                                                                          \
                        }                                                                                   \
                    } else {                                                                                \
                        _dp_e2 = (score_t*)(dp_h + dp_sn * 2);                                              \
                        if (_dp_e2[j] == _pre_dp_e2[j] - gap_ext2) {                                        \
                            _pre_dp_h = (score_t*)(DP_H2E2F + dp_sn * pre_i * 5);                           \
                            if (_pre_dp_h[j] - gap_oe2 == _pre_dp_e2[j]) cur_op = ABPOA_M_OP;               \
                            else cur_op = ABPOA_E2_OP;                                                      \
                            hit = 1; cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);   \
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i);                         \
                            dp_h = DP_H2E2F + dp_sn * i * 5; _dp_h = (score_t*)dp_h;                        \
                            break;                                                                          \
                        }                                                                                   \
                    }                                                                                       \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        /* insertion */                                                                                     \
        if (hit == 0 && cur_op & ABPOA_F_OP) {                                                              \
            if (cur_op & ABPOA_F1_OP) {                                                                     \
                _dp_f1 = (score_t*)(dp_h + dp_sn * 3);                                                      \
                if (cur_op & ABPOA_M_OP) {                                                                  \
                    if (_dp_h[j] == _dp_f1[j]) {                                                            \
                        if (_dp_h[j-1] - gap_oe1 == _dp_f1[j]) cur_op = ABPOA_M_OP, hit = 1;                \
                        else if (_dp_f1[j-1] - gap_ext1 == _dp_f1[j]) cur_op = ABPOA_F1_OP, hit = 1;        \
                        else err_fatal_simple("Error in cg_backtrack. (1)");                                \
                    }                                                                                       \
                } else {                                                                                    \
                    if (_dp_f1[j-1] - gap_ext1 == _dp_f1[j]) cur_op = ABPOA_F1_OP, hit =1;                  \
                    else if (_dp_h[j-1] - gap_oe1 == _dp_f1[j]) cur_op = ABPOA_M_OP, hit = 1;               \
                    else err_fatal_simple("Error in cg_backtrack. (2)");                                    \
                }                                                                                           \
            }                                                                                               \
            if (hit == 0 && cur_op & ABPOA_F2_OP) {                                                         \
                _dp_f2 = (score_t*)(dp_h + dp_sn * 4);                                                      \
                if (cur_op & ABPOA_M_OP) {                                                                  \
                    if (_dp_h[j] == _dp_f2[j]) {                                                            \
                        if (_dp_h[j-1] - gap_oe2 == _dp_f2[j]) cur_op = ABPOA_M_OP, hit = 1;                \
                        else if (_dp_f2[j-1] - gap_ext2 == _dp_f2[j]) cur_op = ABPOA_F2_OP, hit =1;         \
                        else err_fatal_simple("Error in cg_backtrack. (3)");                                \
                    }                                                                                       \
                } else {                                                                                    \
                    if (_dp_f2[j-1] - gap_ext2 == _dp_f2[j]) cur_op = ABPOA_F2_OP, hit =1;                  \
                    else if (_dp_h[j-1] - gap_oe2 == _dp_f2[j]) cur_op = ABPOA_M_OP, hit = 1;               \
                    else err_fatal_simple("Error in cg_backtrack. (4)");                                    \
                }                                                                                           \
            }                                                                                               \
            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1); --j;                       \
            ++res->n_aln_bases;                                                                             \
            hit = 1;                                                                                        \
        }                                                                                                   \
        if (hit == 0) err_fatal_simple("Error in cg_backtrack. (5)");                                       \
     /* fprintf(stderr, "%d, %d, %d\n", i, j, cur_op); */                                                   \
    }                                                                                                       \
    if (j > 0) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, j, -1, j-1);                   \
    /* reverse cigar */                                                                                     \
    res->graph_cigar = abpt->rev_cigar ? cigar : abpoa_reverse_cigar(n_c, cigar);                           \
    res->n_cigar = n_c;                                                                                     \
    res->node_e = best_i, res->query_e = best_j-1; /* 0-based */                                            \
    res->node_s = _start_i, res->query_s = _start_j-1;                                                      \
    /*abpoa_print_cigar(n_c, *graph_cigar, graph);*/                                                        \
}

// simd_abpoa_va
// simd_abpoa_ag_only_var
// sim_abpoa_init_var
#define simd_abpoa_var(score_t, sp, SIMDSetOne, SIMDShiftOneN)                                      \
    /* int tot_dp_sn = 0; */                                                                        \
	/* int blocks_per_grid = qlen / threads_per_block;                                                  \ */\
	\
	\
    abpoa_graph_t *graph = ab->abg; abpoa_simd_matrix_t *abm = ab->abm;                             \
    int matrix_row_n = graph->node_n, matrix_col_n = qlen + 1;                                      \
    int **pre_index, *pre_n, pre_i;                                                                 \
    int i, j, k, *dp_beg, *dp_beg_sn, *dp_end, *dp_end_sn, node_id, index_i;                        \
    int beg, end, beg_sn, end_sn, _beg_sn, _end_sn, pre_beg_sn, pre_end, sn_i;                      \
    int pn, log_n, size, qp_sn, dp_sn; /* pn: value per SIMDi, qp_sn/dp_sn/d_sn: segmented length*/ \
    SIMDi *dp_h, *pre_dp_h, *qp, *qi=NULL;                                                          \
    score_t *_dp_h=NULL, *_qi, best_score = sp.inf_min, inf_min = sp.inf_min;                       \
    int *mat = abpt->mat, m = abpt->m; score_t gap_ext1 = abpt->gap_ext1;                           \
    int w = abpt->wb < 0 ? qlen : abpt->wb+(int)(abpt->wf*qlen); /* when w < 0, do whole global */  \
    int best_i = 0, best_j = 0, best_id = 0, max, max_i=-1;                                         \
    SIMDi zero = SIMDSetZeroi(), SIMD_INF_MIN = SIMDSetOne(inf_min);                                \
    pn = sp.num_of_value; qp_sn = dp_sn = (matrix_col_n + pn - 1) / pn;                             \
    log_n = sp.log_num, size = sp.size; qp = abm->s_mem;                                            \
    int set_num; SIMDi *PRE_MASK, *SUF_MIN, *PRE_MIN;                                               \
    PRE_MASK = (SIMDi*)SIMDMalloc((pn+1) * size, size);                                             \
    SUF_MIN = (SIMDi*)SIMDMalloc((pn+1) * size, size);                                              \
    PRE_MIN = (SIMDi*)SIMDMalloc(pn * size, size);                                                  \
    for (i = 0; i < pn; ++i) {                                                                      \
        score_t *pre_mask = (score_t*)(PRE_MASK+i);                                                 \
        for (j = 0; j <= i; ++j) pre_mask[j] = -1;                                                  \
        for (j = i+1; j < pn; ++j) pre_mask[j] = 0;                                                 \
    } PRE_MASK[pn] = PRE_MASK[pn-1];                                                                \
    SUF_MIN[0] = SIMDShiftLeft(SIMD_INF_MIN, SIMDShiftOneN);                                        \
    for (i = 1; i < pn; ++i)                                                                        \
        SUF_MIN[i] = SIMDShiftLeft(SUF_MIN[i-1], SIMDShiftOneN); SUF_MIN[pn] = SUF_MIN[pn-1];       \
    for (i = 1; i < pn; ++i) {                                                                      \
        score_t *pre_min = (score_t*)(PRE_MIN + i);                                                 \
        for (j = 0; j < i; ++j) pre_min[j] = inf_min;                                               \
        for (j = i; j < pn; ++j) pre_min[j] = 0;                                                    \
    }

#define simd_abpoa_lg_only_var(score_t, SIMDSetOne, SIMDAdd)                \
    SIMDi *DP_H = qp + qp_sn * abpt->m; qi = DP_H + dp_sn * matrix_row_n;   \
    SIMDi GAP_E1 = SIMDSetOne(gap_ext1);                                    \
    SIMDi *GAP_E1S =  (SIMDi*)SIMDMalloc(log_n * size, size);               \
    GAP_E1S[0] = GAP_E1;                                                    \
    for (i = 1; i < log_n; ++i) {                                           \
        GAP_E1S[i] = SIMDAdd(GAP_E1S[i-1], GAP_E1S[i-1]);                   \
    }

#define simd_abpoa_ag_only_var(score_t, SIMDSetOne, SIMDAdd)                                            \
    score_t *_dp_e1, *_dp_f1, gap_open1 = abpt->gap_open1, gap_oe1 = abpt->gap_open1 + abpt->gap_ext1;  \
    SIMDi *DP_HEF, *dp_e1, *pre_dp_e1, *dp_f1; int pre_end_sn;                                          \
    DP_HEF = qp + qp_sn * abpt->m; qi = DP_HEF + dp_sn * matrix_row_n * 3;                              \
    SIMDi GAP_O1 = SIMDSetOne(gap_open1), GAP_E1 = SIMDSetOne(gap_ext1), GAP_OE1 = SIMDSetOne(gap_oe1); \
    SIMDi *GAP_E1S =  (SIMDi*)SIMDMalloc(log_n * size, size);  GAP_E1S[0] = GAP_E1;                     \
    for (i = 1; i < log_n; ++i) {                                                                       \
        GAP_E1S[i] = SIMDAdd(GAP_E1S[i-1], GAP_E1S[i-1]);                                               \
    }

#define simd_abpoa_cg_only_var(score_t, SIMDSetOne, SIMDAdd)                                                        \
    score_t *_dp_e1, *_dp_e2, *_dp_f1, *_dp_f2, gap_open1 = abpt->gap_open1, gap_oe1 = gap_open1 + gap_ext1;        \
    score_t gap_open2 = abpt->gap_open2, gap_ext2 = abpt->gap_ext2, gap_oe2 = gap_open2 + gap_ext2;                 \
    SIMDi *DP_H2E2F, *dp_e1, *dp_e2, *dp_f1, *dp_f2, *pre_dp_e1, *pre_dp_e2; int pre_end_sn;                        \
    SIMDi GAP_O1 = SIMDSetOne(gap_open1), GAP_O2 = SIMDSetOne(gap_open2);                                           \
    SIMDi GAP_E1 = SIMDSetOne(gap_ext1), GAP_E2 = SIMDSetOne(gap_ext2);                                             \
    SIMDi GAP_OE1 = SIMDSetOne(gap_oe1), GAP_OE2 = SIMDSetOne(gap_oe2);                                             \
    DP_H2E2F = qp + qp_sn * abpt->m; qi = DP_H2E2F + dp_sn * matrix_row_n * 5;                                      \
    SIMDi *GAP_E1S =  (SIMDi*)SIMDMalloc(log_n * size, size), *GAP_E2S =  (SIMDi*)SIMDMalloc(log_n * size, size);   \
    GAP_E1S[0] = GAP_E1; GAP_E2S[0] = GAP_E2;                                                                       \
    for (i = 1; i < log_n; ++i) {                                                                                   \
        GAP_E1S[i] = SIMDAdd(GAP_E1S[i-1], GAP_E1S[i-1]);                                                           \
        GAP_E2S[i] = SIMDAdd(GAP_E2S[i-1], GAP_E2S[i-1]);                                                           \
    }

#define simd_abpoa_init_var(score_t) {                                                              \
    /* generate the query profile */                                                                \
    for (i = 0; i < qp_sn * abpt->m; ++i) qp[i] = SIMD_INF_MIN;                                     \
    for (k = 0; k < abpt->m; ++k) { /* SIMD parallelization */                                      \
        int *p = &mat[k * abpt->m];                                                                 \
        score_t *_qp = (score_t*)(qp + k * qp_sn); _qp[0] = 0;                                      \
        for (j = 0; j < qlen; ++j) _qp[j+1] = (score_t)p[query[j]];                                 \
        for (j = qlen+1; j < qp_sn * pn; ++j) _qp[j] = 0;                                           \
    }                                                                                               \
    if (abpt->wb>=0 || abpt->align_mode==ABPOA_LOCAL_MODE || abpt->align_mode==ABPOA_EXTEND_MODE){  \
        _qi = (score_t*)qi; /* query index */                                                       \
        for (i = 0; i <= qlen; ++i) _qi[i] = i;                                                     \
        for (i = qlen+1; i < (qlen/pn+1) * pn; ++i) _qi[i] = -1;                                    \
    }                                                                                               \
    /* for backtrack */                                                                             \
    dp_beg=abm->dp_beg, dp_end=abm->dp_end, dp_beg_sn=abm->dp_beg_sn, dp_end_sn=abm->dp_end_sn;     \
    /* index of pre-node */                                                                         \
    pre_index = (int**)_err_malloc(graph->node_n * sizeof(int*));                                   \
    pre_n = (int*)_err_malloc(graph->node_n * sizeof(int));                                         \
    for (i = 0; i < graph->node_n; ++i) {                                                           \
        node_id = abpoa_graph_index_to_node_id(graph, i); /* i: node index */                       \
        pre_n[i] = graph->node[node_id].in_edge_n;                                                  \
        pre_index[i] = (int*)_err_malloc(pre_n[i] * sizeof(int));                                   \
        for (j = 0; j < pre_n[i]; ++j) {                                                            \
            pre_index[i][j] = abpoa_graph_node_id_to_index(graph, graph->node[node_id].in_id[j]);   \
        }                                                                                           \
    }                                                                                               \
}

#define simd_abpoa_free_var {                                                               \
    for (i = 0; i < graph->node_n; ++i) free(pre_index[i]); free(pre_index); free(pre_n);	\
    SIMDFree(PRE_MASK); SIMDFree(SUF_MIN); SIMDFree(PRE_MIN);                               \
}	                                                                                        \

#define simd_abpoa_lg_var(score_t, sp, SIMDSetOne, SIMDShiftOneN, SIMDAdd)  \
    simd_abpoa_var(score_t, sp, SIMDSetOne, SIMDShiftOneN);                 \
    simd_abpoa_lg_only_var(score_t, SIMDSetOne, SIMDAdd);                   \
    simd_abpoa_init_var(score_t);

#define simd_abpoa_ag_var(score_t, sp, SIMDSetOne, SIMDShiftOneN, SIMDAdd)  \
    simd_abpoa_var(score_t, sp, SIMDSetOne, SIMDShiftOneN);                 \
    simd_abpoa_ag_only_var(score_t, SIMDSetOne, SIMDAdd);                   \
    simd_abpoa_init_var(score_t);

#define simd_abpoa_cg_var(score_t, sp, SIMDSetOne, SIMDShiftOneN, SIMDAdd)  \
    simd_abpoa_var(score_t, sp, SIMDSetOne, SIMDShiftOneN);                 \
    simd_abpoa_cg_only_var(score_t, SIMDSetOne, SIMDAdd);                   \
    simd_abpoa_init_var(score_t);

#define simd_abpoa_lg_first_row {                                                                   \
    /* fill the first row */	                                                                    \
    if (abpt->wb >= 0) {                                                                            \
        graph->node_id_to_max_pos_left[0] = graph->node_id_to_max_pos_right[0] = 0;	                \
        for (i = 0; i < graph->node[0].out_edge_n; ++i) { /* set min/max rank for next_id */	    \
            int out_id = graph->node[0].out_id[i];	                                                \
            graph->node_id_to_max_pos_left[out_id] = graph->node_id_to_max_pos_right[out_id] = 1;	\
        }                                                                                           \
		fprintf(stderr,"graph_out_nodes_num=%d\n",graph->node[0].out_edge_n);                                       \
		fprintf(stderr,"w=%d\n",w);                                       \
		fprintf(stderr,"pn=%d\n",pn);                                       \
        dp_beg[0] = 0, dp_end[0] = GET_AD_DP_END(graph, w, 0, qlen) / pn;                           \
    } else {                                                                                        \
        dp_beg[0] = 0, dp_end[0] = qlen;	                                                        \
    }                                                                                               \
    dp_beg_sn[0] = (dp_beg[0])/pn; dp_end_sn[0] = (dp_end[0])/pn;                                   \
    dp_beg[0] = dp_beg_sn[0] * pn; dp_end[0] = (dp_end_sn[0]+1)*pn-1;                               \
    dp_h = DP_H;_end_sn = MIN_OF_TWO(dp_end_sn[0]+1, dp_sn-1);	                                    \
}

#define simd_abpoa_ag_first_row {                                                                   \
    /* fill the first row */                                                                        \
    if (abpt->wb >= 0) {                                                                            \
        graph->node_id_to_max_pos_left[0] = graph->node_id_to_max_pos_right[0] = 0;                 \
        for (i = 0; i < graph->node[0].out_edge_n; ++i) { /* set min/max rank for next_id */        \
            int out_id = graph->node[0].out_id[i];                                                  \
            graph->node_id_to_max_pos_left[out_id] = graph->node_id_to_max_pos_right[out_id] = 1;   \
        }                                                                                           \
        dp_beg[0] = 0, dp_end[0] = GET_AD_DP_END(graph, w, 0, qlen) / pn;                           \
    } else {                                                                                        \
        dp_beg[0] = 0, dp_end[0] = qlen;                                                            \
    }                                                                                               \
    dp_beg_sn[0] = (dp_beg[0])/pn; dp_end_sn[0] = (dp_end[0])/pn;                                   \
    dp_beg[0] = dp_beg_sn[0] * pn; dp_end[0] = (dp_end_sn[0]+1)*pn-1;                               \
    dp_h = DP_HEF; dp_e1 = dp_h + dp_sn; dp_f1 = dp_e1 + dp_sn;                                     \
    _end_sn = MIN_OF_TWO(dp_end_sn[0]+1, dp_sn-1);                                                  \
}

#define simd_abpoa_cg_first_row {                                                                   \
    /* fill the first row */                                                                        \
    if (abpt->wb >= 0) {                                                                            \
        graph->node_id_to_max_pos_left[0] = graph->node_id_to_max_pos_right[0] = 0;                 \
        for (i = 0; i < graph->node[0].out_edge_n; ++i) { /* set min/max rank for next_id */        \
            int out_id = graph->node[0].out_id[i];                                                  \
            graph->node_id_to_max_pos_left[out_id] = graph->node_id_to_max_pos_right[out_id] = 1;   \
        }                                                                                           \
        dp_beg[0] = 0, dp_end[0] = GET_AD_DP_END(graph, w, 0, qlen) / pn;                           \
    } else {                                                                                        \
        dp_beg[0] = 0, dp_end[0] = qlen;                                                            \
    }                                                                                               \
    dp_beg_sn[0] = (dp_beg[0])/pn; dp_end_sn[0] = (dp_end[0])/pn;                                   \
    dp_beg[0] = dp_beg_sn[0] * pn; dp_end[0] = (dp_end_sn[0]+1)*pn-1;                               \
    dp_h=DP_H2E2F; dp_e1=dp_h+dp_sn; dp_e2=dp_e1+dp_sn; dp_f1=dp_e2+dp_sn; dp_f2=dp_f1+dp_sn;       \
    _end_sn = MIN_OF_TWO(dp_end_sn[0]+1, dp_sn-1);                                                  \
}

#define simd_abpoa_lg_first_dp(score_t) {                                   \
    simd_abpoa_lg_first_row;                                                \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) {                             \
        for (i = 0; i < _end_sn; ++i)                                       \
            dp_h[i] = zero;                                                 \
    } else {                                                                \
        _dp_h = (score_t*)dp_h;	                                            \
        for (i = dp_end[0]/pn; i <= _end_sn; ++i) {	                        \
            dp_h[i] = SIMD_INF_MIN;	                                        \
        }	                                                                \
        for (i = 0; i <= dp_end[0]; ++i) { /* no SIMD parallelization */	\
            _dp_h[i] = -gap_ext1 * i;	                                    \
        }	                                                                \
    }                                                                       \
}

#define simd_abpoa_ag_first_dp(score_t) {                                   \
    simd_abpoa_ag_first_row;                                                \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) {                             \
        for (i = 0; i < _end_sn; ++i) {                                     \
            dp_h[i] = zero; dp_e1[i] = zero, dp_f1[i] = zero;               \
        }                                                                   \
    } else {                                                                \
        _dp_h=(score_t*)dp_h,_dp_e1=(score_t*)dp_e1,_dp_f1=(score_t*)dp_f1; \
        for (i = dp_end[0]/pn; i <= _end_sn; ++i) {                         \
            dp_h[i] = SIMD_INF_MIN; dp_e1[i] = SIMD_INF_MIN;                \
        }                                                                   \
        _dp_h[0] = 0; _dp_e1[0] = -(gap_oe1), _dp_f1[0] = inf_min;          \
        for (i = 1; i <= dp_end[0]; ++i) { /* no SIMD parallelization */    \
            _dp_f1[i] = -gap_open1 - gap_ext1 * i;                          \
            _dp_h[i] = -gap_open1 - gap_ext1 * i;                           \
        }                                                                   \
    }                                                                       \
}

#define simd_abpoa_cg_first_dp(score_t) {                                               \
    simd_abpoa_cg_first_row;                                                            \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                         \
        dp_h[i] = dp_e1[i] = dp_e2[i] = dp_f1[i] = dp_f2[i] = zero;                     \
    } else {                                                                            \
        _dp_h = (score_t*)dp_h, _dp_e1 = (score_t*)dp_e1, _dp_e2 = (score_t*)dp_e2;     \
        _dp_f1 = (score_t*)dp_f1, _dp_f2 = (score_t*)dp_f2;                             \
        for (i = 0; i <= _end_sn; ++i) {                                                \
            dp_h[i] = SIMD_INF_MIN; dp_e1[i] = SIMD_INF_MIN; dp_e2[i] = SIMD_INF_MIN;   \
        }                                                                               \
        _dp_h[0] = 0; _dp_e1[0] = -(gap_oe1); _dp_e2[0] = -(gap_oe2);                   \
        _dp_f1[0] = _dp_f2[0] = inf_min;                                                \
        for (i = 1; i <= dp_end[0]; ++i) { /* no SIMD parallelization */                \
            _dp_f1[i] = -gap_open1 - gap_ext1 * i;                                      \
            _dp_f2[i] = -gap_open2 - gap_ext2 * i;                                      \
            _dp_h[i] = MAX_OF_TWO(_dp_f1[i], _dp_f2[i]);                                \
        }                                                                               \
    }                                                                                   \
}

// mask[pn], suf_min[pn], pre_min[logN]
#define SIMD_SET_F(F, log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E1S, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN) {  \
    if (set_num == pn) {                                                                                                \
		SIMDi temp =SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[0]), SIMDShiftOneN), PRE_MIN[1]); \
        F = SIMDMax(F, SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[0]), SIMDShiftOneN), PRE_MIN[1]));                      \
        if (log_n > 1) {                                                                                                \
            F = SIMDMax(F, SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[1]), SIMDShiftOneN<<1), PRE_MIN[2]));               \
        } if (log_n > 2) {                                                                                              \
            F = SIMDMax(F, SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[2]), SIMDShiftOneN<<2), PRE_MIN[4]));               \
        } if (log_n > 3) {                                                                                              \
            F = SIMDMax(F, SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[3]), SIMDShiftOneN<<3), PRE_MIN[8]));               \
        } if (log_n > 4) {                                                                                              \
            F = SIMDMax(F, SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[4]), SIMDShiftOneN<<4), PRE_MIN[16]));              \
        } if (log_n > 5) {                                                                                              \
            F = SIMDMax(F, SIMDOri(SIMDShiftLeft(SIMDSub(F, GAP_E1S[5]), SIMDShiftOneN<<5), PRE_MIN[32]));              \
        }                                                                                                               \
    } else { /*suffix MIN_INF*/                                                                                                                                     \
        int cov_bit = set_num;                                                                                                                                      \
		/* if (set_num == 1){\ */\
		/*     print_simd(&F, "init", int16_t);\ */\
		/* }\ */\
        F = SIMDMax(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSub(F, GAP_E1S[0]), SIMDShiftOneN), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[1])));          \
		/* if (set_num == 1){\ */\
		/*     print_simd(&F, "", int16_t);\ */\
		/* }\ */\
        if (log_n > 1) {                                                                                                                                            \
            cov_bit += 2;                                                                                                                                           \
            F = SIMDMax(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSub(F, GAP_E1S[1]), SIMDShiftOneN<<1), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[2])));   \
			/* if (set_num == 1){\ */\
			/*     print_simd(&F, "1", int16_t);\ */\
			/* }\ */\
        } if (log_n > 2) {                                                                                                                                          \
            cov_bit += 4;                                                                                                                                           \
            F = SIMDMax(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSub(F, GAP_E1S[2]), SIMDShiftOneN<<2), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[4])));   \
			/* if (set_num == 1){\ */\
			/*     print_simd(&F, "2", int16_t);\ */\
			/* }\ */\
        } if (log_n > 3) {                                                                                                                                          \
            cov_bit += 8;                                                                                                                                           \
            F = SIMDMax(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSub(F, GAP_E1S[3]), SIMDShiftOneN<<3), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[8])));   \
        } if (log_n > 4) {                                                                                                                                          \
            cov_bit += 16;                                                                                                                                          \
            F = SIMDMax(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSub(F, GAP_E1S[4]), SIMDShiftOneN<<4), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[16])));  \
        } if (log_n > 5) {                                                                                                                                          \
            cov_bit += 32;                                                                                                                                          \
            F = SIMDMax(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSub(F, GAP_E1S[5]), SIMDShiftOneN<<5), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[32])));  \
        }                                                                                                                                                           \
    }                                                                                                                                                               \
}

#define simd_abpoa_lg_dp(score_t, SIMDShiftOneN, SIMDMax, SIMDAdd, SIMDSub) {                                   \
    node_id = abpoa_graph_index_to_node_id(graph, index_i);	                                                    \
    SIMDi *q = qp + graph->node[node_id].base * qp_sn, first, remain;	                                        \
    dp_h = &DP_H[index_i * dp_sn]; _dp_h = (score_t*)dp_h;	                                                    \
    int min_pre_beg_sn, max_pre_end_sn;                                                                         \
    if (abpt->wb < 0) {                                                                                         \
        beg = dp_beg[index_i] = 0, end = dp_end[index_i] = qlen;	                                            \
        beg_sn = dp_beg_sn[index_i] = (dp_beg[index_i])/pn; end_sn = dp_end_sn[index_i] = (dp_end[index_i])/pn; \
        min_pre_beg_sn = 0, max_pre_end_sn = end_sn;                                                            \
    } else {                                                                                                    \
        beg = GET_AD_DP_BEGIN(graph, w, node_id, qlen), end = GET_AD_DP_END(graph, w, node_id, qlen);           \
		int max_right = abpoa_graph_node_id_to_max_pos_right(graph, index_i);\
		fprintf(stderr,"index=%d\n",index_i);                                       \
		fprintf(stderr,"max_right=%d\n",max_right);                                       \
		fprintf(stderr,"qlen=%d\n",qlen);                                       \
		fprintf(stderr,"beg=%d\n",beg);                                       \
		fprintf(stderr,"end=%d\n",end);                                       \
        beg_sn = beg / pn; min_pre_beg_sn = INT32_MAX, max_pre_end_sn = -1;                                     \
        for (i = 0; i < pre_n[index_i]; ++i) {                                                                  \
            pre_i = pre_index[index_i][i];                                                                      \
            if (min_pre_beg_sn > dp_beg_sn[pre_i]) min_pre_beg_sn = dp_beg_sn[pre_i];                           \
            if (max_pre_end_sn < dp_end_sn[pre_i]) max_pre_end_sn = dp_end_sn[pre_i];                           \
        } if (beg_sn < min_pre_beg_sn) beg_sn = min_pre_beg_sn;                                                 \
		fprintf(stderr,"max_pre_end=%d\n",max_pre_end_sn);                                       \
        dp_beg_sn[index_i] = beg_sn; beg = dp_beg[index_i] = dp_beg_sn[index_i] * pn;                           \
        end_sn = dp_end_sn[index_i] = end/pn; end = dp_end[index_i] = (dp_end_sn[index_i]+1)*pn-1;              \
    }                                                                                                           \
    /* loop query */	                                                                                                                \
    /* first pre_node */                                                                                                                \
    pre_i = pre_index[index_i][0];	                                                                                                    \
    pre_dp_h = DP_H + pre_i * dp_sn;	                                                                                                \
    pre_end = dp_end[pre_i];	                                                                                                        \
    pre_beg_sn = dp_beg_sn[pre_i];	                                                                                                    \
    /* set M from (pre_i, q_i-1), E from (pre_i, q_i) */	                                                                            \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                                         \
        _beg_sn = 0, _end_sn = end_sn; first = SIMDShiftRight(zero, SIMDTotalBytes-SIMDShiftOneN);                                      \
    } else {                                                                                                                            \
        if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneN);            \
        else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneN);	                                \
        _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);	                                                                    \
        for (i = beg_sn; i < _beg_sn; ++i) dp_h[i] = SIMD_INF_MIN;                                                                      \
        for (i = _end_sn+1; i <= MIN_OF_TWO(end_sn+1, dp_sn-1); ++i) dp_h[i] = SIMD_INF_MIN;                                            \
    }                                                                                                                                   \
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */	                                                        \
        remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneN);	                                                                        \
        dp_h[sn_i] = SIMDMax(SIMDAdd(SIMDOri(first, remain), q[sn_i]), SIMDSub(pre_dp_h[sn_i], GAP_E1));	                            \
        first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneN);	                                                        \
    }                                                                                                                                   \
    /* get max m and e */	                                                                                                            \
    for (i = 1; i < pre_n[index_i]; ++i) {	                                                                                            \
        pre_i = pre_index[index_i][i];	                                                                                                \
        pre_dp_h = DP_H + pre_i * dp_sn;	                                                                                            \
        pre_end = dp_end[pre_i];	                                                                                                    \
        pre_beg_sn = dp_beg_sn[pre_i];	                                                                                                \
        /* set M from (pre_i, q_i-1), E from (pre_i, q_i) */	                                                                        \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                                     \
            first = SIMDShiftRight(zero, SIMDTotalBytes-SIMDShiftOneN);                                                                 \
        } else {                                                                                                                        \
            if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneN);        \
            else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneN);	                            \
            _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);	                                                                \
        }                                                                                                                               \
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */	                                                    \
            remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneN);	                                                                    \
            dp_h[sn_i] = SIMDMax(SIMDAdd(SIMDOri(first, remain), q[sn_i]), SIMDMax(SIMDSub(pre_dp_h[sn_i], GAP_E1), dp_h[sn_i]));	    \
            first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneN);	                                                    \
        } /* now we have max(h,e) stored at dp_h */	                                                                                    \
    }	                                                                                                                                \
    /* new F start */                                                                                                                   \
    first = SIMDOri(SIMDAndi(dp_h[beg_sn], PRE_MASK[0]), SUF_MIN[0]);                                                                   \
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) {                                                                                       \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                                     \
            set_num = pn;                                                                                                               \
        } else {                                                                                                                        \
            if (sn_i < min_pre_beg_sn) {                                                                                                \
                _err_fatal_simple(__func__, "sn_i < min_pre_beg_sn\n");                                                                 \
            } else if (sn_i > max_pre_end_sn) {                                                                                         \
                set_num = sn_i == max_pre_end_sn+1 ? 1 : 0;                                                                             \
            } else set_num = pn;                                                                                                        \
        }                                                                                                                               \
        dp_h[sn_i] = SIMDMax(dp_h[sn_i], first);                                                                                        \
        SIMD_SET_F(dp_h[sn_i], log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E1S, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN);          \
        first = SIMDOri(SIMDAndi(SIMDShiftRight(SIMDSub(dp_h[sn_i], GAP_E1), SIMDTotalBytes-SIMDShiftOneN), PRE_MASK[0]), SUF_MIN[0]);  \
		SIMDi temp = SIMDSub(dp_h[sn_i], GAP_E1);\
		print_simd(&temp, "first_compute_sub", int16_t);\
    }                                                                                                                                   \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) for (sn_i = 0; sn_i <= end_sn; ++sn_i) dp_h[sn_i] = SIMDMax(zero, dp_h[sn_i]);            \
}

#define simd_abpoa_ag_dp(score_t, SIMDShiftOneN, SIMDMax, SIMDAdd, SIMDSub, SIMDGetIfGreater, SIMDSetIfGreater, SIMDSetIfEqual) {\
    node_id = abpoa_graph_index_to_node_id(graph, index_i);                                                                     \
    SIMDi *q = qp + graph->node[node_id].base * qp_sn, first, remain;                                                           \
    dp_h = DP_HEF + index_i * 3 * dp_sn; dp_e1 = dp_h + dp_sn; dp_f1 = dp_e1 + dp_sn;                                           \
    _dp_h = (score_t*)dp_h, _dp_e1 = (score_t*)dp_e1, _dp_f1 = (score_t*)dp_f1;                                                 \
    int min_pre_beg_sn, max_pre_end_sn;                                                                                         \
    if (abpt->wb < 0) {                                                                                                         \
        beg = dp_beg[index_i] = 0, end = dp_end[index_i] = qlen;                                                                \
        beg_sn = dp_beg_sn[index_i] = (dp_beg[index_i])/pn; end_sn = dp_end_sn[index_i] = (dp_end[index_i])/pn;                 \
        min_pre_beg_sn = 0, max_pre_end_sn = end_sn;                                                                            \
    } else {                                                                                                                    \
        beg = GET_AD_DP_BEGIN(graph, w, node_id, qlen), end = GET_AD_DP_END(graph, w, node_id, qlen);                           \
        beg_sn = beg / pn; min_pre_beg_sn = INT32_MAX, max_pre_end_sn = -1;                                                     \
        for (i = 0; i < pre_n[index_i]; ++i) {                                                                                  \
            pre_i = pre_index[index_i][i];                                                                                      \
            if (min_pre_beg_sn > dp_beg_sn[pre_i]) min_pre_beg_sn = dp_beg_sn[pre_i];                                           \
            if (max_pre_end_sn < dp_end_sn[pre_i]) max_pre_end_sn = dp_end_sn[pre_i];                                           \
        } if (beg_sn < min_pre_beg_sn) beg_sn = min_pre_beg_sn;                                                                 \
        dp_beg_sn[index_i] = beg_sn; beg = dp_beg[index_i] = dp_beg_sn[index_i] * pn;                                           \
        end_sn = dp_end_sn[index_i] = end/pn; end = dp_end[index_i] = (dp_end_sn[index_i]+1)*pn-1;                              \
    }                                                                                                                           \
    /* loop query */                                                                                                            \
    /* first pre_node */                                                                                                        \
    pre_i = pre_index[index_i][0];                                                                                              \
    pre_dp_h = DP_HEF + pre_i * 3 * dp_sn; pre_dp_e1 = pre_dp_h + dp_sn;                                                        \
    pre_end = dp_end[pre_i]; pre_beg_sn = dp_beg_sn[pre_i]; pre_end_sn = dp_end_sn[pre_i];                                      \
    /* set M from (pre_i, q_i-1) */                                                                                             \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                                 \
        _beg_sn = 0, _end_sn = end_sn; first = SIMDShiftRight(zero, SIMDTotalBytes-SIMDShiftOneN);                              \
    } else {                                                                                                                    \
        if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneN);    \
        else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneN);	                        \
        _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);                                                                \
        for (i = beg_sn; i < _beg_sn; ++i) dp_h[i] = SIMD_INF_MIN;                                                              \
        for (i = _end_sn+1; i <= MIN_OF_TWO(end_sn+1, dp_sn-1); ++i) dp_h[i] = SIMD_INF_MIN;                                    \
    }                                                                                                                           \
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */                                                  \
        remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneN);                                                                  \
        dp_h[sn_i] = SIMDOri(first, remain);                                                                                    \
        first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneN);                                                   \
    }                                                                                                                           \
    /* set E from (pre_i, q_i) */                                                                                               \
    if (abpt->align_mode != ABPOA_LOCAL_MODE) {                                                                                 \
        _end_sn = MIN_OF_TWO(pre_end_sn, end_sn);                                                                               \
        for (i = beg_sn; i < _beg_sn; ++i) dp_e1[i] = SIMD_INF_MIN;                                                             \
        for (i = _end_sn+1; i <= end_sn; ++i) dp_e1[i] = SIMD_INF_MIN;                                                          \
    }                                                                                                                           \
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i)   /* SIMD parallelization */                                                  \
        dp_e1[sn_i] = pre_dp_e1[sn_i];                                                                                          \
    /* get max m and e */                                                                                                       \
    for (i = 1; i < pre_n[index_i]; ++i) {                                                                                      \
        pre_i = pre_index[index_i][i];                                                                                          \
        pre_dp_h = DP_HEF + pre_i * 3 * dp_sn; pre_dp_e1 = pre_dp_h + dp_sn;                                                    \
        pre_end = dp_end[pre_i]; pre_beg_sn = dp_beg_sn[pre_i]; pre_end_sn = dp_end_sn[pre_i];                                  \
        /* set M from (pre_i, q_i-1) */                                                                                         \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                             \
            first = SIMDShiftRight(zero, SIMDTotalBytes-SIMDShiftOneN);                                                         \
        } else {                                                                                                                \
            if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneN);\
            else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneN);	                    \
            _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);                                                            \
        }                                                                                                                       \
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */                                              \
            remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneN);                                                              \
            dp_h[sn_i] = SIMDMax(SIMDOri(first, remain), dp_h[sn_i]);                                                           \
            first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneN);                                               \
        }                                                                                                                       \
        /* set E from (pre_i, q_i) */                                                                                           \
        _end_sn = MIN_OF_TWO(pre_end_sn, end_sn);                                                                               \
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i)   /* SIMD parallelization */                                              \
            dp_e1[sn_i] = SIMDMax(pre_dp_e1[sn_i], dp_e1[sn_i]);                                                                \
    }                                                                                                                           \
    /* compare M, E, and F */                                                                                                   \
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) { /* SIMD parallelization */                                                    \
        dp_h[sn_i] = SIMDAdd(dp_h[sn_i], q[sn_i]);                                                                              \
    }                                                                                                                           \
    /* new F start */                                                                                                           \
    first = SIMDShiftRight(SIMDShiftLeft(dp_h[beg_sn], SIMDTotalBytes-SIMDShiftOneN), SIMDTotalBytes-SIMDShiftOneN);            \
	print_simd(&first, "first_first", int16_t);\
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) {                                                                               \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                             \
            set_num  = pn;                                                                                                      \
        } else {                                                                                                                \
            if (sn_i < min_pre_beg_sn) {                                                                                        \
                _err_fatal_simple(__func__, "sn_i < min_pre_beg_sn\n");                                                         \
            } else if (sn_i > max_pre_end_sn) {                                                                                 \
                set_num = sn_i == max_pre_end_sn+1 ? 2 : 1;                                                                     \
            } else set_num = pn;                                                                                                \
        }                                                                                                                       \
        /* F = (H << 1 | x) - OE */                                                                                             \
		print_simd(dp_h + sn_i, "no_or_first_F", int16_t);\
        dp_f1[sn_i] = SIMDSub(SIMDOri(SIMDShiftLeft(dp_h[sn_i], SIMDShiftOneN), first), GAP_OE1);                               \
		print_simd(dp_f1 + sn_i, "aoaoao_origin_F", int16_t);\
        /* F = max{F, (F-e)<<1}, F = max{F, (F-2e)<<2} ... */                                                                   \
        SIMD_SET_F(dp_f1[sn_i], log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E1S, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN); \
        /* x = max{H, F+o} */                                                                                                   \
        first = SIMDShiftRight(SIMDMax(dp_h[sn_i], SIMDAdd(dp_f1[sn_i], GAP_O1)), SIMDTotalBytes-SIMDShiftOneN);                \
        /* H = max{H, F} */                                                                                                     \
        dp_h[sn_i] = SIMDMax(dp_h[sn_i], dp_e1[sn_i]); SIMDi tmp = dp_h[sn_i];                                                  \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                             \
            dp_h[sn_i] = SIMDMax(zero, SIMDMax(dp_h[sn_i], dp_f1[sn_i]));                                                       \
            SIMDSetIfEqual(dp_e1[sn_i], dp_h[sn_i],tmp, SIMDMax(SIMDSub(dp_e1[sn_i],GAP_E1), SIMDSub(dp_h[sn_i],GAP_OE1)),zero);\
        } else {                                                                                                                \
            dp_h[sn_i] = SIMDMax(dp_h[sn_i], dp_f1[sn_i]);                                                                      \
            SIMDSetIfEqual(dp_e1[sn_i], dp_h[sn_i],tmp, SIMDMax(SIMDSub(dp_e1[sn_i],GAP_E1), SIMDSub(dp_h[sn_i],GAP_OE1)),SIMD_INF_MIN); \
        }                                                                                                                       \
    }                                                                                                                           \
}

#define simd_abpoa_cg_dp(score_t, SIMDShiftOneN, SIMDMax, SIMDAdd, SIMDSub, SIMDGetIfGreater, SIMDSetIfGreater, SIMDSetIfEqual) {\
    node_id = abpoa_graph_index_to_node_id(graph, index_i);                                                                     \
    SIMDi *q = qp + graph->node[node_id].base * qp_sn, first, remain;                                                           \
    dp_h = DP_H2E2F+index_i*5*dp_sn; dp_e1 = dp_h+dp_sn; dp_e2 = dp_e1+dp_sn; dp_f1 = dp_e2+dp_sn; dp_f2 = dp_f1+dp_sn;         \
    _dp_h=(score_t*)dp_h, _dp_e1=(score_t*)dp_e1, _dp_e2=(score_t*)dp_e2, _dp_f1=(score_t*)dp_f1, _dp_f2=(score_t*)dp_f2;       \
    int min_pre_beg_sn, max_pre_end_sn;                                                                                         \
    if (abpt->wb < 0) {                                                                                                         \
        beg = dp_beg[index_i] = 0, end = dp_end[index_i] = qlen;                                                                \
        beg_sn = dp_beg_sn[index_i] = beg/pn; end_sn = dp_end_sn[index_i] = end/pn;                                             \
        min_pre_beg_sn = 0, max_pre_end_sn = end_sn;                                                                            \
    } else {                                                                                                                    \
        beg = GET_AD_DP_BEGIN(graph, w, node_id, qlen), end = GET_AD_DP_END(graph, w, node_id, qlen);                           \
        beg_sn = beg / pn; min_pre_beg_sn = INT32_MAX, max_pre_end_sn = -1;                                                     \
        for (i = 0; i < pre_n[index_i]; ++i) {                                                                                  \
            pre_i = pre_index[index_i][i];                                                                                      \
            if (min_pre_beg_sn > dp_beg_sn[pre_i]) min_pre_beg_sn = dp_beg_sn[pre_i];                                           \
            if (max_pre_end_sn < dp_end_sn[pre_i]) max_pre_end_sn = dp_end_sn[pre_i];                                           \
        } if (beg_sn < min_pre_beg_sn) beg_sn = min_pre_beg_sn;                                                                 \
        dp_beg_sn[index_i] = beg_sn; beg = dp_beg[index_i] = dp_beg_sn[index_i] * pn;                                           \
        end_sn = dp_end_sn[index_i] = end/pn; end = dp_end[index_i] = (dp_end_sn[index_i]+1)*pn-1;                              \
    }                                                                                                                           \
 /* fprintf(stderr, "index_i: %d, beg_sn: %d, end_sn: %d\n", index_i, beg_sn, end_sn); */                                       \
    /* tot_dp_sn += (end_sn - beg_sn + 1); */                                                                                   \
    /* loop query */                                                                                                            \
    /* first pre_node */                                                                                                        \
    pre_i = pre_index[index_i][0];                                                                                              \
    pre_dp_h = DP_H2E2F + pre_i * 5 * dp_sn; pre_dp_e1 = pre_dp_h + dp_sn; pre_dp_e2 = pre_dp_e1 + dp_sn;                       \
    pre_end = dp_end[pre_i]; pre_beg_sn = dp_beg_sn[pre_i]; pre_end_sn = dp_end_sn[pre_i];                                      \
    /* set M from (pre_i, q_i-1) */                                                                                             \
    if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                                 \
        _beg_sn = 0, _end_sn = end_sn; first = SIMDShiftRight(zero, SIMDTotalBytes-SIMDShiftOneN);                              \
    } else {                                                                                                                    \
        if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneN);    \
        else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneN);                          \
        _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);                                                                \
        for (i = beg_sn; i < _beg_sn; ++i) dp_h[i] = SIMD_INF_MIN;                                                              \
        for (i = _end_sn+1; i <= MIN_OF_TWO(end_sn+1, dp_sn-1); ++i) dp_h[i] = SIMD_INF_MIN;                                    \
    }                                                                                                                           \
 /* fprintf(stderr, "1 index_i: %d, beg_sn: %d, end_sn: %d\n", index_i, _beg_sn, _end_sn); */                                   \
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */                                                  \
        remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneN);                                                                  \
        dp_h[sn_i] = SIMDOri(first, remain);                                                                                    \
        first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneN);                                                   \
    }                                                                                                                           \
    /* set E from (pre_i, q_i) */                                                                                               \
    if (abpt->align_mode != ABPOA_LOCAL_MODE) {                                                                                 \
        _end_sn = MIN_OF_TWO(pre_end_sn, end_sn);                                                                               \
        for (i = beg_sn; i < _beg_sn; ++i) dp_e1[i] = SIMD_INF_MIN, dp_e2[i] = SIMD_INF_MIN;                                    \
        for (i = _end_sn+1; i <= end_sn; ++i) dp_e1[i] = SIMD_INF_MIN, dp_e2[i] = SIMD_INF_MIN;                                 \
    }                                                                                                                           \
 /* fprintf(stderr, "2 index_i: %d, beg_sn: %d, end_sn: %d\n", index_i, _beg_sn, _end_sn); */                                   \
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */                                                  \
        dp_e1[sn_i] = pre_dp_e1[sn_i];                                                                                          \
        dp_e2[sn_i] = pre_dp_e2[sn_i];                                                                                          \
    }                                                                                                                           \
    /* get max m and e */                                                                                                       \
    for (i = 1; i < pre_n[index_i]; ++i) {                                                                                      \
        pre_i = pre_index[index_i][i];                                                                                          \
        pre_dp_h = DP_H2E2F + (pre_i * 5) * dp_sn; pre_dp_e1 = pre_dp_h + dp_sn; pre_dp_e2 = pre_dp_e1 + dp_sn;                 \
        pre_end = dp_end[pre_i]; pre_beg_sn = dp_beg_sn[pre_i]; pre_end_sn = dp_end_sn[pre_i];                                  \
        /* set M from (pre_i, q_i-1) */                                                                                         \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                             \
            first = SIMDShiftRight(zero, SIMDTotalBytes-SIMDShiftOneN);                                                         \
        } else {                                                                                                                \
            if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneN);\
            else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneN);                      \
            _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);                                                            \
        }                                                                                                                       \
     /* fprintf(stderr, "3 index_i: %d, beg_sn: %d, end_sn: %d\n", index_i, _beg_sn, _end_sn); */                               \
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */                                              \
            remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneN);                                                              \
            dp_h[sn_i] = SIMDMax(SIMDOri(first, remain), dp_h[sn_i]);                                                           \
            first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneN);                                               \
        }                                                                                                                       \
        /* set E from (pre_i, q_i) */                                                                                           \
        _end_sn = MIN_OF_TWO(pre_end_sn, end_sn);                                                                               \
     /* fprintf(stderr, "4 index_i: %d, beg_sn: %d, end_sn: %d\n", index_i, _beg_sn, _end_sn); */                               \
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */                                              \
            dp_e1[sn_i] = SIMDMax(pre_dp_e1[sn_i], dp_e1[sn_i]);                                                                \
            dp_e2[sn_i] = SIMDMax(pre_dp_e2[sn_i], dp_e2[sn_i]);                                                                \
        }                                                                                                                       \
    }                                                                                                                           \
    /* compare M, E, and F */                                                                                                   \
 /* fprintf(stderr, "5 index_i: %d, beg_sn: %d, end_sn: %d\n", index_i, _beg_sn, _end_sn); */                                   \
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) { /* SIMD parallelization */                                                    \
        dp_h[sn_i] = SIMDAdd(dp_h[sn_i], q[sn_i]);                                                                              \
    }                                                                                                                           \
    /* new F start */                                                                                                           \
    first = SIMDShiftRight(SIMDShiftLeft(dp_h[beg_sn], SIMDTotalBytes-SIMDShiftOneN), SIMDTotalBytes-SIMDShiftOneN);            \
    SIMDi first2 = first, tmp;                                                                                                  \
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) {                                                                               \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) set_num = pn;                                                                 \
        else {                                                                                                                  \
            if (sn_i < min_pre_beg_sn) {                                                                                        \
                _err_fatal_simple(__func__, "sn_i < min_pre_beg_sn\n");                                                         \
            } else if (sn_i > max_pre_end_sn) {                                                                                 \
                set_num = sn_i == max_pre_end_sn+1 ? 2 : 1;                                                                     \
            } else set_num = pn;                                                                                                \
        }                                                                                                                       \
        /* F = (H << 1 | x) - OE */                                                                                             \
        dp_f1[sn_i] = SIMDSub(SIMDOri(SIMDShiftLeft(dp_h[sn_i], SIMDShiftOneN), first), GAP_OE1);                               \
        dp_f2[sn_i] = SIMDSub(SIMDOri(SIMDShiftLeft(dp_h[sn_i], SIMDShiftOneN), first2), GAP_OE2);                              \
        /* F = max{F, (F-e)<<1}, F = max{F, (F-2e)<<2} ... */                                                                   \
        SIMD_SET_F(dp_f1[sn_i], log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E1S, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN); \
        SIMD_SET_F(dp_f2[sn_i], log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E2S, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN); \
        /* x = max{H, F+o} */                                                                                                   \
        first = SIMDShiftRight(SIMDMax(dp_h[sn_i], SIMDAdd(dp_f1[sn_i], GAP_O1)), SIMDTotalBytes-SIMDShiftOneN);                \
        first2 = SIMDShiftRight(SIMDMax(dp_h[sn_i], SIMDAdd(dp_f2[sn_i], GAP_O2)), SIMDTotalBytes-SIMDShiftOneN);               \
        dp_h[sn_i] =  SIMDMax(SIMDMax(dp_h[sn_i], dp_e1[sn_i]), dp_e2[sn_i]); tmp = dp_h[sn_i];                                 \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                                             \
            dp_h[sn_i] = SIMDMax(zero, SIMDMax(dp_h[sn_i], SIMDMax(dp_f1[sn_i], dp_f2[sn_i])));                                 \
            SIMDSetIfEqual(dp_e1[sn_i],dp_h[sn_i],tmp,SIMDMax(zero,SIMDMax(SIMDSub(dp_e1[sn_i],GAP_E1),SIMDSub(dp_h[sn_i],GAP_OE1))),zero);\
            SIMDSetIfEqual(dp_e2[sn_i],dp_h[sn_i],tmp,SIMDMax(zero,SIMDMax(SIMDSub(dp_e2[sn_i],GAP_E2),SIMDSub(dp_h[sn_i],GAP_OE2))),zero);\
        } else {                                                                                                                \
            /* H = max{H, F}    */                                                                                              \
            dp_h[sn_i] = SIMDMax(dp_h[sn_i], SIMDMax(dp_f1[sn_i], dp_f2[sn_i]));                                                \
            /* e for next cell */                                                                                               \
            SIMDSetIfEqual(dp_e1[sn_i],dp_h[sn_i],tmp,SIMDMax(SIMDSub(dp_e1[sn_i],GAP_E1),SIMDSub(dp_h[sn_i],GAP_OE1)),SIMD_INF_MIN);\
            SIMDSetIfEqual(dp_e2[sn_i],dp_h[sn_i],tmp,SIMDMax(SIMDSub(dp_e2[sn_i],GAP_E2),SIMDSub(dp_h[sn_i],GAP_OE2)),SIMD_INF_MIN);\
        }                                                                                                                       \
    }                                                                                                                           \
}

#define set_global_max_score(score, i, j) {         \
    if (score > best_score) {                       \
        best_score = score; best_i = i; best_j = j; \
    }                                               \
}

#define set_extend_max_score(score, i, j) {                                                             \
    if (score > best_score) {                                                                           \
        best_score = score; best_i = i; best_j = j; best_id = node_id;                                  \
    } else if (abpt->zdrop > 0) {                                                                       \
        int delta_index = graph->node_id_to_max_remain[best_id] - graph->node_id_to_max_remain[node_id];\
        if (best_score - score > abpt->zdrop + gap_ext1 * abs(delta_index-(j-best_j)))                  \
            break;                                                                                      \
    }                                                                                                   \
}

#define simd_abpoa_global_get_max(score_t, DP_M, dp_sn) {	            \
    int end, in_id, in_index;	                                        \
    for (i = 0; i < graph->node[ABPOA_SINK_NODE_ID].in_edge_n; ++i) {   \
        in_id = graph->node[ABPOA_SINK_NODE_ID].in_id[i];	            \
        in_index = abpoa_graph_node_id_to_index(graph, in_id);	        \
        dp_h = DP_M + in_index * dp_sn;	                                \
        _dp_h = (score_t*)dp_h;	                                        \
        if (qlen > dp_end[in_index]) end = dp_end[in_index];            \
        else end = qlen;                                                \
        set_global_max_score(_dp_h[end], in_index, end);               \
    }	                                                                \
}

#define simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater) {\
    /* select max dp_h */                                                   \
    max = inf_min, max_i = -1;                                              \
    SIMDi a = dp_h[end_sn], b = qi[end_sn];                                 \
    if (end_sn == qlen / pn) SIMDSetIfGreater(a, zero, b, SIMD_INF_MIN, a); \
    for (i = beg_sn; i < end_sn; ++i) {                                     \
        SIMDGetIfGreater(b, a, dp_h[i], a, qi[i], b);                       \
    }                                                                       \
    _dp_h = (score_t*)&a, _qi = (score_t*)&b;                               \
    for (i = 0; i < pn; ++i) {                                              \
        if (_dp_h[i] > max) {                                               \
            max = _dp_h[i]; max_i = _qi[i];                                 \
        }                                                                   \
    }                                                                       \
}

#define simd_abpoa_ada_max_i   {                                                                                        \
    /* set max_pos_left/right for next nodes */                                                                         \
    int out_i = max_i + 1;                                                                                              \
	fprintf(stderr,"graph_out_nodes_num=%d\n",graph->node[node_id].out_edge_n);                                       \
    for (i = 0; i < graph->node[node_id].out_edge_n; ++i) {                                                             \
        int out_node_id = graph->node[node_id].out_id[i];                                                               \
		fprintf(stderr,"out_index=%d\n",graph->node_id_to_index[out_node_id]);                                       \
        if (out_i > graph->node_id_to_max_pos_right[out_node_id]) graph->node_id_to_max_pos_right[out_node_id] = out_i; \
        if (out_i < graph->node_id_to_max_pos_left[out_node_id]) graph->node_id_to_max_pos_left[out_node_id] = out_i;   \
    }                                                                                                                   \
}

// TODO end_bonus for extension
// linear gap penalty: gap_open1 == 0
#define simd_abpoa_lg_align_sequence_to_graph_core(score_t, ab, query, qlen, abpt, res, sp,         \
        SIMDSetOne, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN, SIMDSetIfGreater, SIMDGetIfGreater) { \
    simd_abpoa_lg_var(score_t, sp, SIMDSetOne, SIMDShiftOneN, SIMDAdd);                             \
    simd_abpoa_lg_first_dp(score_t);                                                                \
    for (index_i = 1; index_i < matrix_row_n-1; ++index_i) {                                        \
        simd_abpoa_lg_dp(score_t, SIMDShiftOneN, SIMDMax, SIMDAdd, SIMDSub);                        \
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                 \
            simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                     \
            set_global_max_score(max, index_i, max_i);                                              \
        }                                                                                           \
        if (abpt->align_mode == ABPOA_EXTEND_MODE) {                                                \
            simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                     \
            set_extend_max_score(max, index_i, max_i);                                              \
        }                                                                                           \
        if (abpt->wb >= 0) {                                                                        \
            if (abpt->align_mode == ABPOA_GLOBAL_MODE) {                                            \
                simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                 \
            }                                                                                       \
            simd_abpoa_ada_max_i;                                                                   \
        }                                                                                           \
    }                                                                                               \
    if (abpt->align_mode == ABPOA_GLOBAL_MODE) simd_abpoa_global_get_max(score_t, DP_H, dp_sn);     \
 simd_abpoa_print_lg_matrix(score_t);                                                         \
 printf("best_score: (%d, %d) -> %d\n", best_i, best_j, best_score);                          \
    if (abpt->ret_cigar) simd_abpoa_lg_backtrack(score_t);                                          \
    simd_abpoa_free_var; SIMDFree(GAP_E1S);                                                         \
} 

// affine gap penalty: gap_open1 > 0
#define simd_abpoa_ag_align_sequence_to_graph_core(score_t, ab, query, qlen, abpt, res, sp,                     \
    SIMDSetOne, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN, SIMDSetIfGreater, SIMDGetIfGreater, SIMDSetIfEqual) { \
    simd_abpoa_ag_var(score_t, sp, SIMDSetOne, SIMDShiftOneN, SIMDAdd);                                         \
    simd_abpoa_ag_first_dp(score_t);                                                                            \
    for (index_i = 1; index_i < matrix_row_n-1; ++index_i) {                                                    \
        simd_abpoa_ag_dp(score_t, SIMDShiftOneN, SIMDMax, SIMDAdd, SIMDSub, SIMDGetIfGreater, SIMDSetIfGreater, SIMDSetIfEqual);\
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                             \
            simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                                 \
            set_global_max_score(max, index_i, max_i);                                                          \
        } else if (abpt->align_mode == ABPOA_EXTEND_MODE) {                                                     \
            simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                                 \
            set_extend_max_score(max, index_i, max_i);                                                          \
        }                                                                                                       \
        if (abpt->wb >= 0) {                                                                                    \
            if (abpt->align_mode == ABPOA_GLOBAL_MODE) {                                                        \
                simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                             \
            }                                                                                                   \
            simd_abpoa_ada_max_i;                                                                               \
        }                                                                                                       \
    }                                                                                                           \
    if (abpt->align_mode == ABPOA_GLOBAL_MODE) simd_abpoa_global_get_max(score_t, DP_HEF, 3*dp_sn);             \
 /* simd_abpoa_print_ag_matrix(score_t); printf("best_score: (%d, %d) -> %d\n", best_i, best_j, best_score); */ \
    if (abpt->ret_cigar) simd_abpoa_ag_backtrack(score_t);                                                      \
    simd_abpoa_free_var; SIMDFree(GAP_E1S);                                                                     \
}

// convex gap penalty: gap_open1 > 0 && gap_open2 > 0
#define simd_abpoa_cg_align_sequence_to_graph_core(score_t, ab, query, qlen, abpt, res, sp,                     \
    SIMDSetOne, SIMDMax, SIMDAdd, SIMDSub, SIMDShiftOneN, SIMDSetIfGreater, SIMDGetIfGreater, SIMDSetIfEqual) { \
    simd_abpoa_cg_var(score_t, sp, SIMDSetOne, SIMDShiftOneN, SIMDAdd);                                         \
    simd_abpoa_cg_first_dp(score_t);                                                                            \
    for (index_i = 1; index_i < matrix_row_n-1; ++index_i) {                                                    \
        simd_abpoa_cg_dp(score_t, SIMDShiftOneN, SIMDMax, SIMDAdd, SIMDSub, SIMDGetIfGreater, SIMDSetIfGreater, SIMDSetIfEqual);\
        if (abpt->align_mode == ABPOA_LOCAL_MODE) {                                                             \
            simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                                 \
            set_global_max_score(max, index_i, max_i);                                                          \
        } else if (abpt->align_mode == ABPOA_EXTEND_MODE) {                                                     \
            simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                                 \
            set_extend_max_score(max, index_i, max_i);                                                          \
        }                                                                                                       \
        if (abpt->wb >= 0) {                                                                                    \
            if (abpt->align_mode == ABPOA_GLOBAL_MODE) {                                                        \
                simd_abpoa_max_in_row(score_t, SIMDSetIfGreater, SIMDGetIfGreater);                             \
            }                                                                                                   \
            simd_abpoa_ada_max_i;                                                                               \
        }                                                                                                       \
    }                                                                                                           \
 /* printf("dp_sn: %d\n", tot_dp_sn); */                                                                        \
    if (abpt->align_mode == ABPOA_GLOBAL_MODE) simd_abpoa_global_get_max(score_t, DP_H2E2F, 5*dp_sn);           \
 /* simd_abpoa_print_cg_matrix(score_t);fprintf(stderr,"best_score: (%d, %d) -> %d\n",best_i,best_j,best_score); */ \
    if (abpt->ret_cigar) simd_abpoa_cg_backtrack(score_t);                                                      \
    simd_abpoa_free_var; SIMDFree(GAP_E1S); SIMDFree(GAP_E2S);                                                  \
}

abpoa_cuda_matrix_t *abpoa_init_cuda_matrix(void) {
    abpoa_cuda_matrix_t *abcm = (abpoa_cuda_matrix_t*)_err_malloc(sizeof(abpoa_cuda_matrix_t));
    abcm->c_msize = 0; abcm->c_mem = NULL; abcm->rang_m = 0; 
    abcm->cuda_dp_beg = NULL; abcm->cuda_dp_end = NULL;
    return abcm;
}

abpoa_simd_matrix_t *abpoa_init_simd_matrix(void) {
    abpoa_simd_matrix_t *abm = (abpoa_simd_matrix_t*)_err_malloc(sizeof(abpoa_simd_matrix_t));
    abpoa_cuda_matrix_t *abcm = (abpoa_cuda_matrix_t*)_err_malloc(sizeof(abpoa_cuda_matrix_t));
    abm->s_msize = 0; abm->s_mem = NULL; abm->rang_m = 0; 
    abm->dp_beg = NULL; abm->dp_end = NULL; abm->dp_beg_sn = NULL; abm->dp_end_sn = NULL;
    return abm;
}

void abpoa_free_cuda_matrix(abpoa_cuda_matrix_t *abcm) {
    if (abcm->c_mem) free(abcm->c_mem);
    if (abcm->cuda_dp_beg) {
        free(abcm->cuda_dp_beg); free(abcm->cuda_dp_end);
    } free(abcm);
}

void abpoa_free_simd_matrix(abpoa_simd_matrix_t *abm) {
    if (abm->s_mem) SIMDFree(abm->s_mem);
    if (abm->dp_beg) {
        free(abm->dp_beg); free(abm->dp_end); free(abm->dp_beg_sn); free(abm->dp_end_sn);
    } free(abm);
}

// for cuda: realloc memory everytime the graph is updated (nodes are updated already)
// * index_to_node_id/node_id_to_index/node_id_to_max_remain, max_pos_left/right
// * qp, DP_HE/H (if ag/lg), dp_f, qi (if ada/extend)
// * dp_beg/end, dp_beg/end_sn if band
// * pre_n, pre_index
int cuda_simd_abpoa_realloc(abpoa_t *ab, int qlen, abpoa_para_t *abpt) {
    uint64_t node_n = ab->abg->node_n;
    uint64_t s_msize = (qlen + 1) * abpt->m * sizeof(int); // qp

    if (abpt->gap_mode == ABPOA_LINEAR_GAP) s_msize += ((qlen + 1) * node_n * sizeof(int)); // DP_H, linear
    else if (abpt->gap_mode == ABPOA_AFFINE_GAP) s_msize += ((qlen + 1) * node_n * 3 * sizeof(int)); // DP_HEF, affine
    else s_msize += ((qlen + 1) * node_n * 5 * sizeof(int)); // DP_H2E2F, convex

	uint64_t d_msize = s_msize + sizeof(int); // for max_i

	// backtrack matrix and 2 E_SOURCE_ORDER matrix1 M_source_order matrix, and 1 E1_externsion E2_extension F1_extension F2_entension matrix.
    uint64_t dev_backtrack_size = 5 * (qlen + 1) * node_n * sizeof(uint8_t); 

    /* if (abpt->wb >= 0 || abpt->align_mode == ABPOA_EXTEND_MODE) // qi */
    /*     s_msize += (qlen + 1) * sizeof(int); */

    // if (s_msize > UINT32_MAX) {
        // err_func_format_printf(__func__, "Warning: Graph is too large or query is too long.\n");
        // return 1;
    // }
	/* fprintf(stderr, "node_n:%lld, ori_s_mize%lld, new_s_mize%lld\n", (long long)node_n, (long long)ab->abcm->c_msize, (long long)s_msize); */
	/* fprintf(stderr, "strlen:%d\n", qlen); */

	uint64_t temp_msize = s_msize; // for backtrack matrix
	// for backtrack matrix , 2 * E_source_order matrix , 1 M_source_order matrix, and 1 E1_externsion E2_extension F1_extension F2_entension matrix, so * 4.
	// uint8:0000E1E2F1F2, 0 is extension, 1 is stop.
	uint64_t backtrack_size = 5 * (qlen + 1) * node_n * sizeof(uint8_t); 
	if (ab->abcm->c_mem) free(ab->abcm->c_mem);
    kroundup64(temp_msize);
    ab->abcm->c_mem = (int *)malloc(temp_msize);
	if (ab->abcm->cuda_backtrack_matrix) free(ab->abcm->cuda_backtrack_matrix);
	kroundup64(backtrack_size);
	ab->abcm->cuda_backtrack_matrix = (uint8_t *)malloc(backtrack_size);
	if (!ab->abcm->backtrack_size) ab->abcm->backtrack_size = (int *)malloc(sizeof(int));

	/* fprintf(stderr, "kround_s_msize:%d\n", temp_msize); */

    if (s_msize > ab->abcm->c_msize) {
        /* if (ab->abcm->c_mem) free(ab->abcm->c_mem); */
        /* kroundup64(s_msize); ab->abcm->c_msize = s_msize; */
		/* fprintf(stderr, "kround_s_msize:%d\n", s_msize); */
        /* ab->abcm->c_mem = (int *)malloc(ab->abcm->c_msize); */

		/* checkCudaErrors(cudaMallocManaged((void **)&ab->abcm->dev_mem, d_msize)); */
		if (ab->abcm->dev_mem) (cudaFree(ab->abcm->dev_mem));
		checkCudaErrors(cudaMalloc((void **)&ab->abcm->dev_mem, d_msize));
		if (ab->abcm->dev_backtrack_matrix) (cudaFree(ab->abcm->dev_backtrack_matrix));
		checkCudaErrors(cudaMalloc((void **)&ab->abcm->dev_backtrack_matrix, dev_backtrack_size));
    }

    if ((int)node_n > ab->abcm->rang_m) {
        ab->abcm->rang_m = node_n; kroundup32(ab->abcm->rang_m);
        ab->abcm->cuda_dp_beg = (int*)_err_realloc(ab->abcm->cuda_dp_beg, ab->abcm->rang_m * sizeof(int));
        ab->abcm->cuda_dp_end = (int*)_err_realloc(ab->abcm->cuda_dp_end, ab->abcm->rang_m * sizeof(int));
        ab->abcm->cuda_accumulate_beg = (int*)_err_realloc(ab->abcm->cuda_accumulate_beg, ab->abcm->rang_m * sizeof(int));
    }
    return 0;
}

// realloc memory everytime the graph is updated (nodes are updated already)
// * index_to_node_id/node_id_to_index/node_id_to_max_remain, max_pos_left/right
// * qp, DP_HE/H (if ag/lg), dp_f, qi (if ada/extend)
// * dp_beg/end, dp_beg/end_sn if band
// * pre_n, pre_index
int simd_abpoa_realloc(abpoa_t *ab, int qlen, abpoa_para_t *abpt, SIMD_para_t sp) {
    uint64_t pn = sp.num_of_value, size = sp.size, sn = (qlen + sp.num_of_value) / pn, node_n = ab->abg->node_n;
    uint64_t s_msize = sn * abpt->m * size; // qp

    if (abpt->gap_mode == ABPOA_LINEAR_GAP) s_msize += (sn * node_n * size); // DP_H, linear
    else if (abpt->gap_mode == ABPOA_AFFINE_GAP) s_msize += (sn * node_n * 3 * size); // DP_HEF, affine
    else s_msize += (sn * node_n * 5 * size); // DP_H2E2F, convex

    if (abpt->wb >= 0 || abpt->align_mode == ABPOA_EXTEND_MODE) // qi
        s_msize += sn * size;

    // if (s_msize > UINT32_MAX) {
        // err_func_format_printf(__func__, "Warning: Graph is too large or query is too long.\n");
        // return 1;
    // }
    // fprintf(stderr, "%lld, %lld, %lld\n", (long long)node_n, (long long)ab->abm->s_msize, (long long)s_msize);
    if (s_msize > ab->abm->s_msize) {
        if (ab->abm->s_mem) SIMDFree(ab->abm->s_mem);
        kroundup64(s_msize); ab->abm->s_msize = s_msize;
        ab->abm->s_mem = (SIMDi*)SIMDMalloc(ab->abm->s_msize, size);
    }

    if ((int)node_n > ab->abm->rang_m) {
        ab->abm->rang_m = node_n; kroundup32(ab->abm->rang_m);
        ab->abm->dp_beg = (int*)_err_realloc(ab->abm->dp_beg, ab->abm->rang_m * sizeof(int));
        ab->abm->dp_end = (int*)_err_realloc(ab->abm->dp_end, ab->abm->rang_m * sizeof(int));
        ab->abm->dp_beg_sn = (int*)_err_realloc(ab->abm->dp_beg_sn, ab->abm->rang_m * sizeof(int));
        ab->abm->dp_end_sn = (int*)_err_realloc(ab->abm->dp_end_sn, ab->abm->rang_m * sizeof(int));
    }
    return 0;
}

void cuda_abpoa_init_var(abpoa_para_t *abpt, uint8_t *query, int qlen, int *cuda_qp, int *mat, int INF_MIN) {
    int i, j, k;
	for (i = 0;i < (qlen + 1) * abpt->m;i++){
		cuda_qp[i] = INF_MIN;
	}
	for (k = 0; k < abpt->m; k++) {
		int *p = &mat[k * abpt->m];
		int *_cuda_qp = cuda_qp + k * (qlen + 1);
		_cuda_qp[0] = 0;
		for (j = 0;j < qlen;j++) {
			_cuda_qp[j + 1] = (int32_t)p[query[j]];
		}
	}
	

	/* int32_t *_qi; */
	/* [> generate the query profile <] */
	/* for (i = 0; i < qp_sn * abpt->m; ++i) qp[i] = SIMD_INF_MIN; */
	/* for (k = 0; k < abpt->m; ++k) { [> SIMD parallelization <] */
	/*     int *p = &mat[k * abpt->m]; */
	/*     int32_t *_qp = (int32_t*)(qp + k * qp_sn); _qp[0] = 0; */
	/*     for (j = 0; j < qlen; ++j) _qp[j+1] = (int32_t)p[query[j]]; */
	/*     for (j = qlen+1; j < qp_sn * pn; ++j) _qp[j] = 0; */
	/* } */
    /*  */

	//don't need
    /* if (abpt->wb >= 0 || abpt->align_mode == ABPOA_EXTEND_MODE) { [> query index <]  */
    /*     _qi = (int32_t*)qi; */
    /*     for (i = 0; i <= qlen; ++i) _qi[i] = i; */
    /*     for (i = qlen+1; i < (qlen/pn+1) * pn; ++i) _qi[i] = -1; */
    /* } */
}


void abpoa_init_var(abpoa_para_t *abpt, uint8_t *query, int qlen, SIMDi *qp, SIMDi *qi, int *mat, int qp_sn, int pn, SIMDi SIMD_INF_MIN) {
    int i, j, k; int32_t *_qi;
    /* generate the query profile */
    for (i = 0; i < qp_sn * abpt->m; ++i) qp[i] = SIMD_INF_MIN;
    for (k = 0; k < abpt->m; ++k) { /* SIMD parallelization */
        int *p = &mat[k * abpt->m];
        int32_t *_qp = (int32_t*)(qp + k * qp_sn); _qp[0] = 0;
        for (j = 0; j < qlen; ++j) _qp[j+1] = (int32_t)p[query[j]];
        for (j = qlen+1; j < qp_sn * pn; ++j) _qp[j] = 0;
    }                                     
    if (abpt->wb >= 0 || abpt->align_mode == ABPOA_EXTEND_MODE) { /* query index */ 
        _qi = (int32_t*)qi;
        for (i = 0; i <= qlen; ++i) _qi[i] = i;
        for (i = qlen+1; i < (qlen/pn+1) * pn; ++i) _qi[i] = -1;
    }
}

__global__ void cuda_first_dp_hf1f2(int *dp_f, int *dp_f1, int *dp_f2, int _end, int INF_MIN, int gap_open1, int gap_open2, int gap_ext1, int gap_ext2) {
	int idx = threadIdx.x;
	if (idx == 0) {
		dp_f1[idx] = INF_MIN;
		dp_f2[idx] = INF_MIN;
	} else if (idx <= _end) {
		dp_f1[idx] = -(gap_open1 + idx * gap_ext1);
		dp_f2[idx] = -(gap_open2 + idx * gap_ext2);
		dp_f[idx] = MAX_OF_TWO(dp_f1[idx], dp_f2[idx]);
	}
}

__global__ void cuda_first_dp_he1e2(int *dp_h, int *dp_e1, int *dp_e2, int _end, int INF_MIN, int gap_oe1, int gap_oe2) {
	int idx = threadIdx.x;
	if (idx == 0) {
		dp_h[idx] = 0;
		dp_e1[idx] = -gap_oe1;
		dp_e2[idx] = -gap_oe2;
	} else if (idx <= _end) {
		dp_h[idx] = INF_MIN;
		dp_e1[idx] = INF_MIN;
		dp_e2[idx] = INF_MIN;
	}
}

void cuda_abpoa_cg_first_dp(abpoa_para_t *abpt, abpoa_graph_t *graph, int *cuda_dp_beg, int *cuda_dp_end, int qlen, int w, int *DEV_DP_H2E2F, int gap_open1, int gap_ext1, int gap_open2, int gap_ext2, int gap_oe1, int gap_oe2) {
    /* int i, _end_sn; */
	int i;
	int _cuda_end;
	/* int _cuda_beg; */
	if (abpt->wb >= 0) {
		graph->node_id_to_max_pos_left[0] = graph->node_id_to_max_pos_right[0] = 0;
		for (i = 0; i < graph->node[0].out_edge_n; ++i) { 
			int out_id = graph->node[0].out_id[i];
			graph->node_id_to_max_pos_left[out_id] = graph->node_id_to_max_pos_right[out_id] = 1;
		}
		cuda_dp_beg[0] = 0, cuda_dp_end[0] = w; // GET_AD_DP_BEGIN(graph, w, 0, qlen), dp_end[0] = GET_AD_DP_END(graph, w, 0, qlen);
		cuda_dp_end[0] = cuda_dp_end[0] / 8 * 8 + 7;
	} else {
		cuda_dp_beg[0] = 0, cuda_dp_end[0] = qlen;
	}

	int32_t *dev_dp_h = DEV_DP_H2E2F;
	int32_t *dev_dp_e1 = dev_dp_h + qlen + 1;
	int32_t *dev_dp_e2 = dev_dp_e1 + qlen + 1;
	int32_t *dev_dp_f1 = dev_dp_e2 + qlen + 1;
	int32_t *dev_dp_f2 = dev_dp_f1 + qlen + 1;
	_cuda_end = MIN_OF_TWO(qlen, cuda_dp_end[0]);
	//why does the origin way add the sn?


    /* for (i = 0; i <= _cuda_end; ++i) { */
    /*     cuda_dp_h[i] = INF_MIN;  */
	/*     cuda_dp_e1[i] = INF_MIN;  */
	/*     cuda_dp_e2[i] = INF_MIN;    */
    /* }                                                                                */
	int threads_per_block = _cuda_end + 1;
	cuda_first_dp_he1e2<<<1, threads_per_block>>>(dev_dp_h, dev_dp_e1, dev_dp_e2, _cuda_end, INF_MIN, gap_oe1, gap_oe2);
	cuda_first_dp_hf1f2<<<1, threads_per_block>>>(dev_dp_h, dev_dp_f1, dev_dp_f2, _cuda_end, INF_MIN, gap_open1, gap_open2, gap_ext1, gap_ext2);
    /* cuda_dp_h[0] = 0; cuda_dp_e1[0] = -(gap_oe1); cuda_dp_e2[0] = -(gap_oe2); cuda_dp_f1[0] = cuda_dp_f2[0] = INF_MIN; */
    /* for (i = 1; i <= cuda_dp_end[0]; ++i) { [> no SIMD parallelization <] */
    /*     cuda_dp_f1[i] = -(gap_open1 + gap_ext1 * i); */
    /*     cuda_dp_f2[i] = -(gap_open2 + gap_ext2 * i); */
    /*     cuda_dp_h[i] = MAX_OF_TWO(cuda_dp_f1[i], cuda_dp_f2[i]); // -MIN_OF_TWO(gap_open1+gap_ext1*i, gap_open2+gap_ext2*i); */
    /* } */
	
}

void abpoa_cg_first_dp(abpoa_para_t *abpt, abpoa_graph_t *graph, int *dp_beg, int *dp_end, int *dp_beg_sn, int *dp_end_sn, int pn, int qlen, int w, int dp_sn, SIMDi *DP_H2E2F, SIMDi SIMD_INF_MIN, int32_t inf_min, int gap_open1, int gap_ext1, int gap_open2, int gap_ext2, int gap_oe1, int gap_oe2) {
    int i, _end_sn;
    if (abpt->wb >= 0) {
        graph->node_id_to_max_pos_left[0] = graph->node_id_to_max_pos_right[0] = 0;
        for (i = 0; i < graph->node[0].out_edge_n; ++i) { /* set min/max rank for next_id */
            int out_id = graph->node[0].out_id[i];
            graph->node_id_to_max_pos_left[out_id] = graph->node_id_to_max_pos_right[out_id] = 1;
        }
        dp_beg[0] = 0, dp_end[0] = w; // GET_AD_DP_BEGIN(graph, w, 0, qlen), dp_end[0] = GET_AD_DP_END(graph, w, 0, qlen);
    } else {
        dp_beg[0] = 0, dp_end[0] = qlen;
    }
    dp_beg_sn[0] = (dp_beg[0])/pn; dp_end_sn[0] = (dp_end[0])/pn;
    dp_beg[0] = dp_beg_sn[0] * pn; dp_end[0] = (dp_end_sn[0]+1)*pn-1;
    SIMDi *dp_h = DP_H2E2F; SIMDi *dp_e1 = dp_h + dp_sn; SIMDi *dp_e2 = dp_e1 + dp_sn, *dp_f1 = dp_e2 + dp_sn, *dp_f2 = dp_f1 + dp_sn;
    _end_sn = MIN_OF_TWO(dp_end_sn[0]+1, dp_sn-1);

    for (i = 0; i <= _end_sn; ++i) {
        dp_h[i] = SIMD_INF_MIN; dp_e1[i] = SIMD_INF_MIN; dp_e2[i] = SIMD_INF_MIN;   
    }                                                                               
    int32_t *_dp_h = (int32_t*)dp_h, *_dp_e1 = (int32_t*)dp_e1, *_dp_e2 = (int32_t*)dp_e2, *_dp_f1 = (int32_t*)dp_f1, *_dp_f2 = (int32_t*)dp_f2;
    _dp_h[0] = 0; _dp_e1[0] = -(gap_oe1); _dp_e2[0] = -(gap_oe2); _dp_f1[0] = _dp_f2[0] = inf_min;
    for (i = 1; i <= dp_end[0]; ++i) { /* no SIMD parallelization */
        _dp_f1[i] = -(gap_open1 + gap_ext1 * i);
        _dp_f2[i] = -(gap_open2 + gap_ext2 * i);
        _dp_h[i] = MAX_OF_TWO(_dp_f1[i], _dp_f2[i]); // -MIN_OF_TWO(gap_open1+gap_ext1*i, gap_open2+gap_ext2*i);
    }
}

// add qlen for get consist behaviour with simd (for test)
int cuda_abpoa_max(int *dp_h, int beg, int end, int qlen) {
	int max = INF_MIN;
	int i;
	int max_i;
	/* original */
	/* for (i = beg; i <= end; i++) { */
	/*     if (dp_h[i] > max) { */
	/*         max = dp_h[i]; */
	/*         max_i = i; */
	/*     } */
	/* } */

	// for simd consist behaviour test
	int j;
	int k;
	int *a = (int *)malloc(8 * sizeof(int));
	int *b = (int *)malloc(8 * sizeof(int));
	j = 0;
	for (i = end / 8 * 8;i <= end; i++) {
		a[j] = dp_h[i];
		b[j] = i;
		j++;
	}
	if (end > qlen){
		j = 8 - end + qlen;
		for (i = qlen; i <= end; i++) {
			a[j] = INF_MIN;
			b[j] = 0;
			j++;
		}
		end = qlen;
	}
	for (i = beg; i <= end - 8; i = i + 8) {
		j = 0;
		for (k = i; k <= i + 7; k++) {
			if (dp_h[k] > a[j]) {
				a[j] = dp_h[k];
				b[j] = k;
			}
			j++;
		}
	}
	for (j = 0; j <= 7; j++) {
		if (a[j] > max) {
			max = a[j];
			max_i = b[j];
		}
	}
	// for simd consist behaviour test

	return max_i;
}

__global__ void dev_cuda_abpoa_max(int *dp_h, int beg, int end, int *_max_i, int INF_MIN) {
	int max = INF_MIN;
	int i;
	int max_i;
	for (i = beg; i <= end; i++) {
		if (dp_h[i] > max) {
			max = dp_h[i];
			max_i = i;
		}
	}
	*_max_i = max_i;
}

int abpoa_max(SIMDi SIMD_INF_MIN, SIMDi zero, int inf_min, SIMDi *dp_h, SIMDi *qi, int qlen, int pn, int beg_sn, int end_sn) {
    /* select max dp_h */
    int max = inf_min, max_i = -1, i;
    SIMDi a = dp_h[end_sn], b = qi[end_sn];
    if (end_sn == qlen / pn) SIMDSetIfGreateri32(a, zero, b, SIMD_INF_MIN, a);
    for (i = beg_sn; i < end_sn; ++i) {
        SIMDGetIfGreateri32(b, a, dp_h[i], a, qi[i], b);
    }
    int32_t *_dp_h = (int32_t*)&a, *_qi = (int32_t*)&b;
    for (i = 0; i < pn; ++i) {
        if (_dp_h[i] > max) {
            max = _dp_h[i]; max_i = _qi[i];
        }
    }
    return max_i;
}

void abpoa_ada_max_i(int max_i, abpoa_graph_t *graph, int node_id, FILE *file) {
    /* set max_pos_left/right for next nodes */
    int out_i = max_i + 1, i;
	/* fprintf(file, "out_i:%d\n", out_i); */
    for (i = 0; i < graph->node[node_id].out_edge_n; ++i) {
        int out_node_id = graph->node[node_id].out_id[i];
		/* fprintf(file, "prei:%d,origin_max_right:%d\n",i ,graph->node_id_to_max_pos_right[out_node_id]); */
        /* if (out_i > graph->node_id_to_max_pos_right[out_node_id]) graph->node_id_to_max_pos_right[out_node_id] = out_i; */
        /* if (out_i < graph->node_id_to_max_pos_left[out_node_id]) graph->node_id_to_max_pos_left[out_node_id] = out_i; */
		/* fprintf(file, "after_max_right:%d\n", graph->node_id_to_max_pos_right[out_node_id]); */
        if (out_i > graph->node_id_to_max_pos_right[out_node_id]){
			/* fprintf(file, "out_node_id:%d, old_max_position_right:%d\n", out_node_id, graph->node_id_to_max_pos_right[out_node_id]); */
			graph->node_id_to_max_pos_right[out_node_id] = out_i;
		}
        if (out_i < graph->node_id_to_max_pos_left[out_node_id]){
			/* fprintf(file, "out_node_id:%d, old_max_position_left:%d\n", out_node_id, graph->node_id_to_max_pos_left[out_node_id]); */
			graph->node_id_to_max_pos_left[out_node_id] = out_i;
		}
    }
	
}

void cuda_abpoa_global_get_max(abpoa_graph_t *graph, int *CUDA_DP_H_HE, int *DEV_DP_H2E2F, int qlen, int *dp_end, int32_t *best_score, int *best_i, int *best_j) {
    int in_id, in_index, i;
	/* int score = INF_MIN, tempi, tempj, temp_score; */
	int temp_score;
    for (i = 0; i < graph->node[ABPOA_SINK_NODE_ID].in_edge_n; ++i) {
        in_id = graph->node[ABPOA_SINK_NODE_ID].in_id[i];
        in_index = abpoa_graph_node_id_to_index(graph, in_id);
		
		int end = dp_end[in_index];

		// for simd consist behaviour
		if (end > qlen) end = qlen;
		// for simd consist behaviour

		/* fprintf(stderr, "i:%d, dp_h[end]:%d\n", i, dp_end[end]); */
		/* fprintf(stderr, "index:%d\n", in_index); */
		/* fprintf(stderr, "end:%d\n", end); */
		/* if (dp_h[end] > *best_score) { */
		/*     *best_score = dp_h[end]; */
		/*     *best_i = in_index; */
		/*     *best_j = end; */
		/* } */
        /*  */
		int *dev_dp_h = DEV_DP_H2E2F + in_index * (qlen + 1) * 5;
		cudaMemcpy(&temp_score, dev_dp_h + end, sizeof(int), cudaMemcpyDeviceToHost);
		/* int *dp_h = CUDA_DP_H_HE + in_index * (qlen + 1) * 5; */
		/* if (dp_h[end] != temp_score) { */
		/*     fprintf(stderr, "i:%d\n", i); */
		/*     exit(1); */
		/* } else { */
		/*     fprintf(stderr, "dp_h:%d,temp_score:%d\n", dp_h[end], temp_score); */
		/* } */
		if (temp_score > *best_score) {
			*best_score = temp_score;
			*best_i = in_index;
			*best_j = end;
		}

    }
}

void abpoa_global_get_max(abpoa_graph_t *graph, SIMDi *DP_H_HE, int dp_sn, int qlen, int *dp_end, int32_t *best_score, int *best_i, int *best_j) {
    int in_id, in_index, i;
    for (i = 0; i < graph->node[ABPOA_SINK_NODE_ID].in_edge_n; ++i) {
        in_id = graph->node[ABPOA_SINK_NODE_ID].in_id[i];
        in_index = abpoa_graph_node_id_to_index(graph, in_id);
        SIMDi *dp_h = DP_H_HE + in_index * dp_sn;
        int32_t *_dp_h = (int32_t*)dp_h;
        int end;
        if (qlen > dp_end[in_index]) end = dp_end[in_index];
        else end = qlen;
        if (_dp_h[end] > *best_score) {
            *best_score = _dp_h[end]; *best_i = in_index; *best_j = end;
        }
    }
}
__global__ void cuda_set_F_nb2(int *dp_f, int e,int *dp_f2, int e2, int _beg, int _end, int cov_bit, int max_right) {
	int block_num = 0;
	int base = blockDim.x * block_num + _beg;
	int idx = threadIdx.x + base;
	int sum;
	while (base < _end) {
		int i = 1;
		/* if (base > max_right + 1) { */
		/*     sum = 0; */
		/* } else { */
		/*     sum = 1; */
		/* } */
		// sum + 1:the number of haved done.
		sum = 0;
		while (sum + 1 < blockDim.x) {
			/* if (idx >= i + _beg + block_num * blockDim.x && idx <= blockDim.x+ block_num * blockDim.x && idx <= cov_bit+ block_num * blockDim.x && idx <= _end) { */
			if (idx >= i + base && idx <= base + blockDim.x - 1 && idx <= _end) {
				if (dp_f[idx - i] - i * e > dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - i * e;
				}
				if (dp_f2[idx - i] - i * e2 > dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - i * e2;
				}
			}
			__syncthreads();
			sum += i;
			i *= 2;
			cov_bit += i;
		}
		block_num++;
		base = blockDim.x * block_num + _beg;
		idx = threadIdx.x + base;
		/* cov_bit = 1; //change */
		/* if (threadIdx.x == 0 && idx <= _end && dp_f[idx - 1] - e > dp_f[idx]) { */
		/*     dp_f[idx] = dp_f[idx -1] - e; */
		/* } */
		/* if (threadIdx.x == 0 && idx <= _end && dp_f2[idx - 1] - e2 > dp_f2[idx]) { */
		/*     dp_f2[idx] = dp_f2[idx -1] - e2; */
		/* } */
		/* __syncthreads(); */
	}
}

template <unsigned int logn>
__global__ void cuda_set_F_nb(int *dp_f, int e,int *dp_f2, uint8_t *e1e2f1f2_extension, int e2, int _beg, int _end, int cov_bit, int max_right) {
	int raw_idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = _beg + raw_idx;
	int sum;
	int size = blockDim.x * gridDim.x;
	/* int flag = 0; */
	int i = 1;
	int j = 0;
	int raw_e = e;
	int raw_e2 = e2;
	int flag = 0;
		// sum + 1:the number of haved done.
		/* sum = 0; */
		/* i = 1; */
		/* if (blockIdx.x == 0) { */
		/*     j = 1; */
		/* } else { */
		/*     j = 0; */
		/* } */
		/* e = raw_e; */
		/* e2 = raw_e2; */
	/* while (idx <= _end) { */
	/*     e1e2f1f2_extension[raw_idx] |= (1 << 1); */
	/*     idx += size; */
	/*     raw_idx += size; */
	/* } */
	/* raw_idx = threadIdx.x + blockDim.x * blockIdx.x; */
	/* idx = _beg + raw_idx; */

	// 1 is stop, 0 is extension. oe is stop, e is extension, so set 1 first, when renew e, set 0.
	// 0,1反过来，因为把倒数第二位设置为1方便。
	while (idx <= _end) {
		// sum + 1:the number of haved done.
		sum = 0;
		i = 1;
		if (flag == 0) {
			if (blockIdx.x == 0) {
				j = 1;
			}
		} else {
			j = 0;
		}
		e = raw_e;
		e2 = raw_e2;
		/* while (sum + 1 < size) { */
		// 去掉+1，则会影响第一次大循环计算。由于有j的存在，限制了被填充的单元必然在合适的范围，例如size为2，len为2，第一次小循环j=1，sum=0，只有1号线程计算，0号线程不计算
		// （由j的限制），第二次小循环，j=2，sum=1，进入循环，但是j限制了至少2号线程才能计算，所以无线程可计算，符合规则。
		// 第二次大循环以后完全符合要求，只是最后一次小循环只有最后一个线程参与计算，有些浪费资源。
		/* while (sum < size) { */
		/*     if (raw_idx >= j) { */
		/*         dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
		/*         [> if (dp_f[idx - i] - e > dp_f[idx]) { <] */
		/*         [>     dp_f[idx] = dp_f[idx - i] - e; <] */
		/*         [> } <] */
		/*         dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
		/*         [> if (dp_f2[idx - i] - e2 > dp_f2[idx]) { <] */
		/*         [>     dp_f2[idx] = dp_f2[idx - i] - e2; <] */
		/*         [> } <] */
		/*     } */
		/*     e <<= 1; */
		/*     e2 <<= 1; */
		/*     [> if (sum) <] */
		/*     __syncthreads(); */
		/*     sum += i; */
		/*     [> i *= 2; <] */
		/*     i <<= 1; */
		/*     if (flag == 0) { */
		/*         j = i; */
		/*     } else { */
		/*         j = i - 1; */
		/*     } */
		/*     cov_bit += i; */
		/* } */
		// 循环展开，例如size为8，1 2 4 1需要4次小循环，log8 + 1
		// 0 1 3 7  0 1 4 11  1 2 4 1  0 1 2 4  0 1 3 7
		// j的含义：接下来要填idx为j的线程,故j小于size作为循环进入条件

		// 1 loop
		if (logn >= 0) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 2 loop
		if (logn >= 1) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 3 loop
		if (logn >= 2) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 4 loop
		if (logn >= 3) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 5 loop
		if (logn >= 4) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 6 loop
		if (logn >= 5) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 7 loop
		if (logn >= 6) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 8 loop
		if (logn >= 7) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 9 loop
		if (logn >= 8) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 10 loop
		if (logn >= 9) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		// 11 loop
		if (logn >= 10) {
			if (raw_idx >= j) {
				/* dp_f[idx] = dp_f[idx - i] - e > dp_f[idx] ? dp_f[idx - i] - e : dp_f[idx]; */
				if (dp_f[idx - i] - e >= dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - e;
					e1e2f1f2_extension[raw_idx] |= (1 << 1);
				}
				/* dp_f2[idx] = dp_f2[idx - i] - e2 > dp_f2[idx] ? dp_f2[idx - i] - e2 : dp_f2[idx]; */
				if (dp_f2[idx - i] - e2 >= dp_f2[idx]) {
					dp_f2[idx] = dp_f2[idx - i] - e2;
					e1e2f1f2_extension[raw_idx] |= 1;
				}
			}
			e <<= 1;
			e2 <<= 1;
			if (size - j > 32)
				__syncthreads();
			sum += i;
			/* i *= 2; */
			i <<= 1;
			if (flag == 0) {
				j = i;
			} else {
				j = i - 1;
			}
			/* j = i - 1; */
			cov_bit += i;
		}
		idx += size;
		raw_idx += size;
		flag = 1;
	}
}

__global__ void cuda_set_F(int *dp_f, int e, int _beg, int _end, int cov_bit, int max_right) {
	int block_num = 0;
	int base = blockDim.x * block_num + _beg;
	int idx = threadIdx.x + base;
	int sum;
	while (base < _end) {
		int i = 1;
		if (base > max_right + 1) {
			sum = 0;
		} else {
			sum = 1;
		}
		// sum + 1:the number of haved done.
		while (sum + 1 < blockDim.x) {
			/* if (idx >= i + _beg + block_num * blockDim.x && idx <= blockDim.x+ block_num * blockDim.x && idx <= cov_bit+ block_num * blockDim.x && idx <= _end) { */
			if (idx >= i + base && idx <= base + blockDim.x - 1 && idx <= _end) {
				if (dp_f[idx - i] - i * e > dp_f[idx]) {
					dp_f[idx] = dp_f[idx - i] - i * e;
				}
			}
			__syncthreads();
			sum += i;
			i *= 2;
			cov_bit += i;
		}
		block_num++;
		base = blockDim.x * block_num + _beg;
		idx = threadIdx.x + base;
		/* cov_bit = 1; //change */
		if (threadIdx.x == 0 && idx <= _end && dp_f[idx - 1] - e > dp_f[idx]) {
			dp_f[idx] = dp_f[idx -1] - e;
		}
		__syncthreads();
	}
}

// set the max score in dp_f and renew E
__global__ void cuda_get_max_F_and_set_E(int *dp_h, int *dp_e1,int *dp_e2, int *dp_f1,int *dp_f2, uint8_t *e_source_order, uint8_t *e_source_order2, uint8_t *e1e2f1f2_extension, int _beg, int _end, int gap_oe1, int gap_oe2, int gap_ext1, int gap_ext2, int INF_MIN, uint8_t *backtrack) {
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	int flag = 1;
	int flag_f = 0;
	while (idx <= _end) {
		/* e1e2f1f2_extension[raw_idx] = 0; */
		if (idx >= _beg && idx <= _end) {
			if (dp_e1[idx] > dp_h[idx]) {
				dp_h[idx] = dp_e1[idx];
				flag = 2;
			} 
			if(dp_e2[idx] > dp_h[idx]) {
				dp_h[idx] = dp_e2[idx];
				flag = 3;
			} 
			if (dp_f1[idx] > dp_h[idx]){
				dp_h[idx] = dp_f1[idx];
				flag = 0;
				flag_f = 1;
			} 
			if (dp_f2[idx] > dp_h[idx]){
				dp_h[idx] = dp_f2[idx];
				flag = 0;
				flag_f = 2;
			} 
			if (flag == 0) {
				dp_e1[idx] = INF_MIN;
				dp_e2[idx] = INF_MIN;
				if (flag_f == 1) {
					backtrack[raw_idx] = 1; // F1
				} else if (flag_f == 2) {
					backtrack[raw_idx] = 2; // F2
				}
			} else {
				if (flag == 1) backtrack[raw_idx] = 0; // M
				else if (flag == 2) {
					backtrack[raw_idx] = 3; // E1
				} else if (flag == 3) {
					backtrack[raw_idx] = 4; // E2
				}
				if (dp_e1[idx] - gap_ext1 > dp_h[idx] - gap_oe1) {
					dp_e1[idx] = dp_e1[idx] - gap_ext1;
				} else {
					// raw_idx+1 -> raw_idx:E1
					// raw_idx -> raw:idx-1:M(i - 1)
					// stop e1
					e1e2f1f2_extension[raw_idx] |= (1 << 3);
					dp_e1[idx] = dp_h[idx] - gap_oe1;
				}
				if (dp_e2[idx] - gap_ext2 > dp_h[idx] - gap_oe2) {
					dp_e2[idx] = dp_e2[idx] - gap_ext2;
				} else {
					// stop e2
					e1e2f1f2_extension[raw_idx] |= (1 << 2);
					dp_e2[idx] = dp_h[idx] - gap_oe2;
				}
			}
				// record the real e_source_order,it is from e2
			/* if (flag == 3) { */
			/*     e_source_order[raw_idx] = e_source_order2[raw_idx]; */
			/* } */
		}
		idx += gridDim.x * blockDim.x;
		raw_idx += gridDim.x * blockDim.x;
	}
}

/* // a = max(a, b) */
/* __global__ void cuda_max(int *a, int * b, int _beg, int _end) { */
/*     int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg; */
/*     if (idx >= _beg && idx <= _end){ */
/*         a[idx] = a[idx] > b[idx] ? a[idx] : b[idx]; */
/*     } */
/* } */

// f = (h - oe)>>1
// a is the same as dp_h
__global__ void cuda_add_and_suboe1_shift_right_one_nb(int *f, int *f2, int *dp_h, int oe,int oe2, int * q, int _beg, int _end, int INF_MIN, int max_pre_end) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	while (idx <= _end) {
		dp_h[idx] = dp_h[idx] + q[idx];
		idx += blockDim.x * gridDim.x;
	}
	idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	while (idx <= max_pre_end + 2) {
		/* dp_h[idx] += q[idx]; */
		f[idx] = dp_h[idx - 1] - oe;
		f2[idx] = dp_h[idx - 1] - oe2;
		idx += blockDim.x * gridDim.x;
	}
	while (idx <= _end) {
		/* dp_h[idx] += q[idx]; */
		f[idx] = INF_MIN;
		f2[idx] = INF_MIN;
		idx += blockDim.x * gridDim.x;
	}
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		f[_beg] = INF_MIN;
		f2[_beg] = INF_MIN;
	}
	/* while (idx <= _end) { */
	/*     if (idx >= _beg && idx <= _end){ */
	/*         a[idx] = a[idx] + b[idx]; */
	/*     } */
	/*     idx += blockDim.x * gridDim.x; */
	/* } */
}
__global__ void cuda_suboe1_shift_right_one_nb(int *f, int *f2, int *dp_h, int oe,int oe2, int _beg, int _end, int INF_MIN, int max_pre_end) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg + 1;
	while (idx <= max_pre_end + 2) {
		f[idx] = dp_h[idx - 1] - oe;
		f2[idx] = dp_h[idx - 1] - oe2;
		idx += blockDim.x * gridDim.x;
	}
	while (idx <= _end) {
		f[idx] = INF_MIN;
		f2[idx] = INF_MIN;
		idx += blockDim.x * gridDim.x;
	}
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		f[_beg] = INF_MIN;
		f2[_beg] = INF_MIN;
	}
}

__global__ void cuda_suboe1_shift_right_one(int *f, int *dp_h, int oe, int _beg, int _end, int INF_MIN, int max_pre_end) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	while (idx >= _beg && idx <= _end) {
		if (idx == _beg) {
			f[idx] = INF_MIN;
		} else {
			if (idx > max_pre_end + 2) {
				f[idx] = INF_MIN;
			} else {
				f[idx] = dp_h[idx - 1] - oe;
			}
		}
		/* if (idx > _beg && idx <= _end){ */
		/*     if (idx > max_pre_end + 2) { */
		/*         f[idx] = INF_MIN; */
		/*     } else { */
		/*         f[idx] = dp_h[idx - 1] - oe; */
		/*     } */
		/* } else if (idx == _beg) { */
		/*     f[idx] = INF_MIN; */
		/* } */
		idx += blockDim.x * gridDim.x;
	}
}
/* __global__ void cuda_suboe1_shift_right_one(int *f, int *dp_h, int oe, int _beg, int _end, int INF_MIN, int max_pre_end) { */
/*     int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg; */
/*     if (idx > _beg && idx <= _end){ */
/*         if (idx > max_pre_end + 2) { */
/*             f[idx] = INF_MIN; */
/*         } else { */
/*             f[idx] = dp_h[idx - 1] - oe; */
/*         } */
/*     } else if (idx == _beg) { */
/*         f[idx] = INF_MIN; */
/*     } */
/* } */

// a = a + b
__global__ void cuda_add(int *a, int * b, int _beg, int _end) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	while (idx <= _end) {
		a[idx] = a[idx] + b[idx];
		idx += blockDim.x * gridDim.x;
	}
}

// gap is the distance of _cuda_beg and cuda_beg, which is set for m_source_order index.
__global__ void cuda_set_ME(int *dp_h, int * pre_dp_h, uint8_t *m_source_order,int *dp_e, int * pre_dp_e,int *dp_e2, int * pre_dp_e2, uint8_t *e_source_order, uint8_t *e_source_order2, int _beg, int _end, int flag, int order, int gap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x + gap;
	if (idx == _beg && flag == 1 && pre_dp_h[_beg - 1] > dp_h[_beg]) {
		dp_h[_beg] = pre_dp_h[_beg - 1];
		m_source_order[gap] = order;
	}
	/* if (pre_dp_e[_beg] > dp_e[_beg]){ */
	/*     dp_e[_beg] = pre_dp_e[_beg]; */
	/*     e_source_order[gap] = order; */
	/* } */
	/* if (pre_dp_e2[_beg] > dp_e2[_beg]){ */
	/*     dp_e2[_beg] = pre_dp_e2[_beg]; */
	/*     e_source_order2[gap] = order; */
	/* } */
	while (idx <= _end) {
		if (pre_dp_e[idx] > dp_e[idx]){
			dp_e[idx] = pre_dp_e[idx];
			e_source_order[raw_idx] = order;
		}
		if (pre_dp_e2[idx] > dp_e2[idx]){
			dp_e2[idx] = pre_dp_e2[idx];
			e_source_order2[raw_idx] = order;
		}
		if (pre_dp_h[idx] > dp_h[idx + 1]){
			dp_h[idx + 1] = pre_dp_h[idx];
			m_source_order[raw_idx + 1] = order;
		}
		idx += blockDim.x * gridDim.x;
		raw_idx += blockDim.x * gridDim.x;
	}
	/* if (pre_dp_h[_end] > dp_h[_end + 1]){ */
	/*     dp_h[_end + 1] = pre_dp_h[_end]; */
	/*     m_source_order[_end - _beg + 1] = order; */
	/* } */
}

__global__ void cuda_set_E_nb2(int *dp_e, int * pre_dp_e,int *dp_e2, int * pre_dp_e2, uint8_t *e_source_order, uint8_t *e_source_order2, int _beg, int _end, int order, int gap) {
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x + gap;
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	while (idx >= _beg && idx <= _end) {
		if (pre_dp_e[idx] > dp_e[idx]){
			dp_e[idx] = pre_dp_e[idx];
			e_source_order[raw_idx] = order;
		}
		if (pre_dp_e2[idx] > dp_e2[idx]){
			dp_e2[idx] = pre_dp_e2[idx];
			e_source_order2[raw_idx] = order;
		}
		idx += blockDim.x * gridDim.x;
		raw_idx += blockDim.x * gridDim.x;
	}
}

__global__ void cuda_set_E_nb(int *dp_e, int * pre_dp_e, int _beg, int _end) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	while (idx >= _beg && idx <= _end) {
		if (pre_dp_e[idx] > dp_e[idx]){
			dp_e[idx] = pre_dp_e[idx];
		}
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void cuda_set_E(int *dp_e, int * pre_dp_e, int _beg, int _end) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	if (idx >= _beg && idx <= _end){
		if (pre_dp_e[idx] > dp_e[idx]){
			dp_e[idx] = pre_dp_e[idx];
		}
	}
}

// gap is the distance of _cuda_beg and cuda_beg, which is set for m_source_order index.
__global__ void cuda_set_M(int *dp_h, int * pre_dp_h, uint8_t *m_source_order, int _beg, int _end, int flag, int order, int gap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x + gap;
	while (idx <= _end) {
		if (idx > _beg){
			if (pre_dp_h[idx - 1] > dp_h[idx]){
				dp_h[idx] = pre_dp_h[idx - 1];
				m_source_order[raw_idx] = order;
			}
		} else if (idx == _beg) {
			if (flag == 1 && pre_dp_h[idx - 1] > dp_h[idx]) {
				dp_h[idx] = pre_dp_h[idx - 1];
				m_source_order[raw_idx] = order;
			}
			/* if (first > dp_h[idx]){ */
			/*     dp_h[idx] = first; */
			/* } */
		}
		idx += blockDim.x * gridDim.x;
		raw_idx += blockDim.x * gridDim.x;
	}
}

__global__ void cuda_set_ME_first_node_nb(int *dp_h, int * pre_dp_h, uint8_t *m_source_order,int *dp_e, int * pre_dp_e,int *dp_e2, int * pre_dp_e2, uint8_t *e_source_order, uint8_t *e_source_order2, int beg, int end, int _beg, int _end, int INF_MIN, int flag) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + beg;
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x;
	// 掐头去尾，m比e右移一个单元
	if (raw_idx == 0) {
		dp_h[beg] = INF_MIN;
		m_source_order[0] = UINT8_MAX;
	}
	while (idx < end) {
		if ((idx >= beg && idx < _beg) || (idx >_end && idx < end)){
			dp_h[idx + 1] = INF_MIN;
			m_source_order[raw_idx + 1] = UINT8_MAX;
			dp_e[idx] = INF_MIN;
			dp_e2[idx] = INF_MIN;
			e_source_order[raw_idx] = UINT8_MAX;
			e_source_order2[raw_idx] = UINT8_MAX;
		} else if (idx >= _beg && idx <= _end) {
			dp_h[idx + 1] = pre_dp_h[idx];
			m_source_order[raw_idx + 1] = 0;
			dp_e[idx] = pre_dp_e[idx];
			dp_e2[idx] = pre_dp_e2[idx];
			e_source_order[raw_idx] = 0;
			e_source_order2[raw_idx] = 0;
		}
		idx += blockDim.x * gridDim.x;
		raw_idx += blockDim.x * gridDim.x;
	}
	// 掐头去尾，m比e右移一个单元
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		dp_e[end] = INF_MIN;
		dp_e2[end] = INF_MIN;
		e_source_order[end - beg] = UINT8_MAX;
		e_source_order2[end - beg] = UINT8_MAX;
	}
}

__global__ void cuda_set_E_first_node_nb(int *dp_e, int * pre_dp_e,int *dp_e2, int * pre_dp_e2, uint8_t *e_source_order, uint8_t *e_source_order2, int beg, int end, int _beg, int _end, int INF_MIN) {
	int raw_id = blockIdx.x * blockDim.x + threadIdx.x; 
	int idx = blockIdx.x * blockDim.x + threadIdx.x + beg;
	while (idx <= end) {
		if ((idx >= beg && idx < _beg) || (idx >_end && idx <= end)){
			dp_e[idx] = INF_MIN;
			dp_e2[idx] = INF_MIN;
			e_source_order[raw_id] = UINT8_MAX;
			e_source_order2[raw_id] = UINT8_MAX;
		} else if (idx >= _beg && idx <= _end) {
			dp_e[idx] = pre_dp_e[idx];
			dp_e2[idx] = pre_dp_e2[idx];
			e_source_order[raw_id] = 0;
			e_source_order2[raw_id] = 0;
		}
		idx += blockDim.x * gridDim.x;
		raw_id += blockDim.x * gridDim.x;
	}
}

// set E1 E2 for the first node
__global__ void cuda_set_E_first_node(int *dp_e, int * pre_dp_e, int beg, int end, int _beg, int _end, int INF_MIN) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + beg;
	if ((idx >= beg && idx < _beg) || (idx >_end && idx <= end)){
		dp_e[idx] = INF_MIN;
	} else if (idx >= _beg && idx <= _end) {
		dp_e[idx] = pre_dp_e[idx];
	}
}

__global__ void cuda_set_M_first_node_nb(int *dp_h, int * pre_dp_h, uint8_t *m_source_order, int beg, int end, int _beg, int _end, int INF_MIN, int flag) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + beg;
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x;
	while (idx <= end) {
		if ((idx >= beg && idx < _beg) || (idx >_end && idx <= end)){
			dp_h[idx] = INF_MIN;
			m_source_order[raw_idx] = UINT8_MAX;
		} else if (idx >= _beg && idx <= _end) {
			dp_h[idx] = pre_dp_h[idx - 1];
			m_source_order[raw_idx] = 0;
		}
		idx += blockDim.x * gridDim.x;
		raw_idx += blockDim.x * gridDim.x;
	}
}
// set M for the first node
__global__ void cuda_set_M_first_node(int *dp_h, int * pre_dp_h, int beg, int end, int _beg, int _end, int INF_MIN, int flag) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + beg;
	if (idx == _beg) {
		if (flag == 1) {
			dp_h[idx] = pre_dp_h[_beg - 1];
		} else {
			dp_h[idx] = INF_MIN;
		}
	} else if ((idx >= beg && idx < _beg) || (idx >_end && idx <= end)){
		dp_h[idx] = INF_MIN;
	} else if (idx > _beg && idx <= _end) {
		dp_h[idx] = pre_dp_h[idx - 1];
	}
}

// input the [beg, end] part of data,when finished, add beg to max_index 
template <unsigned int block_size>
__global__ void reduce_max_index(int *data, int *out_data, int *out_index, int n) {
	__shared__ int share_data[256];
	__shared__ int share_index[256];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (block_size * 2) + tid;
	unsigned int grid_size = block_size * gridDim.x;
	if (i + block_size < n && data[i] < data[i + block_size]) {
		share_data[tid] = data[i + block_size];
		share_index[tid] = i + block_size;
	} else {
		share_data[tid] = data[i];
		share_index[tid] = i;
	}
	i = grid_size * 2 + blockIdx.x * block_size + tid;
	while (i < n) {
		if (share_data[tid] < data[i]) {
			share_data[tid] = data[i];
			share_index[tid] = i;
		}
		i += grid_size;
	}
	__syncthreads();

	if (block_size >= 512) {
		if (tid < 256) {
			if (share_data[tid] < share_data[tid + 256]) {
				share_data[tid] = share_data[tid + 256];
				share_index[tid] = share_index[tid + 256];
			}
		}
	__syncthreads();
	}
	if (block_size >= 256) {
		if (tid < 128) {
			if (share_data[tid] < share_data[tid + 128]) {
				share_data[tid] = share_data[tid + 128];
				share_index[tid] = share_index[tid + 128];
			}
		}
	__syncthreads();
	}
	if (block_size >= 128) {
		if (tid < 64) {
			if (share_data[tid] < share_data[tid + 64]) {
				share_data[tid] = share_data[tid + 64];
				share_index[tid] = share_index[tid + 64];
			}
		}
	__syncthreads();
	}
	if (tid < 32) {
		if (block_size >= 64) {
			if (share_data[tid] < share_data[tid + 32]) {
				share_data[tid] = share_data[tid + 32];
				share_index[tid] = share_index[tid + 32];
			}
		}
		if (tid < 16) {
			if (block_size >= 32) {
				if (share_data[tid] < share_data[tid + 16]) {
					share_data[tid] = share_data[tid + 16];
					share_index[tid] = share_index[tid + 16];
				}
			}
		}
		if (tid < 8) {
			if (block_size >= 16) {
				if (share_data[tid] < share_data[tid + 8]) {
					share_data[tid] = share_data[tid + 8];
					share_index[tid] = share_index[tid + 8];
				}
			}
		}
		if (tid < 4) {
			if (block_size >= 8) {
				if (share_data[tid] < share_data[tid + 4]) {
					share_data[tid] = share_data[tid + 4];
					share_index[tid] = share_index[tid + 4];
				}
			}
		}
		if (tid < 2) {
			if (block_size >= 4) {
				if (share_data[tid] < share_data[tid + 2]) {
					share_data[tid] = share_data[tid + 2];
					share_index[tid] = share_index[tid + 2];
				}
			}
		}
		if (tid < 1) {
			if (block_size >= 2) {
				if (share_data[tid] < share_data[tid + 1]) {
					share_data[tid] = share_data[tid + 1];
					share_index[tid] = share_index[tid + 1];
				}
			}
		}
		if (tid == 0) {
			out_data[blockIdx.x] = share_data[0];
			out_index[blockIdx.x] = share_index[0];
		}
	}
}
// get max index, only if the blockdim.x == 32, but too slow
__global__ void maxReduce(int* d_data, int n, int beg, int end, int *max_index)
{
    // compute max over all threads, store max in d_data[0]
    int ti = threadIdx.x + beg;

    __shared__ volatile int max_value;
    __shared__ volatile int index;

    if (ti == beg) max_value = d_float_min;
	n += beg;

    for (int bi = 0; bi < n; bi += blockDim.x)
    {
        int i = bi + ti;
        if (i >= n) break;
        
        int v = d_data[i];
		/* __syncthreads(); */

        while (max_value < v)
        {
            max_value = v;
        }
		if (v == max_value) *max_index = i;

		/* __syncthreads(); */
    }

    /* if (ti == beg){ */
	/*     [> d_data[1] = index; <] */
	/*     [> d_data[0] = max_value; <] */
	/*     *max_index = index; */
	/* } */
}

void cuda_print_row_h(FILE *file, int index, int *dp_h, int beg, int end, int qlen) {
	int i;
    fprintf(file, "index:%d\n", index);	                                                                        
	for (i = beg; i <= end; i++ ) {
        fprintf(file, "%d:(%d)\t", i, dp_h[i]);
	}
	fprintf(file, "\n");
}

void cuda_print_row(FILE *file, int index, int *dp_h, int beg, int end, int qlen) {
	int i;
	int *dp_e1 = dp_h + qlen + 1;
	int *dp_e2 = dp_e1 + qlen + 1;
	int *dp_f1 = dp_e2 + qlen + 1;
	int *dp_f2 = dp_f1 + qlen + 1;
    fprintf(file, "index:%d\n", index);	                                                                        
    fprintf(file, "beg:%d\n", beg);	                                                                        
    fprintf(file, "end:%d\n", end);	                                                                        
	for (i = beg; i <= end; i++ ) {
        fprintf(file, "%d:(%d,%d,%d,%d,%d)\n", i, dp_h[i], dp_e1[i],dp_e2[i], dp_f1[i],dp_f2[i]);
	}
}

void my_simd_abpoa_print_cg_matrix_row(int *dp_h, int sn, FILE *file, int index_i, int beg, int end, int numof_value) {                                   
	int i;
	int *dp_e1 = dp_h + numof_value * sn;
	int *dp_e2 = dp_e1 + numof_value * sn;
	int *dp_f1 = dp_e2 + numof_value * sn;
	int *dp_f2 = dp_f1 + numof_value * sn;
    fprintf(file, "index:%d\n", index_i);	                                                                        
    for (i = beg; i <= end; ++i) {	                                
        fprintf(file, "%d:(%d,%d,%d,%d,%d)\t", i, dp_h[i], dp_e1[i],dp_e2[i], dp_f1[i],dp_f2[i]);
    } fprintf(file, "\n");	                                                                        
}

void cuda_abpoa_cg_dp(int *DEV_DP_H2E2F, int **pre_index, int *pre_n, int index_i, abpoa_graph_t *graph, abpoa_para_t *abpt, int qlen, int w, int *CUDA_DP_H2E2F, int *cuda_dp_beg, int *cuda_dp_end, int gap_ext1, int gap_ext2, int gap_oe1, int gap_oe2, FILE *file, int *dev_dp_h, int * dev_dp_e1, int *dev_dp_e2, int *dev_dp_f1, int *dev_dp_f2, int *dev_q, int *dev_pre_dp_h, int *dev_pre_dp_e1, int *dev_pre_dp_e2, int *band, uint8_t *dev_backtrack, int *backtrack_size, int *cuda_accumulate_beg, uint8_t *e_source_order, uint8_t *e_source_order2, uint8_t *m_source_order, uint8_t *e1e2f1f2_extension) {
    int i, pre_i, node_id = abpoa_graph_index_to_node_id(graph, index_i);
    /* int min_pre_beg_sn, max_pre_end_sn, beg, end, beg_sn, end_sn, pre_end, pre_end_sn, pre_beg_sn, sn_i; */
	// the minimum position of the begin of the pre_nodes
	int min_pre_beg;
	// the maxmum position of the end of the pre_nodes
	int max_pre_end;
	// the begin position of the band of the current node
	int cuda_beg;
	// the end position of the band of the current node
	int cuda_end;
	// the end position of the band of one of the pre_nodes
	int cuda_pre_end;
	// the begin position of the band of one of the pre_nodes
	int cuda_pre_beg;
	// the end of the band of the current node 
	int _cuda_end;
	// the begin of the band of the current node 
	int _cuda_beg;
	// gap is the distance of _cuda_beg and cuda_beg
	int gap;
    if (abpt->wb < 0) {
		// no band
        cuda_beg = cuda_dp_beg[index_i] = 0, cuda_end = cuda_dp_end[index_i] = qlen;
        /* beg_sn = dp_beg_sn[index_i] = beg/pn; end_sn = dp_end_sn[index_i] = end/pn; */
        min_pre_beg = 0, max_pre_end = qlen;
    } else {
		// compute the beg and end
        cuda_beg = GET_AD_DP_BEGIN(graph, w, node_id, qlen), cuda_end = GET_AD_DP_END(graph, w, node_id, qlen);
		/* if (index_i == 154) { */
		/*     fprintf(stderr, "abpoa_graph_node_id_to_max_pos_left:%d\n", abpoa_graph_node_id_to_max_pos_left(graph, node_id)); */
		/*     fprintf(stderr, "abpoa_graph_node_id_to_max_remain:%d\n", abpoa_graph_node_id_to_max_remain(graph, node_id)); */
		/* } */
		// simdtest
        /* beg_sn = beg / pn; min_pre_beg_sn = INT32_MAX, max_pre_end_sn = -1; */
		min_pre_beg = INT32_MAX;
		max_pre_end = -1;
		/* fprintf(file, "index:%d, beg:%d, end:%d\n", index_i, cuda_beg, cuda_end); */
        for (i = 0; i < pre_n[index_i]; ++i) {
            pre_i = pre_index[index_i][i];
            if (min_pre_beg > cuda_dp_beg[pre_i]) min_pre_beg = cuda_dp_beg[pre_i];
            if (max_pre_end < cuda_dp_end[pre_i]) max_pre_end = cuda_dp_end[pre_i];
        } if (cuda_beg < min_pre_beg) cuda_beg = min_pre_beg; // beg should be never lower than mix_pre_beg

		/* fprintf(file, "min_pre%d\n", min_pre_beg); */
		cuda_beg = cuda_beg / 8 * 8;
		cuda_end = cuda_end / 8 * 8 + 7;
		/* fprintf(file, "index:%d, beg:%d, end:%d\n", index_i, cuda_beg, cuda_end); */

		// record the begin and end position of the current node.
        cuda_dp_beg[index_i] = cuda_beg;         
		cuda_end = cuda_dp_end[index_i] = cuda_end;     
	}
    pre_i = pre_index[index_i][0];

	dev_pre_dp_h = DEV_DP_H2E2F + pre_i * 5 * (qlen + 1);
	dev_pre_dp_e1 = dev_pre_dp_h + qlen + 1;
	dev_pre_dp_e2 = dev_pre_dp_e1 + qlen + 1;
	
    cuda_pre_end = cuda_dp_end[pre_i]; cuda_pre_beg = cuda_dp_beg[pre_i]; 
	int threads_per_block; 
	int blocks_per_grid;

	int flag;
	int _cuda_beg_m;
	if (cuda_pre_beg < cuda_beg) {
		flag = 1;
		_cuda_beg = cuda_beg;
		_cuda_beg_m = cuda_beg;
	}
	else {
		flag = 0;
		_cuda_beg = cuda_pre_beg;
		_cuda_beg_m = cuda_pre_beg + 1;
	}

	band[index_i] = cuda_end - cuda_beg + 1;
	
	// 纵坐标最大是qlen，共（qlen+1）个元素
	_cuda_end = MIN_OF_THREE(cuda_pre_end + 1, cuda_end, qlen);

	threads_per_block = 512;
	blocks_per_grid = (cuda_end - cuda_beg + 1) + threads_per_block - 1 / threads_per_block;
	/* cuda_set_M_first_node<<<blocks_per_grid, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, cuda_beg, cuda_end, _cuda_beg, _cuda_end, INF_MIN, flag); */
	/* cuda_set_M_first_node_nb<<<3, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, m_source_order, cuda_beg, cuda_end, _cuda_beg_m, _cuda_end, INF_MIN, flag); */

	// assemble E and M first node
	_cuda_end = MIN_OF_TWO(cuda_pre_end, cuda_end);
	cuda_set_ME_first_node_nb<<<3, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, m_source_order,dev_dp_e1, dev_pre_dp_e1,dev_dp_e2, dev_pre_dp_e2, e_source_order, e_source_order2, cuda_beg, cuda_end, _cuda_beg, _cuda_end, INF_MIN, flag);

	_cuda_end = MIN_OF_TWO(cuda_pre_end, cuda_end);
	// ***************************************
	/* fprintf(file, "beg:%d\tend:%d\tband:%d\n", cuda_beg, cuda_end, cuda_end - cuda_beg); */
	/* if (index_i == 3288) { */
	/*     int *result1 = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(file, index_i, result1, cuda_beg, cuda_end, qlen); */
	/*     free(result1); */
	/* } */

	/* cuda_set_E_first_node<<<blocks_per_grid, threads_per_block>>>(dev_dp_e1, dev_pre_dp_e1, cuda_beg, cuda_end, _cuda_beg, _cuda_end, INF_MIN); */
	/* cuda_set_E_first_node<<<blocks_per_grid, threads_per_block>>>(dev_dp_e2, dev_pre_dp_e2, cuda_beg, cuda_end, _cuda_beg, _cuda_end, INF_MIN); */
	/* cuda_set_E_first_node_nb<<<3, threads_per_block>>>(dev_dp_e1, dev_pre_dp_e1,dev_dp_e2, dev_pre_dp_e2, e_source_order, e_source_order2, cuda_beg, cuda_end, _cuda_beg, _cuda_end, INF_MIN); */
	/* if (index_i == 2223) { */
	/*     fprintf(stderr, "prei:%d\n", pre_i); */
	/* } */
	
	/* get max m and e */
	for (i = 1; i < pre_n[index_i]; ++i) {
		pre_i = pre_index[index_i][i];
	/* if (index_i == 2223) { */
	/*     fprintf(stderr, "prei:%d\n", pre_i); */
	/* } */

		cuda_pre_end = cuda_dp_end[pre_i];
		cuda_pre_beg = cuda_dp_beg[pre_i];

		dev_pre_dp_h = DEV_DP_H2E2F + (pre_i * 5) * (qlen + 1);
		dev_pre_dp_e1 = dev_pre_dp_h + qlen + 1;
		dev_pre_dp_e2 = dev_pre_dp_e1 + qlen + 1;

		/* dev_dp_f1 = dev_dp_e2 + qlen + 1; */
		/* dev_dp_f2 = dev_dp_f1 + qlen + 1; */
		
		// 纵坐标最大是qlen，共（qlen+1）个元素
		_cuda_end = MIN_OF_THREE(cuda_pre_end + 1, cuda_end, qlen);
		// set cuda_dp_h[_beg] separately
		if (cuda_pre_beg < cuda_beg) {
			flag = 1;
			_cuda_beg = cuda_beg;
		} else {
			flag = 0;
			_cuda_beg = cuda_pre_beg;
		}
		gap = _cuda_beg - cuda_beg;
		// cuda parallelization
		threads_per_block = 256;
		blocks_per_grid = (_cuda_end - _cuda_beg + 1) + threads_per_block - 1 / threads_per_block;
		/* cuda_set_M<<<3, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, m_source_order,_cuda_beg, _cuda_end, flag, i, gap); */

		_cuda_end = MIN_OF_TWO(cuda_pre_end, cuda_end);
		cuda_set_ME<<<3, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, m_source_order,dev_dp_e1, dev_pre_dp_e1, dev_dp_e2, dev_pre_dp_e2, e_source_order, e_source_order2,_cuda_beg, _cuda_end, flag, i, gap);

		_cuda_end = MIN_OF_TWO(cuda_pre_end, cuda_end);
		/* cuda_set_E<<<blocks_per_grid, threads_per_block>>>(dev_dp_e1, dev_pre_dp_e1, _cuda_beg, _cuda_end); */
		/* cuda_set_E<<<blocks_per_grid, threads_per_block>>>(dev_dp_e2, dev_pre_dp_e2, _cuda_beg, _cuda_end); */

		/* cuda_set_E_nb<<<1, threads_per_block>>>(dev_dp_e1, dev_pre_dp_e1, _cuda_beg, _cuda_end); */
		/* cuda_set_E_nb<<<1, threads_per_block>>>(dev_dp_e2, dev_pre_dp_e2, _cuda_beg, _cuda_end); */
		/* cuda_set_E_nb2<<<1, threads_per_block>>>(dev_dp_e1, dev_pre_dp_e1, dev_dp_e2, dev_pre_dp_e2, e_source_order, e_source_order2, _cuda_beg, _cuda_end, i, gap); */
	}
	threads_per_block = 512;


	// 31-33test
	/* int *result = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/* if (index_i == 20) { */
	/*     checkCudaErrors(cudaMemcpy(result, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(file, index_i, result, cuda_beg, cuda_end, qlen); */
	/*     free(result); */
	/* } */

	blocks_per_grid = (cuda_end - cuda_beg + 1) + threads_per_block - 1 / threads_per_block;
	/* cuda_add<<<3, threads_per_block>>>(dev_dp_h, dev_q, cuda_beg, cuda_end); */

	// 31-33test
	/* result = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/* if (index_i == 20) { */
	/*     checkCudaErrors(cudaMemcpy(result, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(file, index_i, result, cuda_beg, cuda_end, qlen); */
	/*     free(result); */
	/* } */


	/* cuda_suboe1_shift_right_one<<<blocks_per_grid, threads_per_block>>>(dev_dp_f1, dev_dp_h, gap_oe1, cuda_beg, cuda_end, INF_MIN, max_pre_end); */
	/* cuda_suboe1_shift_right_one<<<blocks_per_grid, threads_per_block>>>(dev_dp_f2, dev_dp_h, gap_oe2, cuda_beg, cuda_end, INF_MIN, max_pre_end); */
	/* cuda_suboe1_shift_right_one<<<10, threads_per_block>>>(dev_dp_f1, dev_dp_h, gap_oe1, cuda_beg, cuda_end, INF_MIN, max_pre_end); */
	/* cuda_suboe1_shift_right_one<<<10, threads_per_block>>>(dev_dp_f2, dev_dp_h, gap_oe2, cuda_beg, cuda_end, INF_MIN, max_pre_end); */
	int temp_max_pre_end;
	if (max_pre_end + 2 >= cuda_end) {
		temp_max_pre_end = cuda_end - 2;
	} else {
		temp_max_pre_end = max_pre_end;
	}
	cuda_add_and_suboe1_shift_right_one_nb<<<3, threads_per_block>>>(dev_dp_f2,dev_dp_f1, dev_dp_h, gap_oe2, gap_oe1, dev_q, cuda_beg, cuda_end, INF_MIN, temp_max_pre_end);
	/* cuda_suboe1_shift_right_one_nb<<<10, threads_per_block>>>(dev_dp_f2,dev_dp_f1, dev_dp_h, gap_oe2, gap_oe1, cuda_beg, cuda_end, INF_MIN, temp_max_pre_end); */

	// 31-33test
	/* if (index_i == 52) { */
	/*     int *result1 = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(stderr, index_i, result1, cuda_beg, cuda_end, qlen); */
	/*     free(result1); */
	/* } */

	int cov_bit;
	cov_bit = MIN_OF_THREE(max_pre_end + 1, cuda_end, qlen) + 1;
	

	int len = cuda_end - cuda_beg + 1;

	int F_block_dim = 3;
	int F_threads_per_block = 256;
	/* int F_threads_per_block = (len - 1) / 256 * 256 + 256; */

	/* int F_threads_per_block = 256; */
	/* int F_block_dim = (len - 1) / 256 + 1; */

	/* cuda_set_F<<<1, F_threads_per_block>>>(dev_dp_f1, gap_ext1, cuda_beg, cuda_end, cov_bit, max_pre_end); */
	/* cuda_set_F<<<1, F_threads_per_block>>>(dev_dp_f2, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end); */

	int logn = (int)log2(F_block_dim * F_threads_per_block);
	switch(logn) {
		case 1:
			cuda_set_F_nb<1><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 2:
			cuda_set_F_nb<2><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 3:
			cuda_set_F_nb<3><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 4:
			cuda_set_F_nb<4><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 5:
			cuda_set_F_nb<5><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 6:
			cuda_set_F_nb<6><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 7:
			cuda_set_F_nb<7><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 8:
			cuda_set_F_nb<8><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 9:
			cuda_set_F_nb<9><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
		case 10:
			cuda_set_F_nb<10><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end);break;
	}
	/* F_threads_per_block = 256; */
	/* cuda_set_F_nb2<<<1, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end); */

	/* if (index_i == 52) { */
	/*     int *result1 = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(stderr, index_i, result1, cuda_beg, cuda_end, qlen); */
	/*     free(result1); */
	/* } */

	cuda_get_max_F_and_set_E<<<10, threads_per_block>>>(dev_dp_h, dev_dp_e1, dev_dp_e2, dev_dp_f1, dev_dp_f2, e_source_order, e_source_order2, e1e2f1f2_extension, cuda_beg, cuda_end, gap_oe1, gap_oe2, gap_ext1, gap_ext2, INF_MIN, dev_backtrack);
	/* if (index_i == 2) { */
	/*     uint8_t *result1 = (uint8_t *)malloc(sizeof(uint8_t) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_backtrack, (qlen + 1) * 5 * sizeof(uint8_t), cudaMemcpyDeviceToHost)); */
	/*     for (int i = 0; i < cuda_end - cuda_beg + 1; i++) { */
	/*         fprintf(stderr, "(%d,%d)\t", i, result1[i]); */
	/*     } */
	/*     free(result1); */
	/* } */
	cuda_accumulate_beg[index_i] = *backtrack_size + 1;
	*backtrack_size += cuda_end - cuda_beg + 1;
	/* if (index_i == 1007) { */
	/*     int *result1 = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(stderr, index_i, result1, cuda_beg, cuda_end, qlen); */
	/*     free(result1); */
	/* } */
	/* checkCudaErrors(cudaEventRecord(end, 0)); */
	/* checkCudaErrors(cudaEventSynchronize(end)); */
	/* checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end)); */
	/* fprintf(stderr, "F_time2: %lf ms\n", elapsedTime); */


	/* int *result = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/* checkCudaErrors(cudaMemcpy(result, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/* cuda_print_row(file, index_i, result, cuda_beg, cuda_end, qlen); */
}

int abpoa_cg_dp(SIMDi *q, SIMDi *dp_h, SIMDi *dp_e1, SIMDi *dp_e2, SIMDi *dp_f1, SIMDi *dp_f2, int **pre_index, int *pre_n, int index_i, abpoa_graph_t *graph, abpoa_para_t *abpt, int dp_sn, int pn, int qlen, int w, SIMDi *DP_H2E2F, SIMDi SIMD_INF_MIN, SIMDi GAP_O1, SIMDi GAP_O2, SIMDi GAP_E1, SIMDi GAP_E2, SIMDi GAP_OE1, SIMDi GAP_OE2, SIMDi* GAP_E1S, SIMDi* GAP_E2S, SIMDi *PRE_MIN, SIMDi *PRE_MASK, SIMDi *SUF_MIN, int log_n, int *dp_beg, int *dp_end, int *dp_beg_sn, int *dp_end_sn) {
    int tot_dp_sn = 0, i, pre_i, node_id = abpoa_graph_index_to_node_id(graph, index_i);
    int min_pre_beg_sn, max_pre_end_sn, beg, end, beg_sn, end_sn, pre_end, pre_end_sn, pre_beg_sn, sn_i;
    if (abpt->wb < 0) {
        beg = dp_beg[index_i] = 0, end = dp_end[index_i] = qlen;
        beg_sn = dp_beg_sn[index_i] = beg/pn; end_sn = dp_end_sn[index_i] = end/pn;
        min_pre_beg_sn = 0, max_pre_end_sn = end_sn;
    } else {
        beg = GET_AD_DP_BEGIN(graph, w, node_id, qlen), end = GET_AD_DP_END(graph, w, node_id, qlen);
        beg_sn = beg / pn; min_pre_beg_sn = INT32_MAX, max_pre_end_sn = -1;
        for (i = 0; i < pre_n[index_i]; ++i) {
            pre_i = pre_index[index_i][i];
            if (min_pre_beg_sn > dp_beg_sn[pre_i]) min_pre_beg_sn = dp_beg_sn[pre_i];
            if (max_pre_end_sn < dp_end_sn[pre_i]) max_pre_end_sn = dp_end_sn[pre_i];
        } if (beg_sn < min_pre_beg_sn) beg_sn = min_pre_beg_sn;
        dp_beg_sn[index_i] = beg_sn; beg = dp_beg[index_i] = dp_beg_sn[index_i] * pn;
        end_sn = dp_end_sn[index_i] = end/pn; end = dp_end[index_i] = (dp_end_sn[index_i]+1)*pn-1;
    }
    tot_dp_sn += (end_sn - beg_sn + 1);
    /* loop query */
    // new init start
    int _beg_sn, _end_sn;
    // first pre_node
    pre_i = pre_index[index_i][0];
    SIMDi *pre_dp_h = DP_H2E2F + (pre_i * 5) * dp_sn; SIMDi *pre_dp_e1 = pre_dp_h + dp_sn; SIMDi *pre_dp_e2 = pre_dp_e1 + dp_sn;
    pre_end = dp_end[pre_i]; pre_beg_sn = dp_beg_sn[pre_i]; pre_end_sn = dp_end_sn[pre_i];
    SIMDi first, remain;
    /* set M from (pre_i, q_i-1) */
    if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneNi32);
    else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneNi32);
    _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);
    for (i = beg_sn; i < _beg_sn; ++i) dp_h[i] = SIMD_INF_MIN;
    for (i = _end_sn+1; i <= MIN_OF_TWO(end_sn+1, dp_sn-1); ++i) dp_h[i] = SIMD_INF_MIN;
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */
        remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneNi32);
        dp_h[sn_i] = SIMDOri(first, remain);
        first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneNi32);
    }
    /* set E from (pre_i, q_i) */
    _end_sn = MIN_OF_TWO(pre_end_sn, end_sn);
    for (i = beg_sn; i < _beg_sn; ++i) dp_e1[i] = SIMD_INF_MIN, dp_e2[i] = SIMD_INF_MIN;                                        \
    for (i = _end_sn+1; i <= end_sn; ++i) dp_e1[i] = SIMD_INF_MIN, dp_e2[i] = SIMD_INF_MIN;                                     \
    for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */
        dp_e1[sn_i] = pre_dp_e1[sn_i];                                                                    
        dp_e2[sn_i] = pre_dp_e2[sn_i];                                                                    
    }
    // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("1", int32_t, index_i);
    // new init end
    /* get max m and e */
    for (i = 1; i < pre_n[index_i]; ++i) {
        pre_i = pre_index[index_i][i];
        pre_dp_h = DP_H2E2F + (pre_i * 5) * dp_sn; pre_dp_e1 = pre_dp_h + dp_sn; pre_dp_e2 = pre_dp_e1 + dp_sn;
        pre_end = dp_end[pre_i]; pre_beg_sn = dp_beg_sn[pre_i]; pre_end_sn = dp_end_sn[pre_i];
        /* set M from (pre_i, q_i-1) */
        if (pre_beg_sn < beg_sn) _beg_sn = beg_sn, first = SIMDShiftRight(pre_dp_h[beg_sn-1], SIMDTotalBytes-SIMDShiftOneNi32);
        else _beg_sn = pre_beg_sn, first = SIMDShiftRight(SIMD_INF_MIN, SIMDTotalBytes-SIMDShiftOneNi32);
        _end_sn = MIN_OF_THREE((pre_end+1)/pn, end_sn, dp_sn-1);
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */
            remain = SIMDShiftLeft(pre_dp_h[sn_i], SIMDShiftOneNi32);
            dp_h[sn_i] = SIMDMaxi32(SIMDOri(first, remain), dp_h[sn_i]);
            first = SIMDShiftRight(pre_dp_h[sn_i], SIMDTotalBytes-SIMDShiftOneNi32);
        }
        /* set E from (pre_i, q_i) */
        _end_sn = MIN_OF_TWO(pre_end_sn, end_sn);
        for (sn_i = _beg_sn; sn_i <= _end_sn; ++sn_i) { /* SIMD parallelization */
            dp_e1[sn_i] = SIMDMaxi32(pre_dp_e1[sn_i], dp_e1[sn_i]);
            dp_e2[sn_i] = SIMDMaxi32(pre_dp_e2[sn_i], dp_e2[sn_i]);
        }
    }
    // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("2", int32_t, index_i);
    /* if (index_i == 4) debug_simd_abpoa_print_cg_matrix_row("11111", int32_t, index_i); */
	/* if (index_i == 4) fprintf(stderr, "max_pre_end_sn:%d\n", max_pre_end_sn); */
    /* compare M, E, and F */
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) { /* SIMD parallelization */
        dp_h[sn_i] =  SIMDAddi32(dp_h[sn_i], q[sn_i]);
    }
    /* if (index_i == 4) debug_simd_abpoa_print_cg_matrix_row("11111", int32_t, index_i); */
    // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("3", int32_t, index_i);
    /* new F start */
    first = SIMDShiftRight(SIMDShiftLeft(dp_h[beg_sn], SIMDTotalBytes-SIMDShiftOneNi32), SIMDTotalBytes-SIMDShiftOneNi32); 
    SIMDi first2 = first, tmp; int set_num;
    for (sn_i = beg_sn; sn_i <= end_sn; ++sn_i) {
        if (sn_i < min_pre_beg_sn) {
            _err_fatal_simple(__func__, "sn_i < min_pre_beg_sn\n");
        } else if (sn_i > max_pre_end_sn) {
            set_num = sn_i == max_pre_end_sn+1 ? 2 : 1;
        } else set_num = pn;
        /* F = (H << 1 | x) - OE */
        // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("4.1", int32_t, index_i);
        dp_f1[sn_i] = SIMDSubi32(SIMDOri(SIMDShiftLeft(dp_h[sn_i], SIMDShiftOneNi32), first), GAP_OE1);
        dp_f2[sn_i] = SIMDSubi32(SIMDOri(SIMDShiftLeft(dp_h[sn_i], SIMDShiftOneNi32), first2), GAP_OE2);
        /* F = max{F, (F-e)<<1}, F = max{F, (F-2e)<<2} ... */
        // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("4.2", int32_t, index_i);
        SIMD_SET_F(dp_f1[sn_i], log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E1S, SIMDMaxi32, SIMDAddi32, SIMDSubi32, SIMDShiftOneNi32);
        SIMD_SET_F(dp_f2[sn_i], log_n, set_num, PRE_MIN, PRE_MASK, SUF_MIN, GAP_E2S, SIMDMaxi32, SIMDAddi32, SIMDSubi32, SIMDShiftOneNi32);
        /* x = max{H, F+o} */
        // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("4.3", int32_t, index_i);
        first = SIMDShiftRight(SIMDMaxi32(dp_h[sn_i], SIMDAddi32(dp_f1[sn_i], GAP_O1)), SIMDTotalBytes-SIMDShiftOneNi32);
        first2 = SIMDShiftRight(SIMDMaxi32(dp_h[sn_i], SIMDAddi32(dp_f2[sn_i], GAP_O2)), SIMDTotalBytes-SIMDShiftOneNi32);
        /* H = max{H, F}    */
        dp_h[sn_i] = SIMDMaxi32(SIMDMaxi32(dp_h[sn_i], dp_e1[sn_i]), dp_e2[sn_i]); tmp = dp_h[sn_i];
        dp_h[sn_i] = SIMDMaxi32(SIMDMaxi32(dp_h[sn_i], dp_f1[sn_i]), dp_f2[sn_i]);
        // if (index_i == 13095) debug_simd_abpoa_print_cg_matrix_row("4.4", int32_t, index_i);
        /* e for next cell */
        SIMDSetIfEquali32(dp_e1[sn_i], dp_h[sn_i], tmp, SIMDMaxi32(SIMDSubi32(dp_e1[sn_i], GAP_E1), SIMDSubi32(dp_h[sn_i], GAP_OE1)), SIMD_INF_MIN);
        SIMDSetIfEquali32(dp_e2[sn_i], dp_h[sn_i], tmp, SIMDMaxi32(SIMDSubi32(dp_e2[sn_i], GAP_E2), SIMDSubi32(dp_h[sn_i], GAP_OE2)), SIMD_INF_MIN);
    }
    return tot_dp_sn;
}

void cuda_my_abpoa_cg_backtrack(uint8_t *backtrack_matrix, uint8_t *e_source_order1, uint8_t *e_source_order2, uint8_t *m_source_order, uint8_t *e1e2f1f2_extension, int *cuda_accumulate_beg, int **pre_index, int *pre_n, int *dp_beg, int m, int *mat, int start_i, int start_j, int best_i, int best_j, int qlen, abpoa_graph_t *graph, abpoa_para_t *abpt, uint8_t *query, abpoa_res_t *res) {
    int i, j, k, pre_i, n_c = 0, s, m_c = 0, hit, cur_op = ABPOA_ALL_OP, _start_i, _start_j;
	abpoa_cigar_t *cigar = 0;
    i = best_i, j = best_j, _start_i = best_i, _start_j = best_j;
    int id = abpoa_graph_index_to_node_id(graph, i);
    if (best_j < qlen) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, qlen-j, -1, qlen-1);
	int backtrack_beg, gap; // index for backtrack matrix
	int f1,f2,e1,e2;
	char file_name[] = "backtrack";
	FILE *file = fopen(file_name, "w");
    while (i > start_i && j > start_j) {
        _start_i = i, _start_j = j;
        s = mat[m * graph->node[id].base + query[j-1]]; hit = 0;
		backtrack_beg = cuda_accumulate_beg[i];
		gap = j - dp_beg[i];
		if (backtrack_matrix[backtrack_beg + gap] == 0) {
			cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CMATCH, 1, id, j-1);
			k = m_source_order[backtrack_beg + gap];
			pre_i = pre_index[i][k];
			i = pre_i; --j; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
			++res->n_aln_bases; res->n_matched_bases += s == mat[0] ? 1 : 0;
		}
		// f1
        if (hit == 0 && backtrack_matrix[backtrack_beg + gap] == 1) {
			f1 = (e1e2f1f2_extension[backtrack_beg + gap] >> 1) & 1;
			while (f1 != 0) {
				cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1);
				++res->n_aln_bases;
				j--;
				gap--;
				f1 = (e1e2f1f2_extension[backtrack_beg + gap] >> 1) & 1;
			}
			cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1);
			++res->n_aln_bases;
			j--;
			gap--;
			/* f1 = (e1e2f1f2_extension[backtrack_beg + gap] >> 1) & 1; */
			hit = 1;
		}
		// f2
        if (hit == 0 && backtrack_matrix[backtrack_beg + gap] == 2) {
			f2 = (e1e2f1f2_extension[backtrack_beg + gap]) & 1;
			while (f2 != 0) {
				cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1);
				++res->n_aln_bases;
				j--;
				gap--;
				f2 = (e1e2f1f2_extension[backtrack_beg + gap]) & 1;
			}
			cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1);
			++res->n_aln_bases;
			j--;
			gap--;
			/* f2 = (e1e2f1f2_extension[backtrack_beg + gap] >> 1) & 1; */
			hit = 1;
		}
		// e1
        if (hit == 0 && backtrack_matrix[backtrack_beg + gap] == 3) {
			e1 = (e1e2f1f2_extension[backtrack_beg + gap] >> 3) & 1;
			while (e1 != 1) {
				k = e_source_order1[backtrack_beg + gap];
				pre_i = pre_index[i][k];
				cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);
				i = pre_i; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
				backtrack_beg = cuda_accumulate_beg[i];
				gap = j - dp_beg[i];
				e1 = (e1e2f1f2_extension[backtrack_beg + gap] >> 3) & 1;
			}
		}
		// e2
        if (hit == 0 && backtrack_matrix[backtrack_beg + gap] == 4) {
			e2= (e1e2f1f2_extension[backtrack_beg + gap] >> 2) & 1;
			while (e2!= 1) {
				k = e_source_order2[backtrack_beg + gap];
				pre_i = pre_index[i][k];
				cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);
				i = pre_i; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
				backtrack_beg = cuda_accumulate_beg[i];
				gap = j - dp_beg[i];
				e2= (e1e2f1f2_extension[backtrack_beg + gap] >> 2) & 1;
			}
		}

		if (hit == 0) fprintf(stderr, "i:%d\tj:%d,backtrack:%d\t", i, j, backtrack_matrix[backtrack_beg + gap]), exit(1);
	} 

		/* fprintf(file, "%d, %d, %d\n", i, j, cur_op); */
	fclose(file);
    if (j > start_j) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, j-start_j, -1, j-1);
    /* reverse cigar */
    res->graph_cigar = abpt->rev_cigar ? cigar : abpoa_reverse_cigar(n_c, cigar);
    res->n_cigar = n_c;
    res->node_e = best_i, res->query_e = best_j-1; /* 0-based */
    res->node_s = _start_i, res->query_s = _start_j-1;
	/* abpoa_print_cigar(n_c, res->graph_cigar, graph); */
	/* abpoa_print_cigar(n_c, *graph_cigar, graph); */
}

void cuda_abpoa_cg_backtrack(int  *CUDA_DP_H2E2F, int **pre_index, int *pre_n, int *dp_beg, int *dp_end, int m, int *mat, int gap_ext1, int gap_ext2, int gap_oe1, int gap_oe2, int start_i, int start_j, int best_i, int best_j, int qlen, abpoa_graph_t *graph, abpoa_para_t *abpt, uint8_t *query, abpoa_res_t *res) {
    int i, j, k, pre_i, n_c = 0, s, m_c = 0, id, hit, cur_op = ABPOA_ALL_OP, _start_i, _start_j;
    int *dp_h=NULL, *dp_e1, *dp_e2, *dp_f1, *dp_f2; abpoa_cigar_t *cigar = 0;
	int *pre_dp_h, *pre_dp_e1, *pre_dp_e2;
    i = best_i, j = best_j, _start_i = best_i, _start_j = best_j;
    id = abpoa_graph_index_to_node_id(graph, i);
    if (best_j < qlen) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, qlen-j, -1, qlen-1);
    dp_h = CUDA_DP_H2E2F + (qlen + 1) * (i * 5);
	int n, add_e, sum;
	char file_name[] = "backtrack";
	FILE *file = fopen(file_name, "w");
    while (i > start_i && j > start_j) {
        _start_i = i, _start_j = j;
        int *pre_index_i = pre_index[i];
        s = mat[m * graph->node[id].base + query[j-1]]; hit = 0;
        if (cur_op & ABPOA_M_OP) {
            for (k = 0; k < pre_n[i]; ++k) {
                pre_i = pre_index_i[k];
                if (j-1 < dp_beg[pre_i] || j-1 > dp_end[pre_i]) continue;
                pre_dp_h = (int32_t*)(CUDA_DP_H2E2F + (qlen + 1) * (pre_i * 5));
                if (pre_dp_h[j-1] + s ==dp_h[j]) { /* match/mismatch */
                    cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CMATCH, 1, id, j-1);
                    i = pre_i; --j; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
                    dp_h = CUDA_DP_H2E2F + (qlen + 1) * (i * 5);
                    cur_op = ABPOA_ALL_OP;
                    ++res->n_aln_bases; res->n_matched_bases += s == mat[0] ? 1 : 0;
                    break;
                }
            }
        }
        if (hit == 0 && cur_op & ABPOA_E_OP) {
           dp_e1 = (int32_t*)(dp_h+(qlen + 1)),dp_e2 = (int32_t*)(dp_h+(qlen + 1)*2);
            for (k = 0; k < pre_n[i]; ++k) {
                pre_i = pre_index_i[k];
                if (j < dp_beg[pre_i] || j > dp_end[pre_i]) continue;
                pre_dp_h = (int32_t*)(CUDA_DP_H2E2F + (qlen + 1) * (pre_i * 5));
                if (cur_op & ABPOA_E1_OP) {
                    pre_dp_e1 = (int32_t*)(CUDA_DP_H2E2F + (qlen + 1) * ((pre_i * 5) + 1));
                    if (cur_op & ABPOA_M_OP) {
                        if (dp_h[j] == pre_dp_e1[j]) {
                            if (dp_h[j] == pre_dp_h[j] - gap_oe1) {
								cur_op = ABPOA_M_OP;
								/* if (dp_h[j] != pre_dp_h[j] - gap_oe1) fprintf(stderr, "yes\n"); */
							}
                            else {
								/* if (dp_h[j] != pre_dp_h[j] - gap_ext1) { */
								/*     fprintf(stderr, "yes\n"); */
								/*     if (dp_e1[j] == pre_dp_e1[j] - gap_ext1) */
								/*     fprintf(stderr, "yes2\n"); */
								/* } */
								cur_op = ABPOA_E1_OP;
							}
                            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
                            dp_h = CUDA_DP_H2E2F + (qlen + 1) * (i * 5);
                            break;
                        }
                    } else {
                        if (dp_e1[j] == pre_dp_e1[j] - gap_ext1) {
                            if (pre_dp_h[j] - gap_oe1 == pre_dp_e1[j]) cur_op = ABPOA_M_OP;
                            else cur_op = ABPOA_E1_OP;
                            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
                            dp_h = CUDA_DP_H2E2F + (qlen + 1) * (i * 5);
                            break;
                        }
                    }
                }
                if (cur_op & ABPOA_E2_OP) {
                    pre_dp_e2 = (int32_t*)(CUDA_DP_H2E2F + (qlen + 1) * ((pre_i * 5) + 2));
                    if (cur_op & ABPOA_M_OP) {
                        if (dp_h[j] == pre_dp_e2[j]) {
                            if (pre_dp_h[j] - gap_oe2 == pre_dp_e2[j]) cur_op = ABPOA_M_OP;
                            else cur_op = ABPOA_E2_OP;
                            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
                            dp_h = CUDA_DP_H2E2F + (qlen + 1) * (i * 5);
                            break;
                        }
                    } else {
                        if (dp_e2[j] == pre_dp_e2[j] - gap_ext2) {
                            if (pre_dp_h[j] - gap_oe2 == pre_dp_e2[j]) cur_op = ABPOA_M_OP;
                            else cur_op = ABPOA_E2_OP;
                            cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CDEL, 1, id, j-1);
                            i = pre_i; id = abpoa_graph_index_to_node_id(graph, i); hit = 1;
                            dp_h = CUDA_DP_H2E2F + (qlen + 1) * (i * 5);
                            break;
                        }
                    }
                }
            }
        }
        if (hit == 0 && cur_op & ABPOA_F_OP) {
            if (cur_op & ABPOA_F1_OP) {
			   dp_f1 = (int32_t*)(dp_h + (qlen + 1) * 3);
				if (cur_op & ABPOA_M_OP) {
					if (dp_h[j] ==dp_f1[j]) {
						if (dp_h[j-1] - gap_oe1 ==dp_h[j]) cur_op = ABPOA_M_OP, hit = 1;
						else if (dp_f1[j-1] - gap_ext1 ==dp_f1[j]) cur_op = ABPOA_F1_OP, hit =1;
						else exit(1);
					}
				} else {
					if (dp_f1[j-1] - gap_ext1 ==dp_f1[j]) cur_op = ABPOA_F1_OP, hit =1;
					else if (dp_h[j-1] - gap_oe1 ==dp_f1[j]) cur_op = ABPOA_M_OP, hit = 1;
					else exit(1);
				}
				sum = 1;

				/* n = j; */
				/* sum = 0; */
				/* add_e = gap_oe1; */
				/* while (n > dp_beg[i]) { */
				/*     sum++; */
				/*     if (dp_h[n - 1] - add_e == dp_h[j]) { */
				/*         cur_op = abpoa_m_op, hit =1; */
				/*         break; */
				/*     } */
				/*     add_e += gap_ext1; */
				/*     n--; */
				/* } */
            }
            if (hit == 0 && cur_op & ABPOA_F2_OP) {
               dp_f2 = (int32_t*)(dp_h + (qlen + 1) * 4);
                if (cur_op & ABPOA_M_OP) {
                    if (dp_h[j] ==dp_f2[j]) {
                        if (dp_h[j-1] - gap_oe2 ==dp_f2[j]) cur_op = ABPOA_M_OP, hit = 1;
                        else if (dp_f2[j-1] - gap_ext2 ==dp_f2[j]) cur_op = ABPOA_F2_OP, hit =1;
                        else exit(1);
                    }
                } else {
                    if (dp_f2[j-1] - gap_ext2 ==dp_f2[j]) cur_op = ABPOA_F2_OP, hit =1;
                    else if (dp_h[j-1] - gap_oe2 ==dp_f2[j]) cur_op = ABPOA_M_OP, hit = 1;
                    else exit(1);
                }
				sum = 1;
            }
			for (n = 0; n < sum; n++) {
				cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CINS, 1, id, j-1); --j;
				++res->n_aln_bases;
			}
            hit = 1;
		}
		if (hit == 0) fprintf(stderr, "i:%d\tj:%d\t", i, j), exit(1);
	} 

		/* fprintf(file, "%d, %d, %d\n", i, j, cur_op); */
	fclose(file);
    if (j > start_j) cigar = abpoa_push_cigar(&n_c, &m_c, cigar, ABPOA_CSOFT_CLIP, j-start_j, -1, j-1);
    /* reverse cigar */
    res->graph_cigar = abpt->rev_cigar ? cigar : abpoa_reverse_cigar(n_c, cigar);
    res->n_cigar = n_c;
    res->node_e = best_i, res->query_e = best_j-1; /* 0-based */
    res->node_s = _start_i, res->query_s = _start_j-1;
	/* abpoa_print_cigar(n_c, res->graph_cigar, graph); */
	/* abpoa_print_cigar(n_c, *graph_cigar, graph); */
}

void cuda_print_matrix(int *CUDA_DP_H2E2F, int qlen, int *dp_beg, int *dp_end, int row) {
	int i;
	int *dp_h;
	int beg, end;
    FILE *file = fopen("cuda_matrix.fa", "w"); 
	for (i = 0; i < row - 1; i++) {
		dp_h = CUDA_DP_H2E2F + i * (qlen + 1) * 5;
		/* fprintf(file, "index:%d\n", i); */
		beg = dp_beg[i];
		end = dp_end[i];
		cuda_print_row(file, i, dp_h, beg, end, qlen);
		/* for (j = beg; j <= end; j++) { */
		/*     fprintf(file, "%d:(%d,%d,%d,%d,%d)\t\n", j, dp_h[j], dp_e1[j],dp_e2[j], dp_f1[j],dp_f2[j]);\ */
		/* } */
	}
	fclose(file);
}

void find_difference(int *CUDA_DP_H2E2F, SIMDi* DP_H2E2F, int *dp_beg, int *dp_end, int matrix_row_n, int qlen, int dp_sn, char* file_name, int numof_value) {
	int i;
	int j;
	int *cuda_dp_h;
	int *dp_h;
	int flag = 0;
	FILE *file = fopen(file_name, "w");
	for (i = 0; i < matrix_row_n - 1; i++) {
		cuda_dp_h = CUDA_DP_H2E2F + 5 * (qlen + 1) * i;
		dp_h = (int *)(DP_H2E2F + dp_sn * 5 * i);
		for (j = dp_beg[i]; j <= dp_end[i]; j++) {
			if (dp_h[j] != cuda_dp_h[j]) {
				flag = 1;
				fprintf(file, "index:%d, j:%d, dp_h:%d, cuda_dp_h: %d\n", i, j, dp_h[j], cuda_dp_h[j]);
			}
		}
		if (flag == 1) {
			my_simd_abpoa_print_cg_matrix_row(dp_h, dp_sn, file, i, dp_beg[i], dp_end[i], numof_value);
			cuda_print_row(file, i, cuda_dp_h, dp_beg[i], dp_end[i], qlen);
		}
		flag = 0;
	}
	fclose(file);
}

void print_backtrack_matrix(uint8_t *backtrack_matrix, int backtrack_size, int *cuda_accumulate_beg, int matrix_row_n, FILE *backtrack_file) {
	int i, beg, j, end;
	fprintf(backtrack_file, "backtrack_size:%d\n", backtrack_size);
	for (i = 1; i < matrix_row_n - 1; i++) {
		beg = cuda_accumulate_beg[i];
		if (i != matrix_row_n - 2) {
			end = cuda_accumulate_beg[i + 1] - 1;
		} else {
			end = backtrack_size - 1;
		}
		fprintf(backtrack_file, "index:%d\t,beg:%d\n", i, beg);
		for (j = beg; j <= end; j++) {
			fprintf(backtrack_file, "(%d,%d)\t", j, backtrack_matrix[j]);
		}
		fprintf(backtrack_file, "\n");
	}
}

int cuda_abpoa_cg_global_align_sequence_to_graph_core(abpoa_t *ab, int qlen, uint8_t *query, abpoa_para_t *abpt, SIMD_para_t sp, abpoa_res_t *res) {
	abpoa_graph_t *graph = ab->abg;
	int matrix_row_n = graph->node_n;
	int **pre_index, *pre_n;
	int i, j, node_id, index_i;
	int *mat = abpt->mat; int32_t gap_ext1 = abpt->gap_ext1;
	int w = abpt->wb < 0 ? qlen : abpt->wb + (int)(abpt->wf * qlen); 
	int32_t gap_open1 = abpt->gap_open1, gap_oe1 = gap_open1 + gap_ext1;
	int32_t gap_open2 = abpt->gap_open2, gap_ext2 = abpt->gap_ext2, gap_oe2 = gap_open2 + gap_ext2;

	// cuda variable
	abpoa_cuda_matrix_t *abcm = ab->abcm;
	int * cuda_dp_beg, *cuda_dp_end;
	int cuda_beg, cuda_end;
	/* int *cuda_dp_h; */
	int *cuda_qp;
	cuda_qp = abcm->c_mem;
	int *CUDA_DP_H2E2F;
	int *DEV_DP_H2E2F;
	uint8_t *DEV_BACKTRACK, *E_SOURCE_ORDER, *E_SOURCE_ORDER2, *M_SOURCE_ORDER, *E1E2F1F2_EXTENSION;
	int *dev_qp;
	/* int *cuda_dp_e1; */
	/* int *cuda_dp_e2; */
	/* int *cuda_dp_f1; */
	/* int *cuda_dp_f2; */
	/* cuda_dp_h = (int *)malloc((qlen + 1) * sizeof(int)); */
	int *dev_max_i;
	CUDA_DP_H2E2F = cuda_qp + (qlen + 1) * abpt->m;
	dev_max_i = abcm->dev_mem; 
	/* dev_qp = abcm->dev_mem; */
	dev_qp = dev_max_i + 1;
	DEV_DP_H2E2F = dev_qp + abpt->m * (qlen + 1);
	DEV_BACKTRACK = abcm->dev_backtrack_matrix;
	E_SOURCE_ORDER = DEV_BACKTRACK + (qlen + 1) * ab->abg->node_n;
	E_SOURCE_ORDER2 = E_SOURCE_ORDER + (qlen + 1) * ab->abg->node_n;
	M_SOURCE_ORDER = E_SOURCE_ORDER2 + (qlen + 1) * ab->abg->node_n;
	E1E2F1F2_EXTENSION = M_SOURCE_ORDER + (qlen + 1) * ab->abg->node_n;
	int *cuda_accumulate_beg = abcm->cuda_accumulate_beg;
	*ab->abcm->backtrack_size = 0;

	// for SET_F mask[pn], suf_min[pn], pre_min[logN]

	cuda_abpoa_init_var(abpt, query, qlen, cuda_qp, mat, INF_MIN);

	cuda_dp_beg = abcm->cuda_dp_beg;
	cuda_dp_end = abcm->cuda_dp_end;

	/* don't need anymore */\

	/* index of pre-node */
	pre_index = (int**)_err_malloc(graph->node_n * sizeof(int*));
	pre_n = (int*)_err_malloc(graph->node_n * sizeof(int));
	for (i = 0; i < graph->node_n; ++i) {
		node_id = abpoa_graph_index_to_node_id(graph, i);
		pre_n[i] = graph->node[node_id].in_edge_n;
		pre_index[i] = (int*)_err_malloc(pre_n[i] * sizeof(int));
		for (j = 0; j < pre_n[i]; ++j)
			pre_index[i][j] = abpoa_graph_node_id_to_index(graph, graph->node[node_id].in_id[j]);
	}
	cuda_abpoa_cg_first_dp(abpt, graph, cuda_dp_beg, cuda_dp_end, qlen, w, DEV_DP_H2E2F, gap_open1, gap_ext1, gap_open2, gap_ext2, gap_oe1, gap_oe2);

	//the cuda part
		char file_name[] = "cuda_max_dp_h.fa";
		FILE *file = fopen(file_name, "w");

	int *dev_dp_h;
	int *dev_dp_e1;
	int *dev_dp_e2;
	int *dev_pre_dp_h;
	int *dev_pre_dp_e1;
	int *dev_pre_dp_e2;
	int *dev_q;
	int *dev_dp_f1;
	int *dev_dp_f2;
	u_int8_t *dev_backtrack, *e_source_order, *e_source_order2, *m_source_order, *e1e2f1f2_extension;
	int max_i ;
	int *backtrack_size = ab->abcm->backtrack_size;
	*backtrack_size = -1;
	/* checkCudaErrors(cudaMalloc((void **)&dev_max_i, sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_dp_h, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_dp_e1, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_dp_e2, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_pre_dp_h, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_pre_dp_e1, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_pre_dp_e2, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_q, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_dp_f1, (qlen + 1) * sizeof(int))); */
	/* checkCudaErrors(cudaMalloc((void **)&dev_dp_f2, (qlen + 1) * sizeof(int))); */

	checkCudaErrors(cudaMemcpy(dev_qp, cuda_qp, abpt->m * (qlen + 1) * sizeof(int), cudaMemcpyHostToDevice));
	char file_name_graph[] = "graph_max_right";
	FILE *file_graph = fopen(file_name_graph, "w");
	const char* max_position_name = "max_position_name.fa";
	FILE *file_max_position = fopen(max_position_name, "w");
	int *reduce_index;
	cudaMalloc((void **)&reduce_index, sizeof(int));
	int *out_data;
	cudaMalloc((void **)&out_data, sizeof(int));
	int *out_index;
	cudaMalloc((void **)&out_index, sizeof(int));
	int *band = (int *)malloc(matrix_row_n * sizeof(int));

	// test
	/* abpt->wb = -1; */
	for (index_i = 1; index_i < matrix_row_n-1; ++index_i) {
		node_id = abpoa_graph_index_to_node_id(graph, index_i);
		/* SIMDi *q = qp + graph->node[node_id].base * qp_sn; */
		/* cuda_dp_h = CUDA_DP_H2E2F + (index_i*5) * (qlen + 1); cuda_dp_e1 = cuda_dp_h + (qlen + 1); cuda_dp_e2 = cuda_dp_e1 + (qlen + 1); cuda_dp_f1 = cuda_dp_e2 + (qlen + 1); cuda_dp_f2 = cuda_dp_f1 + (qlen + 1); */
		dev_dp_h = DEV_DP_H2E2F + (index_i * 5) * (qlen + 1);
		dev_dp_e1 = dev_dp_h + qlen + 1;
		dev_dp_e2 = dev_dp_e1 + qlen + 1;
		dev_dp_f1 = dev_dp_e2 + qlen + 1;
		dev_dp_f2 = dev_dp_f1 + qlen + 1;
		dev_backtrack = DEV_BACKTRACK + *backtrack_size + 1;
		e_source_order = E_SOURCE_ORDER + *backtrack_size + 1;
		e_source_order2 = E_SOURCE_ORDER2 + *backtrack_size + 1;
		m_source_order = M_SOURCE_ORDER + *backtrack_size + 1;
		e1e2f1f2_extension = E1E2F1F2_EXTENSION + *backtrack_size + 1;

		/* fprintf(file_graph, "index:%\n", index_i); */
		/* fprintf(stderr, "index_i:%d\n", index_i); */
		/* fprintf(stderr, "dev_max_i:%x\n", dev_max_i); */
		/* fprintf(stderr, "max_i:%x\n", &max_i); */

		dev_q = dev_qp +  graph->node[node_id].base * (qlen + 1);
		cuda_abpoa_cg_dp(DEV_DP_H2E2F, pre_index, pre_n, index_i, graph, abpt, qlen, w, CUDA_DP_H2E2F, cuda_dp_beg, cuda_dp_end, gap_ext1, gap_ext2, gap_oe1, gap_oe2, file, dev_dp_h, dev_dp_e1, dev_dp_e2,dev_dp_f1, dev_dp_f2, dev_q, dev_pre_dp_h, dev_pre_dp_e1, dev_pre_dp_e2, band, dev_backtrack, backtrack_size, cuda_accumulate_beg, e_source_order, e_source_order2, m_source_order, e1e2f1f2_extension);
		//test
		/* if (index_i == 1) { */
		/*     uint8_t *result1 = (uint8_t *)malloc(sizeof(uint8_t) * (qlen + 1) * 5); */
		/*     checkCudaErrors(cudaMemcpy(result1, DEV_BACKTRACK, (qlen + 1) * 5 * sizeof(uint8_t), cudaMemcpyDeviceToHost)); */
		/*     cuda_beg = cuda_dp_beg[index_i];	 */
		/*     cuda_end = cuda_dp_end[index_i];	 */
		/*     for (int i = 0; i < cuda_end - cuda_beg + 1; i++) { */
		/*         fprintf(stderr, "(%d,%d)\t", i, result1[i]); */
		/*     } */
		/*     free(result1); */
		/* } */
		/* tot_dp_sn += abpoa_cg_dp(q, dp_h, dp_e1, dp_e2, dp_f1, dp_f2, pre_index, pre_n, index_i, graph, abpt, dp_sn, pn, qlen, w, DP_H2E2F, SIMD_INF_MIN, GAP_O1, GAP_O2, GAP_E1, GAP_E2, GAP_OE1, GAP_OE2, GAP_E1S, GAP_E2S, PRE_MIN, PRE_MASK, SUF_MIN, log_n, dp_beg, dp_end, dp_beg_sn, dp_end_sn); */
		if (abpt->wb >= 0) {
			cuda_beg = cuda_dp_beg[index_i];	
			cuda_end = cuda_dp_end[index_i];	

			// cpu compute max_index
			/* int *cuda_dp_h = (int *)malloc((qlen + 1) * sizeof(int)); */
			/* cudaMemcpy(cuda_dp_h, dev_dp_h, (qlen + 1) * sizeof(int), cudaMemcpyDeviceToHost); */
			/* int cpu_max_i = cuda_abpoa_max(cuda_dp_h, cuda_beg, cuda_end, qlen); */
			/* free(cuda_dp_h); */

			// unified memory
			/* cudaDeviceSynchronize(); */
			/* int max_i = cuda_abpoa_max(dev_dp_h, cuda_beg, cuda_end, qlen); */

			// gpu compute max_index
			/* dev_cuda_abpoa_max<<<1, 1>>>(dev_dp_h, cuda_beg, cuda_end, dev_max_i, inf_min); */
			/* dev_cuda_abpoa_max<<<1, 1>>>(dev_dp_h, cuda_beg, cuda_end, dev_max_i, INF_MIN); */
			/* checkCudaErrors(cudaMemcpy(&max_i, dev_max_i, sizeof(int), cudaMemcpyDeviceToHost)); */
			/* abpoa_ada_max_i(max_i, graph, node_id, file_max_position); */

			// gpu reduce compute max_index
			/* int reduce_cpu_index; */
			/* maxReduce<<<1, 32>>>(dev_dp_h, cuda_end - cuda_beg + 1, cuda_beg, cuda_end, reduce_index); */
			/* checkCudaErrors(cudaMemcpy(&reduce_cpu_index, reduce_index, sizeof(int), cudaMemcpyDeviceToHost)); */
			/* abpoa_ada_max_i(reduce_cpu_index, graph, node_id, file_max_position); */

			// gpu reduce compute max_index 2
			int reduce_cpu_index2;
			int temp_block_size = pow(2, (int)log2(cuda_end - cuda_beg + 1 - 1));
			int block_size = temp_block_size > 256 ? 256 : temp_block_size;
			switch(block_size) {
				case 512:
					reduce_max_index<512><<<1, 512>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 256:
					reduce_max_index<256><<<1, 256>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 128:
					reduce_max_index<128><<<1, 128>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 64:
					reduce_max_index<64><<<1, 64>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 32:
					reduce_max_index<32><<<1, 32>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 16:
					reduce_max_index<16><<<1, 16>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 8:
					reduce_max_index<8><<<1, 8>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 4:
					reduce_max_index<4><<<1, 4>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 2:
					reduce_max_index<2><<<1, 2>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
				case 1:
					reduce_max_index<1><<<1, 1>>>(dev_dp_h + cuda_beg, out_data, out_index, cuda_end - cuda_beg + 1);break;
			}
			checkCudaErrors(cudaMemcpy(&reduce_cpu_index2, out_index, sizeof(int), cudaMemcpyDeviceToHost));
			abpoa_ada_max_i(reduce_cpu_index2 + cuda_beg, graph, node_id, file_max_position);

			/* if (cuda_dp_h[reduce_cpu_index2 + cuda_beg] != cuda_dp_h[reduce_cpu_index]) { */
			/*     cuda_print_row_h(stderr, index_i, dev_dp_h, cuda_dp_beg[index_i], cuda_dp_end[index_i], 11); */
			/*     fprintf(stderr,"index_i:%d\n", index_i ); */
			/*     printf("index_i:%d\n", index_i); */
			/*     fprintf(stderr,"right:%d, reduce:%d\n", reduce_cpu_index2, reduce_cpu_index ); */
			/*     printf("right:%d, reduce:%d", reduce_cpu_index2, reduce_cpu_index); */
			/*     exit(0); */
			/* } */
			/* free(cuda_dp_h); */
			
		}
	}
	
	// test average band length
	/* int sum = 0; */
	/* for (i = 1; i < matrix_row_n; i++) { */
	/*     sum += band[i]; */
	/* } */
	/* fprintf(stderr, "band:%d\n", sum / (matrix_row_n - 1)); */


	fclose(file_graph);
	/* checkCudaErrors(cudaFree(dev_dp_h)); */
	/* checkCudaErrors(cudaFree(dev_dp_e1)); */
	/* checkCudaErrors(cudaFree(dev_dp_e2)); */
	/* checkCudaErrors(cudaFree(dev_dp_f1)); */
	/* checkCudaErrors(cudaFree(dev_dp_f2)); */
	/* checkCudaErrors(cudaFree(dev_q)); */
	/* checkCudaErrors(cudaFree(dev_pre_dp_h)); */
	/* checkCudaErrors(cudaFree(dev_pre_dp_e1)); */
	/* checkCudaErrors(cudaFree(dev_pre_dp_e2)); */


	// check matrix
	int matrix_size = ((qlen + 1) * matrix_row_n * 5 * sizeof(int)); // DP_H2E2F, convex
	/* checkCudaErrors(cudaMemcpy(CUDA_DP_H2E2F, DEV_DP_H2E2F, matrix_size, cudaMemcpyDeviceToHost)); */
	/* cuda_print_matrix(CUDA_DP_H2E2F, qlen, cuda_dp_beg, cuda_dp_end, matrix_row_n); */

	/* uint64_t temp_msize = (qlen + 1) * matrix_row_n * sizeof(uint8_t); // for backtrack matrix */
	uint8_t *backtrack_matrix = ab->abcm->cuda_backtrack_matrix;
	uint8_t *e_source_order_matrix1 = ab->abcm->cuda_backtrack_matrix + (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	uint8_t *e_source_order_matrix2 = ab->abcm->cuda_backtrack_matrix + 2 * (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	uint8_t *m_source_order_matrix = ab->abcm->cuda_backtrack_matrix + 3 * (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	uint8_t *e1e2f1f2_extension_matrix = ab->abcm->cuda_backtrack_matrix + 4 * (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	*backtrack_size = *backtrack_size + 1; // the first value is -1, so here add 1.
	checkCudaErrors(cudaMemcpy(backtrack_matrix, DEV_BACKTRACK, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(e_source_order_matrix1, E_SOURCE_ORDER, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(e_source_order_matrix2, E_SOURCE_ORDER2, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_source_order_matrix, M_SOURCE_ORDER, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(e1e2f1f2_extension_matrix, E1E2F1F2_EXTENSION, *backtrack_size, cudaMemcpyDeviceToHost));
	const char* backtrack_file_name = "backtrack_matrix";
	const char* e_source_order_name1 = "e_source_order_matrix1";
	const char* e_source_order_name2 = "e_source_order_matrix2";
	const char* m_source_order_name = "m_source_order_matrix";
	const char* e1e2f1f2_extension_name = "e1e2f1f2_extension_matrix";
	FILE *backtrack_file = fopen(backtrack_file_name, "w");
	FILE *e_source_order_file1 = fopen(e_source_order_name1, "w");
	FILE *e_source_order_file2 = fopen(e_source_order_name2, "w");
	FILE *m_source_order_file = fopen(m_source_order_name, "w");
	FILE *e1e2f1f2_extension_file = fopen(e1e2f1f2_extension_name, "w");
	/* print_backtrack_matrix(backtrack_matrix, *backtrack_size, cuda_accumulate_beg, matrix_row_n, backtrack_file); */
	/* print_backtrack_matrix(e_source_order_matrix1, *backtrack_size, cuda_accumulate_beg, matrix_row_n, e_source_order_file1); */
	/* print_backtrack_matrix(e_source_order_matrix2, *backtrack_size, cuda_accumulate_beg, matrix_row_n, e_source_order_file2); */
	/* print_backtrack_matrix(m_source_order_matrix, *backtrack_size, cuda_accumulate_beg, matrix_row_n, m_source_order_file); */
	/* print_backtrack_matrix(e1e2f1f2_extension_matrix, *backtrack_size, cuda_accumulate_beg, matrix_row_n, e1e2f1f2_extension_file); */
	fclose(backtrack_file);
	fclose(e_source_order_file1);
	fclose(e_source_order_file2);
	fclose(m_source_order_file);
	fclose(e1e2f1f2_extension_file);
	

	fclose(file);

	// printf("dp_sn: %d\n", tot_dp_sn);
	// printf("dp_sn: %d, node_n: %d, seq_n: %d\n", tot_dp_sn, graph->node_n, qlen);
	int cuda_best_score = INF_MIN;
	int cuda_best_i;
	int cuda_best_j;
	// new add
	/* cudaDeviceSynchronize(); */
	cuda_abpoa_global_get_max(graph, CUDA_DP_H2E2F, DEV_DP_H2E2F, qlen, cuda_dp_end, &cuda_best_score, &cuda_best_i, &cuda_best_j);

	/* fprintf(stderr, "best_score: (%d, %d) -> %d\n", best_i, best_j, best_score); */
	fprintf(stderr, "cuda_best_score: (%d, %d) -> %d\n", cuda_best_i, cuda_best_j, cuda_best_score);

	/* cuda_abpoa_cg_backtrack(CUDA_DP_H2E2F, pre_index, pre_n, cuda_dp_beg, cuda_dp_end, abpt->m, mat, gap_ext1, gap_ext2, gap_oe1, gap_oe2, 0, 0, cuda_best_i, cuda_best_j, qlen, graph, abpt, query, res); */
	cuda_my_abpoa_cg_backtrack(backtrack_matrix, e_source_order_matrix1, e_source_order_matrix2, m_source_order_matrix, e1e2f1f2_extension_matrix, cuda_accumulate_beg, pre_index, pre_n, cuda_dp_beg, abpt->m, mat, 0, 0, cuda_best_i, cuda_best_j, qlen, graph, abpt, query, res);
	for (i = 0; i < graph->node_n; ++i) free(pre_index[i]); free(pre_index); free(pre_n);
	/* SIMDFree(PRE_MASK); SIMDFree(SUF_MIN); SIMDFree(PRE_MIN); */
	/* SIMDFree(GAP_E1S); SIMDFree(GAP_E2S); */
	return cuda_best_score;
}

void my_test(abpoa_para_t *abpt){
	int inf_min = abpt->inf_min;
	SIMDi GAP_E1 = SIMDSetOnei32(abpt->gap_ext1);
	SIMDi GAP_E2 = SIMDSetOnei32(abpt->gap_ext2);
	int pn = 4;
	SIMDi *F_array = (SIMDi*)SIMDMalloc(4 * 16, 16);
	for (int i = 0;i < pn;i++){
		F_array[i] = SIMDSetOnei32(326);
	}
	int32_t *f = (int32_t *)F_array;
	f[0] = 200;
	f[3] = 400;
	SIMDi *PRE_MASK, *SUF_MIN, *PRE_MIN, *GAP_E1S, *GAP_E2S;
	int log_n = 2;
	SIMDi SIMD_INF_MIN = SIMDSetOnei32(inf_min);
	int size = 32;
	PRE_MASK = (SIMDi*)SIMDMalloc((pn+1) * size, size), SUF_MIN = (SIMDi*)SIMDMalloc((pn+1) * size, size), PRE_MIN = (SIMDi*)SIMDMalloc(pn * size, size), GAP_E1S =  (SIMDi*)SIMDMalloc(log_n * size, size), GAP_E2S =  (SIMDi*)SIMDMalloc(log_n * size, size);
	int i;
	int j;
	for (i = 0; i < pn; ++i) {
		int32_t *pre_mask = (int32_t*)(PRE_MASK + i);
		for (j = 0; j <= i; ++j) pre_mask[j] = -1;
		for (j = i+1; j < pn; ++j) pre_mask[j] = 0;
	} PRE_MASK[pn] = PRE_MASK[pn-1];
	SUF_MIN[0] = SIMDShiftLeft(SIMD_INF_MIN, SIMDShiftOneNi32);
	for (i = 1; i < pn; ++i) SUF_MIN[i] = SIMDShiftLeft(SUF_MIN[i-1], SIMDShiftOneNi32); SUF_MIN[pn] = SUF_MIN[pn-1];
	for (i = 1; i < pn; ++i) {
		int32_t *pre_min = (int32_t*)(PRE_MIN + i);
		for (j = 0; j < i; ++j) pre_min[j] = inf_min;
		for (j = i; j < pn; ++j) pre_min[j] = 0;
	}
	GAP_E1S[0] = GAP_E1; GAP_E2S[0] = GAP_E2;
	for (i = 1; i < log_n; ++i) {
		GAP_E1S[i] = SIMDAddi32(GAP_E1S[i-1], GAP_E1S[i-1]); GAP_E2S[i] = SIMDAddi32(GAP_E2S[i-1], GAP_E2S[i-1]);
	}
	int cov_bit = 0;
	SIMDi F = F_array[0];
	SIMDi F2 = F_array[0];
	print_simd(&F, "first", int32_t);
	F = SIMDMaxi32(F, SIMDOri(SIMDAndi(SIMDShiftLeft(SIMDSubi32(F, GAP_E1S[0]), SIMDShiftOneNi32), PRE_MASK[cov_bit]), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[1])));          \
	print_simd(&F, "orign F", int32_t);
	F2 = SIMDMaxi32(F2, SIMDOri(SIMDShiftLeft(SIMDSubi32(F, GAP_E1S[0]), SIMDShiftOneNi32), SIMDOri(SUF_MIN[cov_bit], PRE_MIN[1])));          \
	print_simd(&F2, "noAndF", int32_t);
}

/* int abpoa_cg_global_align_sequence_to_graph_core(abpoa_t *ab, int qlen, uint8_t *query, abpoa_para_t *abpt, SIMD_para_t sp, abpoa_res_t *res) { */
/*     int tot_dp_sn = 0; */
/*     abpoa_graph_t *graph = ab->abg; abpoa_simd_matrix_t *abm = ab->abm; */
/*     int matrix_row_n = graph->node_n, matrix_col_n = qlen + 1; */
/*     int **pre_index, *pre_n; */
/*     int i, j, *dp_beg, *dp_beg_sn, *dp_end, *dp_end_sn, node_id, index_i; */
/*     int beg_sn, end_sn; */
/*     int pn, log_n, size, qp_sn, dp_sn; [> pn: value per SIMDi, qp_sn/dp_sn/d_sn: segmented length<] */
/*     SIMDi *dp_h, *qp, *qi; */
/*     int32_t best_score = sp.inf_min, inf_min = sp.inf_min; */
/*     int *mat = abpt->mat, best_i = 0, best_j = 0; int32_t gap_ext1 = abpt->gap_ext1; */
/*     int w = abpt->wb < 0 ? qlen : abpt->wb + (int)(abpt->wf * qlen); [> when w < 0, do whole global <] */
/*     SIMDi zero = SIMDSetZeroi(), SIMD_INF_MIN = SIMDSetOnei32(inf_min); */
/*     pn = sp.num_of_value; qp_sn = dp_sn = (matrix_col_n + pn - 1) / pn, log_n = sp.log_num, size = sp.size; */
/*     qp = abm->s_mem; */
/*  */
/*     int32_t gap_open1 = abpt->gap_open1, gap_oe1 = gap_open1 + gap_ext1; */
/*     int32_t gap_open2 = abpt->gap_open2, gap_ext2 = abpt->gap_ext2, gap_oe2 = gap_open2 + gap_ext2; */
/*     SIMDi *DP_H2E2F, *dp_e1, *dp_e2, *dp_f2, *dp_f1; */
/*     SIMDi GAP_O1 = SIMDSetOnei32(gap_open1), GAP_O2 = SIMDSetOnei32(gap_open2), GAP_E1 = SIMDSetOnei32(gap_ext1), GAP_E2 = SIMDSetOnei32(gap_ext2), GAP_OE1 = SIMDSetOnei32(gap_oe1), GAP_OE2 = SIMDSetOnei32(gap_oe2); */
/*     DP_H2E2F = qp + qp_sn * abpt->m; qi = DP_H2E2F + dp_sn * matrix_row_n * 5; */
/*  */
/*     // for SET_F mask[pn], suf_min[pn], pre_min[logN] */
/*     SIMDi *PRE_MASK, *SUF_MIN, *PRE_MIN, *GAP_E1S, *GAP_E2S; */
/*     PRE_MASK = (SIMDi*)SIMDMalloc((pn+1) * size, size), SUF_MIN = (SIMDi*)SIMDMalloc((pn+1) * size, size), PRE_MIN = (SIMDi*)SIMDMalloc(pn * size, size), GAP_E1S =  (SIMDi*)SIMDMalloc(log_n * size, size), GAP_E2S =  (SIMDi*)SIMDMalloc(log_n * size, size); */
/*     for (i = 0; i < pn; ++i) { */
/*         int32_t *pre_mask = (int32_t*)(PRE_MASK + i); */
/*         for (j = 0; j <= i; ++j) pre_mask[j] = -1; */
/*         for (j = i+1; j < pn; ++j) pre_mask[j] = 0; */
/*     } PRE_MASK[pn] = PRE_MASK[pn-1]; */
/*     SUF_MIN[0] = SIMDShiftLeft(SIMD_INF_MIN, SIMDShiftOneNi32);  */
/*     for (i = 1; i < pn; ++i) SUF_MIN[i] = SIMDShiftLeft(SUF_MIN[i-1], SIMDShiftOneNi32); SUF_MIN[pn] = SUF_MIN[pn-1]; */
/*     for (i = 1; i < pn; ++i) { */
/*         int32_t *pre_min = (int32_t*)(PRE_MIN + i); */
/*         for (j = 0; j < i; ++j) pre_min[j] = inf_min; */
/*         for (j = i; j < pn; ++j) pre_min[j] = 0; */
/*     } */
/*     GAP_E1S[0] = GAP_E1; GAP_E2S[0] = GAP_E2; */
/*     for (i = 1; i < log_n; ++i) { */
/*         GAP_E1S[i] = SIMDAddi32(GAP_E1S[i-1], GAP_E1S[i-1]); GAP_E2S[i] = SIMDAddi32(GAP_E2S[i-1], GAP_E2S[i-1]); */
/*     } */
/*     abpoa_init_var(abpt, query, qlen, qp, qi, mat, qp_sn, pn, SIMD_INF_MIN); */
/*     dp_beg = abm->dp_beg, dp_end = abm->dp_end, dp_beg_sn = abm->dp_beg_sn, dp_end_sn = abm->dp_end_sn; */
/*     [> index of pre-node <] */
/*     pre_index = (int**)_err_malloc(graph->node_n * sizeof(int*)); */
/*     pre_n = (int*)_err_malloc(graph->node_n * sizeof(int)); */
/*     for (i = 0; i < graph->node_n; ++i) { */
/*         node_id = abpoa_graph_index_to_node_id(graph, i); */
/*         pre_n[i] = graph->node[node_id].in_edge_n; */
/*         pre_index[i] = (int*)_err_malloc(pre_n[i] * sizeof(int)); */
/*         for (j = 0; j < pre_n[i]; ++j) */
/*             pre_index[i][j] = abpoa_graph_node_id_to_index(graph, graph->node[node_id].in_id[j]); */
/*     } */
/*     abpoa_cg_first_dp(abpt, graph, dp_beg, dp_end, dp_beg_sn, dp_end_sn, pn, qlen, w, dp_sn, DP_H2E2F, SIMD_INF_MIN, inf_min, gap_open1, gap_ext1, gap_open2, gap_ext2, gap_oe1, gap_oe2); */
/*  */
/*     for (index_i = 1; index_i < matrix_row_n-1; ++index_i) { */
/*         node_id = abpoa_graph_index_to_node_id(graph, index_i); */
/*         SIMDi *q = qp + graph->node[node_id].base * qp_sn; */
/*         dp_h = DP_H2E2F + (index_i*5) * dp_sn; dp_e1 = dp_h + dp_sn; dp_e2 = dp_e1 + dp_sn; dp_f1 = dp_e2 + dp_sn; dp_f2 = dp_f1 + dp_sn;  */
/*         tot_dp_sn += abpoa_cg_dp(q, dp_h, dp_e1, dp_e2, dp_f1, dp_f2, pre_index, pre_n, index_i, graph, abpt, dp_sn, pn, qlen, w, DP_H2E2F, SIMD_INF_MIN, GAP_O1, GAP_O2, GAP_E1, GAP_E2, GAP_OE1, GAP_OE2, GAP_E1S, GAP_E2S, PRE_MIN, PRE_MASK, SUF_MIN, log_n, dp_beg, dp_end, dp_beg_sn, dp_end_sn); */
/*         if (abpt->wb >= 0) { */
/*             beg_sn = dp_beg_sn[index_i], end_sn = dp_end_sn[index_i]; */
/*             int max_i = abpoa_max(SIMD_INF_MIN, zero, inf_min, dp_h, qi, qlen, pn, beg_sn, end_sn); */
/*             [> abpoa_ada_max_i(max_i, graph, node_id); <] */
/*         }  */
/*     } */
/*     // printf("dp_sn: %d\n", tot_dp_sn); */
/*     // printf("dp_sn: %d, node_n: %d, seq_n: %d\n", tot_dp_sn, graph->node_n, qlen);    */
/*     abpoa_global_get_max(graph, DP_H2E2F, 5*dp_sn, qlen, dp_end, &best_score, &best_i, &best_j); */
/*     simd_abpoa_print_cg_matrix(int32_t);  */
/*     fprintf(stderr, "best_score: (%d, %d) -> %d\n", best_i, best_j, best_score);         */
/*     abpoa_cg_backtrack(DP_H2E2F, pre_index, pre_n, dp_beg, dp_end, dp_sn, abpt->m, mat, gap_ext1, gap_ext2, gap_oe1, gap_oe2, 0, 0, best_i, best_j, qlen, graph, abpt, query, res); */
/*     for (i = 0; i < graph->node_n; ++i) free(pre_index[i]); free(pre_index); free(pre_n); */
/*     SIMDFree(PRE_MASK); SIMDFree(SUF_MIN); SIMDFree(PRE_MIN); */
/*     SIMDFree(GAP_E1S); SIMDFree(GAP_E2S); */
/*     return best_score; */
/* } */

int simd_abpoa_align_sequence_to_graph(abpoa_t *ab, abpoa_para_t *abpt, uint8_t *query, int qlen, abpoa_res_t *res) {
    if (abpt->simd_flag == 0) err_fatal_simple("No SIMD instruction available.");

#ifdef __DEBUG__
    _simd_p32.inf_min = MAX_OF_TWO(abpt->gap_ext1, abpt->gap_ext2) * pow(2, (int)log2(qlen)) +MAX_OF_THREE(INT32_MIN + abpt->mismatch, INT32_MIN + abpt->gap_open1 + abpt->gap_ext1, INT32_MIN + abpt->gap_open2 + abpt->gap_ext2);
	INF_MIN = _simd_p32.inf_min;
    if (simd_abpoa_realloc(ab, qlen, abpt, _simd_p32)) return 0;
    if (cuda_simd_abpoa_realloc(ab, qlen, abpt)) return 0;
	if (abpt->gap_mode == ABPOA_CONVEX_GAP) return cuda_abpoa_cg_global_align_sequence_to_graph_core(ab, qlen, query, abpt, _simd_p32, res);
	else return 0;
#else
    int max_score, bits, mem_ret=0, gap_ext1 = abpt->gap_ext1, gap_ext2 = abpt->gap_ext2;
    int gap_oe1 = abpt->gap_open1+gap_ext1, gap_oe2 = abpt->gap_open2+gap_ext2;
    if (abpt->simd_flag & SIMD_AVX512F && !(abpt->simd_flag & SIMD_AVX512BW)) 
        max_score = INT16_MAX + 1; // AVX512F has no 8/16 bits operations
    else {
        int len = qlen > ab->abg->node_n ? qlen : ab->abg->node_n;
        max_score = MAX_OF_TWO(qlen * abpt->match, len * abpt->gap_ext1 + abpt->gap_open1);
    }
	printf("INT16_MAX:%x\n",INT16_MAX);
	printf("INT16_MAX + 1:%x\n",INT16_MAX + 1);
    if (max_score <= INT16_MAX-abpt->mismatch - gap_oe1 - gap_oe2) {
        _simd_p16.inf_min = MAX_OF_THREE(INT16_MIN + abpt->mismatch, INT16_MIN + gap_oe1, INT16_MIN + gap_oe2) + 31 * MAX_OF_TWO(gap_ext1, gap_ext2);
		INF_MIN = MAX_OF_THREE(INT16_MIN + abpt->mismatch, INT16_MIN + gap_oe1, INT16_MIN + gap_oe2) + 31 * MAX_OF_TWO(gap_ext1, gap_ext2);
        mem_ret = simd_abpoa_realloc(ab, qlen, abpt, _simd_p16);
        bits = 16;
    } else { 
        _simd_p32.inf_min = MAX_OF_THREE(INT32_MIN + abpt->mismatch, INT32_MIN + gap_oe1, INT32_MIN + gap_oe2) + 31 * MAX_OF_TWO(gap_ext1, gap_ext2);
		INF_MIN = MAX_OF_THREE(INT32_MIN + abpt->mismatch, INT32_MIN + gap_oe1, INT32_MIN + gap_oe2) + 31 * MAX_OF_TWO(gap_ext1, gap_ext2);
        mem_ret = simd_abpoa_realloc(ab, qlen, abpt, _simd_p32);
        bits = 32;
    }
    if (mem_ret) return 0;
	printf("bits:%d\n",bits);

    if (bits == 16) {
        if (abpt->gap_mode == ABPOA_LINEAR_GAP) {
            simd_abpoa_lg_align_sequence_to_graph_core(int16_t, ab, query, qlen, abpt, res, _simd_p16,  \
                    SIMDSetOnei16, SIMDMaxi16, SIMDAddi16, SIMDSubi16, SIMDShiftOneNi16, SIMDSetIfGreateri16, SIMDGetIfGreateri16);
        } else if (abpt->gap_mode == ABPOA_AFFINE_GAP) {
            simd_abpoa_ag_align_sequence_to_graph_core(int16_t, ab, query, qlen, abpt, res, _simd_p16,  \
                    SIMDSetOnei16, SIMDMaxi16, SIMDAddi16, SIMDSubi16, SIMDShiftOneNi16, SIMDSetIfGreateri16, SIMDGetIfGreateri16, SIMDSetIfEquali16);
			fprintf(stderr, "yejinlaile!\n");                                       \
        } else if (abpt->gap_mode == ABPOA_CONVEX_GAP) {
            simd_abpoa_cg_align_sequence_to_graph_core(int16_t, ab, query, qlen, abpt, res, _simd_p16,  \
                    SIMDSetOnei16, SIMDMaxi16, SIMDAddi16, SIMDSubi16, SIMDShiftOneNi16, SIMDSetIfGreateri16, SIMDGetIfGreateri16, SIMDSetIfEquali16);
        }
    } else { // 2147483647, DP_H/E/F: 32 bits
        if (abpt->gap_mode == ABPOA_LINEAR_GAP) {
            simd_abpoa_lg_align_sequence_to_graph_core(int32_t, ab, query, qlen, abpt, res, _simd_p32,  \
                    SIMDSetOnei32, SIMDMaxi32, SIMDAddi32, SIMDSubi32, SIMDShiftOneNi32, SIMDSetIfGreateri32, SIMDGetIfGreateri32);
        } else if (abpt->gap_mode == ABPOA_AFFINE_GAP) {
            simd_abpoa_ag_align_sequence_to_graph_core(int32_t, ab, query, qlen, abpt, res, _simd_p32,  \
                    SIMDSetOnei32, SIMDMaxi32, SIMDAddi32, SIMDSubi32, SIMDShiftOneNi32, SIMDSetIfGreateri32, SIMDGetIfGreateri32, SIMDSetIfEquali32);
        } else if (abpt->gap_mode == ABPOA_CONVEX_GAP) {
            simd_abpoa_cg_align_sequence_to_graph_core(int32_t, ab, query, qlen, abpt, res, _simd_p32,  \
                    SIMDSetOnei32, SIMDMaxi32, SIMDAddi32, SIMDSubi32, SIMDShiftOneNi32, SIMDSetIfGreateri32, SIMDGetIfGreateri32, SIMDSetIfEquali32);
        }
    }
#endif
   return 0;
}

