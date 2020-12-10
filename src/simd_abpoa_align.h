#ifndef SIMD_ABPOA_ALIGN_H
#define SIMD_ABPOA_ALIGN_H

#include "abpoa.h"
#include "abpoa_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

int simd_abpoa_align_sequence_to_graph(abpoa_t *ab, abpoa_para_t *abpt, uint8_t *query, int qlen, abpoa_res_t *res);
abpoa_simd_matrix_t *abpoa_init_simd_matrix(void);
abpoa_cuda_matrix_t *abpoa_init_cuda_matrix(void);
void abpoa_free_simd_matrix(abpoa_simd_matrix_t *abm);
void abpoa_free_cuda_matrix(abpoa_cuda_matrix_t *abcm);
// abpoa_simd_matrix_t *abpoa_init_simd_matrix(void);
// void abpoa_free_simd_matrix(abpoa_simd_matrix_t *abm);

#ifdef __cplusplus
}
#endif

#endif
