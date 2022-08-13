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
#include "cuda_abpoa_align.h"

#define __DEBUG__\

/* no need SIMD_para_t */\
int INF_MIN;
__device__ const int d_float_min = INT_MIN;

/* 功能：根据当前行最大值下标，计算后继行比对带计算相关参数 */
/* 参数： */
/*     max_i：当前行最大值下标  */
/*     *graph：DAG  */
/*     node_id：当前行对应的DAG的node */
/*     *file：debug输出文件 */
/* 返回值：void */
void abpoa_ada_max_i(int max_i, abpoa_graph_t *graph, int node_id, FILE *file);


/* 功能：找到全局比对回溯的起始坐标。DAG可能有多个终止节点，选择这些节点最末端位置得分最高的作为回溯起始坐标 */
/*     *graph：DAG  */
/*     *CUDA_DP_H_HE：打分矩阵（CPU内存） */
/*     *DEV_DP_H2E2F：打分矩阵（GPU内存） */
/*     qlen：当前偏序比对单序列的长度  */
/*     *dp_end：比对带的终止位置数组  */
/*     *best_score：当前最高得分（局部比对回溯） */
/*     *best_i：当前最高得分的行（局部比对回溯） */
/*     *best_j：当前最高得分的列（局部比对回溯） */
/* 返回值：void */
void cuda_abpoa_global_get_max(abpoa_graph_t *graph, int *CUDA_DP_H_HE, int *DEV_DP_H2E2F, int qlen, int *dp_end, int32_t *best_score, int *best_i, int *best_j);

/* 功能：计算INFMIN（防止溢出）,执行偏序比对 */
/* 参数： */
/*     ab：序列比对的数据，包括DAG、打分矩阵 */
/*     abpt：序列比对的参数 */
/*     query：将ACGT转换为对应数字，便于加速比对 */
/*     qlen：当前偏序比对单序列的长度 */
/*     res：回溯的相关数据 */
/* 返回值： */
/*     1：异常退出， 0：正常退出 */
int simd_abpoa_align_sequence_to_graph(abpoa_t *ab, abpoa_para_t *abpt, uint8_t *query, int qlen, abpoa_res_t *res) {
	// 防止溢出
    INF_MIN = MAX_OF_TWO(abpt->gap_ext1, abpt->gap_ext2) * pow(2, (int)log2(qlen)) +MAX_OF_THREE(INT32_MIN + abpt->mismatch, INT32_MIN + abpt->gap_open1 + abpt->gap_ext1, INT32_MIN + abpt->gap_open2 + abpt->gap_ext2);
    if (cuda_simd_abpoa_realloc(ab, qlen, abpt)) return 0;
	if (abpt->gap_mode == ABPOA_CONVEX_GAP) return cuda_abpoa_cg_global_align_sequence_to_graph_core(ab, qlen, query, abpt, res);
	else return 0;
}

/* 功能：执行偏序比对 */
/* 参数： */
/*     ab：序列比对的数据，包括DAG、打分矩阵 */
/*     abpt：序列比对的参数 */
/*     query：将ACGT转换为对应数字，便于加速比对 */
/*     qlen：当前偏序比对单序列的长度 */
/*     res：回溯的相关数据 */
/* 返回值： */
/*     1：异常退出， 0：正常退出 */
int cuda_abpoa_cg_global_align_sequence_to_graph_core(abpoa_t *ab, int qlen, uint8_t *query, abpoa_para_t *abpt, abpoa_res_t *res) {
	// DAG
	abpoa_graph_t *graph = ab->abg;
	// DAG节点数量，也是打分矩阵的行数
	int matrix_row_n = graph->node_n;
	// 第一维表示打分矩当前行id，第二维表示第一维id的前驱行id
	int **pre_index;
	// 索引是打分矩阵当前行id，值是当前行前驱行的id
	int *pre_n;
	// DAG节点id
	int node_id;
	// 打分矩阵行id
	int index_i;
	// 5*5数组，表示ACGTN两两对应的打分值
	int *mat = abpt->mat; 
	// 比对带计算相关参数
	int w = abpt->wb < 0 ? qlen : abpt->wb + (int)(abpt->wf * qlen); 
	// convex比对模式的横向打分计算相关参数
	int32_t gap_ext1 = abpt->gap_ext1;
	int32_t gap_open1 = abpt->gap_open1, gap_oe1 = gap_open1 + gap_ext1;
	int32_t gap_open2 = abpt->gap_open2, gap_ext2 = abpt->gap_ext2, gap_oe2 = gap_open2 + gap_ext2;

	int i, j;

	// cuda 相关参数
	// abpoa相关内存数据
	abpoa_cuda_matrix_t *abcm = ab->abcm;
	// 比对带起始、终止位置数组
	int * cuda_dp_beg, *cuda_dp_end;
	// 当前行比对带起始、终止位置
	int cuda_beg, cuda_end;
	// 5 * (qlen + 1), 保存单序列每个碱基与ACGTN的替换、匹配得分，加速比对
	int *cuda_qp;
	cuda_qp = abcm->c_mem;
	// cuda打分矩阵，CPU内存。五行作为一个大单位。第一行保存当前行最终得分，第二行保存以当前行为来源的斜向得分，第三行保存以当前行为来源的斜向得分2，第四行保存以当前行为来源的横向得分，第五行保存以当前行为来源的横向得分2
	int *CUDA_DP_H2E2F;
	// cuda打分矩阵，GPU内存。
	int *DEV_DP_H2E2F;
	// 回溯矩阵，0-匹配，1-横向1,2-横向2，3-纵向1， 4-纵向2
	uint8_t *DEV_BACKTRACK;
	// 回溯矩阵-纵向来源1，保存前驱节点的索引，通过该索引在前驱节点数组中找到实际的前驱节点id
	uint8_t *E_SOURCE_ORDER;
	// 回溯矩阵-纵向来源2，保存前驱节点的索引，通过该索引在前驱节点数组中找到实际的前驱节点id
	uint8_t *E_SOURCE_ORDER2;
	// 回溯矩阵-横向来源，保存前驱节点的索引，通过该索引在前驱节点数组中找到实际的前驱节点id
	uint8_t *M_SOURCE_ORDER;
	// 回溯矩阵-延伸标记。每个元素的后四位分别标记纵向1、纵向2、横向1、横向2的延伸，0表示可以继续延伸，1表示延伸终止，接下来必然是匹配
	uint8_t *E1E2F1F2_EXTENSION;
	// GPU内存,含义同前面qp
	int *dev_qp;
	// GPU内存，保存当前行最大值下标
	int *dev_max_i;
	// 回溯矩阵每一行的起始位置
	int *cuda_accumulate_beg; 
	CUDA_DP_H2E2F = cuda_qp + (qlen + 1) * abpt->m;
	dev_max_i = abcm->dev_mem; 
	dev_qp = dev_max_i + 1;
	DEV_DP_H2E2F = dev_qp + abpt->m * (qlen + 1);
	// 回溯矩阵整体结构：以下五个矩阵以此存储，每个矩阵为当前偏序比对打分矩阵大小×1字节
	DEV_BACKTRACK = abcm->dev_backtrack_matrix;
	E_SOURCE_ORDER = DEV_BACKTRACK + (qlen + 1) * ab->abg->node_n;
	E_SOURCE_ORDER2 = E_SOURCE_ORDER + (qlen + 1) * ab->abg->node_n;
	M_SOURCE_ORDER = E_SOURCE_ORDER2 + (qlen + 1) * ab->abg->node_n;
	E1E2F1F2_EXTENSION = M_SOURCE_ORDER + (qlen + 1) * ab->abg->node_n;
	cuda_accumulate_beg = abcm->cuda_accumulate_beg;
	// 计算回溯矩阵大小，只包含比对表带大小
	*ab->abcm->backtrack_size = 0;

	// 初始化qp
	cuda_abpoa_init_var(abpt, query, qlen, cuda_qp, mat, INF_MIN);

	cuda_dp_beg = abcm->cuda_dp_beg;
	cuda_dp_end = abcm->cuda_dp_end;

	/* index of pre-node */
	// 前驱节点信息
	pre_index = (int**)_err_malloc(graph->node_n * sizeof(int*));
	pre_n = (int*)_err_malloc(graph->node_n * sizeof(int));
	for (i = 0; i < graph->node_n; ++i) {
		node_id = abpoa_graph_index_to_node_id(graph, i);
		pre_n[i] = graph->node[node_id].in_edge_n;
		pre_index[i] = (int*)_err_malloc(pre_n[i] * sizeof(int));
		for (j = 0; j < pre_n[i]; ++j)
			pre_index[i][j] = abpoa_graph_node_id_to_index(graph, graph->node[node_id].in_id[j]);
	}
	// 计算第一行
	cuda_abpoa_cg_first_dp(abpt, graph, cuda_dp_beg, cuda_dp_end, qlen, w, DEV_DP_H2E2F, gap_open1, gap_ext1, gap_open2, gap_ext2, gap_oe1, gap_oe2);

	//the cuda part
		char file_name[] = "cuda_max_dp_h.fa";
		FILE *file = fopen(file_name, "w");

	// 以下变量从矩阵中抽取某一行作为当前行处理,dev即为GPU中的数据
	int *dev_dp_h;
	int *dev_dp_e1;
	int *dev_dp_e2;
	int *dev_dp_f1;
	int *dev_dp_f2;
	u_int8_t *dev_backtrack, *e_source_order, *e_source_order2, *m_source_order, *e1e2f1f2_extension;
	// 保存某个前驱行
	int *dev_pre_dp_h;
	int *dev_pre_dp_e1;
	int *dev_pre_dp_e2;
	// GPU内存中的qp数组
	int *dev_q;
	// 当前行最大值下标
	int max_i ;
	// 回溯矩阵大小
	int *backtrack_size = ab->abcm->backtrack_size;
	*backtrack_size = -1;

	// 将dp数组从cpu拷贝到gpu
	checkCudaErrors(cudaMemcpy(dev_qp, cuda_qp, abpt->m * (qlen + 1) * sizeof(int), cudaMemcpyHostToDevice));
	// debug输出文件
	char file_name_graph[] = "graph_max_right";
	FILE *file_graph = fopen(file_name_graph, "w");
	const char* max_position_name = "max_position_name.fa";
	FILE *file_max_position = fopen(max_position_name, "w");
	// gpu中求当前行最大值下标用到的数据
	int *reduce_index;
	cudaMalloc((void **)&reduce_index, sizeof(int));
	int *out_data;
	cudaMalloc((void **)&out_data, sizeof(int));
	int *out_index;
	cudaMalloc((void **)&out_index, sizeof(int));
	// 每一行比对带的长度
	int *band = (int *)malloc(matrix_row_n * sizeof(int));

	// 依次计算每一行的打分矩阵、回溯矩阵
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

		// bebug
		/* fprintf(file_graph, "index:%\n", index_i); */
		/* fprintf(stderr, "index_i:%d\n", index_i); */
		/* fprintf(stderr, "dev_max_i:%x\n", dev_max_i); */
		/* fprintf(stderr, "max_i:%x\n", &max_i); */

		dev_q = dev_qp +  graph->node[node_id].base * (qlen + 1);
		// 计算当前行
		cuda_abpoa_cg_dp(DEV_DP_H2E2F, pre_index, pre_n, index_i, graph, abpt, qlen, w, CUDA_DP_H2E2F, cuda_dp_beg, cuda_dp_end, gap_ext1, gap_ext2, gap_oe1, gap_oe2, file, dev_dp_h, dev_dp_e1, dev_dp_e2,dev_dp_f1, dev_dp_f2, dev_q, dev_pre_dp_h, dev_pre_dp_e1, dev_pre_dp_e2, band, dev_backtrack, backtrack_size, cuda_accumulate_beg, e_source_order, e_source_order2, m_source_order, e1e2f1f2_extension);
		// debug
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
			// 计算当前行最大值下标
			cuda_beg = cuda_dp_beg[index_i];	
			cuda_end = cuda_dp_end[index_i];	

			// 以下为各种不同的算法实现
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
			// cpu内存中保存最大值下标
			int reduce_cpu_index2;
			// 
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
			// 见函数解释
			abpoa_ada_max_i(reduce_cpu_index2 + cuda_beg, graph, node_id, file_max_position);

			// debug
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


	// debug
	// check matrix
	/* int matrix_size = ((qlen + 1) * matrix_row_n * 5 * sizeof(int)); // DP_H2E2F, convex */
	/* checkCudaErrors(cudaMemcpy(CUDA_DP_H2E2F, DEV_DP_H2E2F, matrix_size, cudaMemcpyDeviceToHost)); */

	/* uint64_t temp_msize = (qlen + 1) * matrix_row_n * sizeof(uint8_t); // for backtrack matrix */
	// cpu中保存回溯矩阵
	uint8_t *backtrack_matrix = ab->abcm->cuda_backtrack_matrix;
	uint8_t *e_source_order_matrix1 = ab->abcm->cuda_backtrack_matrix + (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	uint8_t *e_source_order_matrix2 = ab->abcm->cuda_backtrack_matrix + 2 * (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	uint8_t *m_source_order_matrix = ab->abcm->cuda_backtrack_matrix + 3 * (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	uint8_t *e1e2f1f2_extension_matrix = ab->abcm->cuda_backtrack_matrix + 4 * (qlen + 1) * sizeof(uint8_t) * ab->abg->node_n;
	// 只拷贝比对带大小的回溯矩阵
	*backtrack_size = *backtrack_size + 1; // the first value is -1, so here add 1.
	checkCudaErrors(cudaMemcpy(backtrack_matrix, DEV_BACKTRACK, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(e_source_order_matrix1, E_SOURCE_ORDER, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(e_source_order_matrix2, E_SOURCE_ORDER2, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_source_order_matrix, M_SOURCE_ORDER, *backtrack_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(e1e2f1f2_extension_matrix, E1E2F1F2_EXTENSION, *backtrack_size, cudaMemcpyDeviceToHost));

	// debug
	/* const char* backtrack_file_name = "backtrack_matrix"; */
	/* const char* e_source_order_name1 = "e_source_order_matrix1"; */
	/* const char* e_source_order_name2 = "e_source_order_matrix2"; */
	/* const char* m_source_order_name = "m_source_order_matrix"; */
	/* const char* e1e2f1f2_extension_name = "e1e2f1f2_extension_matrix"; */
	/* FILE *backtrack_file = fopen(backtrack_file_name, "w"); */
	/* FILE *e_source_order_file1 = fopen(e_source_order_name1, "w"); */
	/* FILE *e_source_order_file2 = fopen(e_source_order_name2, "w"); */
	/* FILE *m_source_order_file = fopen(m_source_order_name, "w"); */
	/* FILE *e1e2f1f2_extension_file = fopen(e1e2f1f2_extension_name, "w"); */
	/* print_backtrack_matrix(backtrack_matrix, *backtrack_size, cuda_accumulate_beg, matrix_row_n, backtrack_file); */
	/* print_backtrack_matrix(e_source_order_matrix1, *backtrack_size, cuda_accumulate_beg, matrix_row_n, e_source_order_file1); */
	/* print_backtrack_matrix(e_source_order_matrix2, *backtrack_size, cuda_accumulate_beg, matrix_row_n, e_source_order_file2); */
	/* print_backtrack_matrix(m_source_order_matrix, *backtrack_size, cuda_accumulate_beg, matrix_row_n, m_source_order_file); */
	/* print_backtrack_matrix(e1e2f1f2_extension_matrix, *backtrack_size, cuda_accumulate_beg, matrix_row_n, e1e2f1f2_extension_file); */
	/* fclose(backtrack_file); */
	/* fclose(e_source_order_file1); */
	/* fclose(e_source_order_file2); */
	/* fclose(m_source_order_file); */
	/* fclose(e1e2f1f2_extension_file); */

	// printf("dp_sn: %d\n", tot_dp_sn);
	// printf("dp_sn: %d, node_n: %d, seq_n: %d\n", tot_dp_sn, graph->node_n, qlen);

	// 全局比对最佳得分、回溯起始坐标
	int cuda_best_score = INF_MIN;
	int cuda_best_i;
	int cuda_best_j;
	// new add
	/* cudaDeviceSynchronize(); */
	cuda_abpoa_global_get_max(graph, CUDA_DP_H2E2F, DEV_DP_H2E2F, qlen, cuda_dp_end, &cuda_best_score, &cuda_best_i, &cuda_best_j);

	/* fprintf(stderr, "best_score: (%d, %d) -> %d\n", best_i, best_j, best_score); */
	fprintf(stderr, "cuda_best_score: (%d, %d) -> %d\n", cuda_best_i, cuda_best_j, cuda_best_score);

	// 回溯，计算cigar串
	cuda_my_abpoa_cg_backtrack(backtrack_matrix, e_source_order_matrix1, e_source_order_matrix2, m_source_order_matrix, e1e2f1f2_extension_matrix, cuda_accumulate_beg, pre_index, pre_n, cuda_dp_beg, abpt->m, mat, 0, 0, cuda_best_i, cuda_best_j, qlen, graph, abpt, query, res);
	for (i = 0; i < graph->node_n; ++i) free(pre_index[i]); free(pre_index); free(pre_n);
	/* SIMDFree(PRE_MASK); SIMDFree(SUF_MIN); SIMDFree(PRE_MIN); */
	/* SIMDFree(GAP_E1S); SIMDFree(GAP_E2S); */

	fclose(file);
	fclose(file_graph);
	return cuda_best_score;
}

/* 功能：计算打分矩阵的一行、计算回溯矩阵的一行 */
/* 参数： */
/*     DEV_DP_H2E2F：打分矩阵（GPU内存） */
/*     pre_index：索引是打分矩阵当前行id，值是当前行前驱行的id */
/*     pre_n：索引是打分矩阵当前行id，值是当前行前驱行的id */
/*     index_i：当前打分矩阵行id */
/*     graph：DAG */
/*     abpt：序列比对的参数 */
/*     qlen：当前偏序比对单序列的长度 */
/*     w：比对带计算相关参数  */
/*     CUDA_DP_H2E2F：打分矩阵（CPU内存） */
/*     cuda_dp_beg：一行比对带的起始id */
/*     cuda_dp_end：一行比对带的终止id */
/*     gap_ext1：convex比对模式的连续插入、删除打分计算相关参数  */
/*     gap_ext2：convex比对模式的连续插入、删除打分计算相关参数  */
/*     gap_oe1：convex比对模式的连续插入、删除打分计算相关参数 */
/*     gap_oe2：convex比对模式的横向打分计算相关参数  */
/*     file：debug用的输出文件 */
/*     dev_dp_h：打分矩阵实际的一行 */
/*     dev_dp_e1：打分矩阵某一行的纵向打分1 */
/*     dev_dp_e2：打分矩阵某一行的纵向打分2 */
/*     dev_dp_f1：打分矩阵某一行的横向打分1 */
/*     dev_dp_f2：打分矩阵某一行的横向打分2 */
/*     dev_q：辅助计算斜向打分 */
/*     dev_pre_dp_h：前驱行的实际打分 */
/*     dev_pre_dp_e1：前驱行的纵向打分1 */
/*     dev_pre_dp_e2：前驱行纵向打分2 */
/*     band：每一行比对带的长度  */
/*     dev_backtrack：回溯矩阵  */
/*     backtrack_size：回溯矩阵大小 */
/*     cuda_accumulate_beg：回溯矩阵每一行的起始id  */
/*     e_source_order：回溯矩阵-纵向来源1 */
/*     e_source_order2：回溯矩阵-纵向来源2 */
/*     m_source_order：回溯矩阵-斜向来源 */
/*     e1e2f1f2_extension：回溯矩阵-连续插入、删除延伸标志 */
/* 返回值：void */
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
		
		// 计算所有前驱行比对带起始点最小值，和终止点最大值
        for (i = 0; i < pre_n[index_i]; ++i) {
            pre_i = pre_index[index_i][i];
            if (min_pre_beg > cuda_dp_beg[pre_i]) min_pre_beg = cuda_dp_beg[pre_i];
            if (max_pre_end < cuda_dp_end[pre_i]) max_pre_end = cuda_dp_end[pre_i];
		  // 比对带起始位置不应该小于min-pre
        } if (cuda_beg < min_pre_beg) cuda_beg = min_pre_beg; // beg should be never lower than mix_pre_beg

		/* fprintf(file, "min_pre%d\n", min_pre_beg); */
		// 与simd行为统一，方便debug
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
	// 计算实际的起始位置，不应该小于当前前驱节点比对带的起始位置
	if (cuda_pre_beg < cuda_beg) {
		flag = 1;
		_cuda_beg = cuda_beg;
		_cuda_beg_m = cuda_beg;
	}
	else {
		flag = 0;
		_cuda_beg = cuda_pre_beg;
		// 斜向打分错开一个单元
		_cuda_beg_m = cuda_pre_beg + 1;
	}

	// 当前行比对带长度
	band[index_i] = cuda_end - cuda_beg + 1;
	
	// 纵坐标最大是qlen，共（qlen+1）个元素
	_cuda_end = MIN_OF_THREE(cuda_pre_end + 1, cuda_end, qlen);

	threads_per_block = 512;
	blocks_per_grid = (cuda_end - cuda_beg + 1) + threads_per_block - 1 / threads_per_block;
	/* cuda_set_M_first_node<<<blocks_per_grid, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, cuda_beg, cuda_end, _cuda_beg, _cuda_end, INF_MIN, flag); */
	/* cuda_set_M_first_node_nb<<<3, threads_per_block>>>(dev_dp_h, dev_pre_dp_h, m_source_order, cuda_beg, cuda_end, _cuda_beg_m, _cuda_end, INF_MIN, flag); */

	// assemble E and M first node
	_cuda_end = MIN_OF_TWO(cuda_pre_end, cuda_end);
	// 用第一个前驱节点初始化
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
	// 遍历每一个前驱节点，找到最好的纵向、斜向得分，并计算回溯矩阵
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


	int temp_max_pre_end;
	if (max_pre_end + 2 >= cuda_end) {
		temp_max_pre_end = cuda_end - 2;
	} else {
		temp_max_pre_end = max_pre_end;
	}
	// 计算实际的斜向打分，并初始化横向打分。max-pre-end后两个单元是不能初始化横向得分的，应该设置为INF-MIN
	cuda_add_and_suboe1_shift_right_one_nb<<<3, threads_per_block>>>(dev_dp_f2,dev_dp_f1, dev_dp_h, gap_oe2, gap_oe1, dev_q, cuda_beg, cuda_end, INF_MIN, temp_max_pre_end);
	/* cuda_suboe1_shift_right_one_nb<<<10, threads_per_block>>>(dev_dp_f2,dev_dp_f1, dev_dp_h, gap_oe2, gap_oe1, cuda_beg, cuda_end, INF_MIN, temp_max_pre_end); */

	// 31-33test
	/* if (index_i == 52) { */
	/*     int *result1 = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(stderr, index_i, result1, cuda_beg, cuda_end, qlen); */
	/*     free(result1); */
	/* } */

	// 暂时没用了
	int cov_bit;
	cov_bit = MIN_OF_THREE(max_pre_end + 1, cuda_end, qlen) + 1;
	// 比对带长度
	int len = cuda_end - cuda_beg + 1;
	// cuda线程参数
	int F_block_dim = 3;
	int F_threads_per_block = 256;
	/* int F_threads_per_block = (len - 1) / 256 * 256 + 256; */

	/* int F_threads_per_block = 256; */
	/* int F_block_dim = (len - 1) / 256 + 1; */

	/* cuda_set_F<<<1, F_threads_per_block>>>(dev_dp_f1, gap_ext1, cuda_beg, cuda_end, cov_bit, max_pre_end); */
	/* cuda_set_F<<<1, F_threads_per_block>>>(dev_dp_f2, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end); */

	// 计算横向得分
	int logn = (int)log2(F_block_dim * F_threads_per_block);
	switch(logn) {
		case 1:
			cuda_set_F_nb<1><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 2:
			cuda_set_F_nb<2><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 3:
			cuda_set_F_nb<3><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 4:
			cuda_set_F_nb<4><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 5:
			cuda_set_F_nb<5><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 6:
			cuda_set_F_nb<6><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 7:
			cuda_set_F_nb<7><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 8:
			cuda_set_F_nb<8><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 9:
			cuda_set_F_nb<9><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
		case 10:
			cuda_set_F_nb<10><<<F_block_dim, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, e1e2f1f2_extension, gap_ext2, cuda_beg, cuda_end);break;
	}
	/* F_threads_per_block = 256; */
	/* cuda_set_F_nb2<<<1, F_threads_per_block>>>(dev_dp_f1, gap_ext1, dev_dp_f2, gap_ext2, cuda_beg, cuda_end, cov_bit, max_pre_end); */

	/* if (index_i == 52) { */
	/*     int *result1 = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/*     checkCudaErrors(cudaMemcpy(result1, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudaMemcpyDeviceToHost)); */
	/*     cuda_print_row(stderr, index_i, result1, cuda_beg, cuda_end, qlen); */
	/*     free(result1); */
	/* } */

	// 求得最终得分，并设置纵向来源得分
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
	/* fprintf(stderr, "f_time2: %lf ms\n", elapsedtime); */


	/* int *result = (int *)malloc(sizeof(int) * (qlen + 1) * 5); */
	/* checkcudaerrors(cudamemcpy(result, dev_dp_h, (qlen + 1) * 5 * sizeof(int), cudamemcpydevicetohost)); */
	/* cuda_print_row(file, index_i, result, cuda_beg, cuda_end, qlen); */
}

/* 功能：全局比对模式回溯，记录cigar串 */
/* 参数： */
/*     backtrack_matrix：回溯矩阵  */
/*     e_source_order1：打分矩阵某一行的纵向打分1  */
/*     e_source_order2：打分矩阵某一行的纵向打分2  */
/*     m_source_order：回溯矩阵-斜向来源  */
/*     e1e2f1f2_extension：回溯矩阵-删除延伸标志  */
/*     cuda_accumulate_beg：回溯矩阵每一行的起始id  */
/*     pre_index：值是当前行前驱行的id  */
/*     pre_n：前驱行节点个数 */
/*     dp_beg：比对带的起始位置数组  */
/*     m：值为5，代表ACGTN五种符号 */
/*     mat：5*5数组，表示ACGTN两两对应的打分值 */
/*     start_i：回溯起始行 */
/*     start_j：回溯起始列 */
/*     best_i：回溯终止行 */
/*     best_j：回溯终止列 */
/*     qlen：当前偏序比对单序列的长度  */
/*     graph：DAG  */
/*     abpt：序列比对的参数  */
/*     query：将ACGT转换为对应数字  */
/*     res：回溯的相关数据 */
/* 返回值：void */
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
		/* fprintf(stderr, "i:%d\t,j:%d\t\n", i, j); */
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
				/* if (i == 46 && j == 35) */
				/*     fprintf(stderr, "pre_i:%d\t\n", pre_i); */
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
	// debug
	/* abpoa_print_cigar(n_c, res->graph_cigar, graph); */
	/* abpoa_print_cigar(n_c, *graph_cigar, graph); */
}

/* 功能：根据当前行最大值下标，计算后继行比对带计算相关参数 */
/* 参数： */
/*     max_i：当前行最大值下标  */
/*     *graph：DAG  */
/*     node_id：当前行对应的DAG的node */
/*     *file：debug输出文件 */
/* 返回值：void */
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

/* 功能：初始化qp数组，便于加速比对  */
/* 参数： */
/*     *abpt：序列比对的参数 */
/*     *query：将ACGT转换为对应数字，便于加速比对  */
/*     qlen：当前偏序比对单序列的长度  */
/*     *cuda_qp：qp数组  */
/*     *mat：5*5数组，表示ACGTN两两对应的打分值  */
/*     INF_MIN：初始化打分矩阵用，需要考虑防止溢出 */
/* 返回值：void */
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
	
}

/* 功能：初始化打分矩阵第一行 */
/* 参数： */
/*     *dp_f：第一行最终打分 */
/*     *dp_f1：convex打分模式下横向打分1 */
/*     *dp_f2：concex打分模式下横向打分2 */
/*     _end：当前行比对带终止位置 */
/*     INF_MIN：初始化打分矩阵用  */
/*     gap_open1：convex比对模式的连续插入、删除打分计算相关参数 */
/*     gap_open2：convex比对模式的连续插入、删除打分计算相关参数 */
/*     gap_ext1：convex比对模式的连续插入、删除打分计算相关参数 */
/*     gap_ext2：convex比对模式的连续插入、删除打分计算相关参数 */
/* 返回值：void */
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

/* 功能：初始化打分矩阵第一行  */
/* 参数： */
/*     *dp_h：第一行最终打分  */
/*     *dp_e1：concex打分模式下纵向打分1 */
/*     *dp_e2：convex打分模式下纵向打分2 */
/*     _end：当前行比对带终止位置  */
/*     INF_MIN：初始化打分矩阵用  */
/*     gap_oe1：convex比对模式的连续插入、删除打分计算相关参数 */
/*     gap_oe2：convex比对模式的连续插入、删除打分计算相关参数 */
/* 返回值：void */
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

/* 功能：初始化打分矩阵第一行  */
/* 参数：（前面的函数都有解释） */
/*     *abpt */
/*     *graph */
/*     *cuda_dp_beg */
/*     *cuda_dp_end */
/*     qlen */
/*     w */
/*     *dev_dp_h2e2f */
/*     gap_open1 */
/*     gap_ext1 */
/*     gap_open2 */
/*     gap_ext2 */
/*     gap_oe1 */
/*     gap_oe2 */
/* 返回值：void */
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

/* 功能：计算当前行的纵向、斜向来源得分以及回溯矩阵相关数据 */
/* 参数： */
/*     *dp_h：打分矩阵当前行  */
/*     * pre_dp_h：前驱行  */
/*     *m_source_order：回溯矩阵-斜向来源  */
/*     *dp_e：打分矩阵当前行纵向打分1 */
/*     * pre_dp_e：前驱行纵向打分  */
/*     *dp_e2：打分矩阵当前行纵向打分2 */
/*     * pre_dp_e2：前驱行纵向打分2  */
/*     *e_source_order：回溯矩阵-纵向来源1  */
/*     *e_source_order2：回溯矩阵-纵向来源2  */
/*     beg：当前行比对带起始位置 */
/*     end：当前行比对带终止位置  */
/*     _beg：需要与前驱行比较的比对带起始位置  */
/*     _end：需要与前驱行比较的比对带终止位置 */
/*     INF_MIN：初始化用  */
/*     flag：标志当前行与前驱行比对带起始位置大小关系 */
/* 返回值：void */
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

/* 功能：计算横向打分、回溯矩阵-横向延伸标记,使用模板，在编译时就可以循环展开。线程数32时取消同步 */
/* 参数： */
/*     *dp_f：当前行横向打分1 */
/*     e：convex：比对模式的横向打分计算相关参数  */
/*     *dp_f2：当前行横向打分2 */
/*     *e1e2f1f2_extension：回溯矩阵-延伸标记 */
/*     e2：convex比对模式的横向打分计算相关参数  */
/*     _beg：比对带起始位置  */
/*     _end：比对带终止位置  */
/* 返回值：void */
template <unsigned int logn>
__global__ void cuda_set_F_nb(int *dp_f, int e,int *dp_f2, uint8_t *e1e2f1f2_extension, int e2, int _beg, int _end) {
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
		}
		idx += size;
		raw_idx += size;
		flag = 1;
	}
}

/* 功能：计算横向打分、回溯矩阵-横向延伸标记,使用模板，在编译时就可以循环展开。线程数32时取消同步 */
/* 参数： */
/* 返回值：void */
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
/* 功能：计算当前行最终得分，并计算回溯矩阵 */
/* 参数：上面的函数都有介绍 */
/*     *dp_h */
/*     *dp_e1 */
/*     *dp_e2 */
/*     *dp_f1 */
/*     *dp_f2 */
/*     *e_source_order */
/*     *e_source_order2 */
/*     *e1e2f1f2_extension */
/*     _beg */
/*     _end */
/*     gap_oe1 */
/*     gap_oe2 */
/*     gap_ext1 */
/*     gap_ext2 */
/*     INF_MIN */
/*     *backtrack */
/* 返回值：void */
__global__ void cuda_get_max_F_and_set_E(int *dp_h, int *dp_e1,int *dp_e2, int *dp_f1,int *dp_f2, uint8_t *e_source_order, uint8_t *e_source_order2, uint8_t *e1e2f1f2_extension, int _beg, int _end, int gap_oe1, int gap_oe2, int gap_ext1, int gap_ext2, int INF_MIN, uint8_t *backtrack) {
	int raw_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x + _beg;
	int flag = 1;
	int flag_f = 0;
	while (idx <= _end) {
		e1e2f1f2_extension[raw_idx] = 0;
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

// f = (h - oe)>>1
// a is the same as dp_h
/* 功能：计算出最终的斜向来源得分，并右移一位 */
/* 参数：上面的函数都有介绍 */
/*     q：当前行发生碱基替换/匹配的得分，如果加上前驱行的得分，那么就是最终斜向来源得分 */
/* 返回值：void */
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

// gap is the distance of _cuda_beg and cuda_beg, which is set for m_source_order index.
/* 功能：找到斜向、纵向来源得分最高的前驱行，并计算相关回溯矩阵 */
/* 参数：上面的函数都有介绍  */
/* 返回值：void */
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

// gap is the distance of _cuda_beg and cuda_beg, which is set for m_source_order index.


/* 功能：找到当前行最大值的下标（CPU中直接遍历查找） */
/* 参数： */
/*     *dp_h：打分矩阵当前行 */
/*     beg：比对带起始位置  */
/*     end：比对带终止位置  */
/*     qlen：当前偏序比对单序列的长度 */
/* 返回值：当前行最大值下标 */
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

/* 功能：找到当前行最大值的下标（GPU中直接遍历查找） */
/* 参数： */
/*     *dp_h：打分矩阵当前行 */
/*     beg：比对带起始位置 */
/*     end：比对带终止位置 */
/*     qlen：当前偏序比对单序列的长度 */
/*     *_max_i：当前行最大值下标  */
/*     INF_MIN：初始化用 */
/* 返回值：void */
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

// input the [beg, end] part of data,when finished, add beg to max_index 
/* 功能：找到当前行最大值的下标（GPU规约算法,使用模板，在编译时就可以循环展开。） */
/* 参数： */
/*     *data：当前行  */
/*     *out_data：保存最大值 */
/*     *out_index：保存最大值下标 */
/*     n：当前行长度 */
/* 返回值：void */
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
/* 功能：找到当前行最大值的下标（GPU规约算法） */
/* 参数： */
/*     d_data：当前行  */
/*     n：当前行长度  */
/*     beg：比对带起始位置  */
/*     end：比对带终止位置  */
/*     max_index：最大值下标 */
/* 返回值：void */
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


// for cuda: realloc memory everytime the graph is updated (nodes are updated already)
// * index_to_node_id/node_id_to_index/node_id_to_max_remain, max_pos_left/right
// * qp, DP_HE/H (if ag/lg), dp_f, qi (if ada/extend)
// * dp_beg/end, dp_beg/end_sn if band
// * pre_n, pre_index
/* 功能：更新DAG时，必要时重新分配内存 */
/* 参数：上面的函数都有介绍  */
/* 返回值：0：正常退出；1：异常退出 */
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
	// GPU内存必须先释放再分配
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
