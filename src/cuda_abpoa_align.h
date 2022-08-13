#include "abpoa.h"
#include "abpoa_graph.h"

void cuda_print_row_h(FILE *file, int index, int *dp_h, int beg, int end, int qlen);
void cuda_print_row(FILE *file, int index, int *dp_h, int beg, int end, int qlen);


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
__global__ void cuda_first_dp_hf1f2(int *dp_f, int *dp_f1, int *dp_f2, int _end, int INF_MIN, int gap_open1, int gap_open2, int gap_ext1, int gap_ext2);

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
__global__ void cuda_first_dp_he1e2(int *dp_h, int *dp_e1, int *dp_e2, int _end, int INF_MIN, int gap_oe1, int gap_oe2);

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
__global__ void cuda_set_ME_first_node_nb(int *dp_h, int * pre_dp_h, uint8_t *m_source_order,int *dp_e, int * pre_dp_e,int *dp_e2, int * pre_dp_e2, uint8_t *e_source_order, uint8_t *e_source_order2, int beg, int end, int _beg, int _end, int INF_MIN, int flag);

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
__global__ void cuda_set_F_nb(int *dp_f, int e,int *dp_f2, uint8_t *e1e2f1f2_extension, int e2, int _beg, int _end);

/* 功能：计算横向打分、回溯矩阵-横向延伸标记,使用模板，在编译时就可以循环展开。线程数32时取消同步 */
/* 参数： */
/* 返回值：void */
__global__ void cuda_set_F(int *dp_f, int e, int _beg, int _end, int cov_bit, int max_right);

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
__global__ void cuda_get_max_F_and_set_E(int *dp_h, int *dp_e1,int *dp_e2, int *dp_f1,int *dp_f2, uint8_t *e_source_order, uint8_t *e_source_order2, uint8_t *e1e2f1f2_extension, int _beg, int _end, int gap_oe1, int gap_oe2, int gap_ext1, int gap_ext2, int INF_MIN, uint8_t *backtrack);

/* 功能：计算出最终的斜向来源得分，并右移一位 */
/* 参数：上面的函数都有介绍 */
/*     q：当前行发生碱基替换/匹配的得分，如果加上前驱行的得分，那么就是最终斜向来源得分 */
/* 返回值：void */
__global__ void cuda_add_and_suboe1_shift_right_one_nb(int *f, int *f2, int *dp_h, int oe,int oe2, int * q, int _beg, int _end, int INF_MIN, int max_pre_end);

/* 功能：找到斜向、纵向来源得分最高的前驱行，并计算相关回溯矩阵 */
/* 参数：上面的函数都有介绍  */
/* 返回值：void */
__global__ void cuda_set_ME(int *dp_h, int * pre_dp_h, uint8_t *m_source_order,int *dp_e, int * pre_dp_e,int *dp_e2, int * pre_dp_e2, uint8_t *e_source_order, uint8_t *e_source_order2, int _beg, int _end, int flag, int order, int gap);


/* 功能：找到当前行最大值的下标（CPU中直接遍历查找） */
/* 参数： */
/*     *dp_h：打分矩阵当前行 */
/*     beg：比对带起始位置  */
/*     end：比对带终止位置  */
/*     qlen：当前偏序比对单序列的长度 */
/* 返回值：当前行最大值下标 */
int cuda_abpoa_max(int *dp_h, int beg, int end, int qlen);

/* 功能：找到当前行最大值的下标（GPU中直接遍历查找） */
/* 参数： */
/*     *dp_h：打分矩阵当前行 */
/*     beg：比对带起始位置 */
/*     end：比对带终止位置 */
/*     qlen：当前偏序比对单序列的长度 */
/*     *_max_i：当前行最大值下标  */
/*     INF_MIN：初始化用 */
/* 返回值：void */
__global__ void dev_cuda_abpoa_max(int *dp_h, int beg, int end, int *_max_i, int INF_MIN);

// input the [beg, end] part of data,when finished, add beg to max_index 
/* 功能：找到当前行最大值的下标（GPU规约算法） */
/* 参数： */
/*     *data：当前行  */
/*     *out_data：保存最大值 */
/*     *out_index：保存最大值下标 */
/*     n：当前行长度 */
/* 返回值：void */
template <unsigned int block_size>
__global__ void reduce_max_index(int *data, int *out_data, int *out_index, int n);


/* 功能：找到当前行最大值的下标（GPU规约算法） */
/* 参数： */
/*     d_data：当前行  */
/*     n：当前行长度  */
/*     beg：比对带起始位置  */
/*     end：比对带终止位置  */
/*     max_index：最大值下标 */
/* 返回值：void */
__global__ void maxReduce(int* d_data, int n, int beg, int end, int *max_index);

/* 功能：更新DAG时，必要时重新分配内存 */
/* 参数：上面的函数都有介绍  */
/* 返回值：0：正常退出；1：异常退出 */
int cuda_simd_abpoa_realloc(abpoa_t *ab, int qlen, abpoa_para_t *abpt);

/* 功能：初始化qp数组，便于加速比对  */
/* 参数： */
/*     *abpt：序列比对的参数 */
/*     *query：将ACGT转换为对应数字，便于加速比对  */
/*     qlen：当前偏序比对单序列的长度  */
/*     *cuda_qp：qp数组  */
/*     *mat：5*5数组，表示ACGTN两两对应的打分值  */
/*     INF_MIN：初始化打分矩阵用，需要考虑防止溢出 */
/* 返回值：void */
void cuda_abpoa_init_var(abpoa_para_t *abpt, uint8_t *query, int qlen, int *cuda_qp, int *mat, int INF_MIN);

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
void cuda_abpoa_cg_first_dp(abpoa_para_t *abpt, abpoa_graph_t *graph, int *cuda_dp_beg, int *cuda_dp_end, int qlen, int w, int *DEV_DP_H2E2F, int gap_open1, int gap_ext1, int gap_open2, int gap_ext2, int gap_oe1, int gap_oe2);

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
void cuda_abpoa_cg_dp(int *DEV_DP_H2E2F, int **pre_index, int *pre_n, int index_i, abpoa_graph_t *graph, abpoa_para_t *abpt, int qlen, int w, int *CUDA_DP_H2E2F, int *cuda_dp_beg, int *cuda_dp_end, int gap_ext1, int gap_ext2, int gap_oe1, int gap_oe2, FILE *file, int *dev_dp_h, int * dev_dp_e1, int *dev_dp_e2, int *dev_dp_f1, int *dev_dp_f2, int *dev_q, int *dev_pre_dp_h, int *dev_pre_dp_e1, int *dev_pre_dp_e2, int *band, uint8_t *dev_backtrack, int *backtrack_size, int *cuda_accumulate_beg, uint8_t *e_source_order, uint8_t *e_source_order2, uint8_t *m_source_order, uint8_t *e1e2f1f2_extension);

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
void cuda_my_abpoa_cg_backtrack(uint8_t *backtrack_matrix, uint8_t *e_source_order1, uint8_t *e_source_order2, uint8_t *m_source_order, uint8_t *e1e2f1f2_extension, int *cuda_accumulate_beg, int **pre_index, int *pre_n, int *dp_beg, int m, int *mat, int start_i, int start_j, int best_i, int best_j, int qlen, abpoa_graph_t *graph, abpoa_para_t *abpt, uint8_t *query, abpoa_res_t *res);

/* 功能：执行偏序比对 */
/* 参数： */
/*     ab：序列比对的数据，包括DAG、打分矩阵 */
/*     abpt：序列比对的参数 */
/*     query：将ACGT转换为对应数字，便于加速比对 */
/*     qlen：当前偏序比对单序列的长度 */
/*     res：回溯的相关数据 */
/* 返回值： */
/*     1：异常退出， 0：正常退出 */
int cuda_abpoa_cg_global_align_sequence_to_graph_core(abpoa_t *ab, int qlen, uint8_t *query, abpoa_para_t *abpt, abpoa_res_t *res);

/* 功能：计算INFMIN（防止溢出）,执行偏序比对 */
/* 参数： */
/*     ab：序列比对的数据，包括DAG、打分矩阵 */
/*     abpt：序列比对的参数 */
/*     query：将ACGT转换为对应数字，便于加速比对 */
/*     qlen：当前偏序比对单序列的长度 */
/*     res：回溯的相关数据 */
/* 返回值： */
/*     1：异常退出， 0：正常退出 */
int simd_abpoa_align_sequence_to_graph(abpoa_t *ab, abpoa_para_t *abpt, uint8_t *query, int qlen, abpoa_res_t *res);

