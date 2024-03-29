#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "helper_cuda.h"
#include "helper_string.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include "abpoa.h"
#include "abpoa_graph.h"
#include "abpoa_align.h"
#include "seq.h"
#include "kseq.h"
#include "utils.h"
#include <helper_timer.h>

KSEQ_INIT(gzFile, gzread)

char NAME[20] = "abPOA";
char PROG[20] = "abpoa";
#define _ba BOLD UNDERLINE "a" NONE
#define _bb BOLD UNDERLINE "b" NONE
#define _bP BOLD UNDERLINE "P" NONE
#define _bO BOLD UNDERLINE "O" NONE
#define _bA BOLD UNDERLINE "A" NONE
char DESCRIPTION[100] = _ba "daptive " _bb "anded " _bP "artial " _bO "rder " _bA "lignment";
char VERSION[20] = "1.0.3";
char CONTACT[30] = "yangao07@hit.edu.cn";

const struct option abpoa_long_opt [] = {
    { "align-mode", 1, NULL, 'm' },

    { "match", 1, NULL, 'M' },
    { "mismatch", 1, NULL, 'X' },
    { "gap-open", 1, NULL, 'O' },
    { "gap-ext", 1, NULL, 'E' },

    { "extra-b", 1, NULL, 'b' },
    { "extra-f", 1, NULL, 'f' },
    { "zdrop", 1, NULL, 'z' },
    { "bouns", 1, NULL, 'e' },

    { "in-list", 0, NULL, 'l' },
    { "output", 1, NULL, 'o' },
    { "result", 1, NULL, 'r' },
    { "out-pog", 1, NULL, 'g' },
    { "cons-alg", 1, NULL, 'a' },
    { "diploid", 0, NULL, 'd', },
    { "min-freq", 1, NULL, 'q', },

    { "help", 0, NULL, 'h' },
    { "version", 0, NULL, 'v' },

    { 0, 0, 0, 0}
};

int abpoa_usage(void)
{
    err_printf("\n");
    err_printf("%s: %s \n\n", PROG, DESCRIPTION);
    err_printf("Version: %s\t", VERSION);
	err_printf("Contact: %s\n\n", CONTACT);
    err_printf("Usage: %s [options] <in.fa/fq> > cons.fa/msa.out\n\n", PROG);
    err_printf("Options:\n");
    err_printf("  Alignment:\n");
    err_printf("    -m --aln-mode INT       alignment mode [%d]\n", ABPOA_GLOBAL_MODE);
    err_printf("                              %d: global, %d: local, %d: extension\n", ABPOA_GLOBAL_MODE, ABPOA_LOCAL_MODE, ABPOA_EXTEND_MODE);
    err_printf("    -M --match    INT       match score [%d]\n", ABPOA_MATCH);
    err_printf("    -X --mismatch INT       mismatch penalty [%d]\n", ABPOA_MISMATCH);
    err_printf("    -O --gap-open INT(,INT) gap opening penalty (O1,O2) [%d,%d]\n", ABPOA_GAP_OPEN1, ABPOA_GAP_OPEN2);
    err_printf("    -E --gap-ext  INT(,INT) gap extension penalty (E1,E2) [%d,%d]\n", ABPOA_GAP_EXT1, ABPOA_GAP_EXT2);
    err_printf("                            %s provides three gap penalty modes, cost of a g-long gap:\n", NAME);
    err_printf("                            - convex (default): min{O1+g*E1, O2+g*E2}\n");
    err_printf("                            - affine (set O2 as 0): O1+g*E1\n");
    err_printf("                            - linear (set O1 as 0): g*E1\n");
    err_printf("  Adaptive banded DP:\n");
    err_printf("    -b --extra-b  INT       first adaptive banding parameter [%d]\n", ABPOA_EXTRA_B);
    err_printf("                            set b as < 0 to disable adaptive banded DP\n");
    err_printf("    -f --extra-f  FLOAT     second adaptive banding parameter [%.2f]\n", ABPOA_EXTRA_F);
    err_printf("                            the number of extra bases added on both sites of the band is\n");
    err_printf("                            b+f*L, where L is the length of the aligned sequence\n");
    // err_printf("    -z --zdrop    INT       Z-drop score in extension alignment [-1]\n");
    // err_printf("                            set as <= 0 to disable Z-drop extension\n");
    // err_printf("    -e --bonus    INT       end bonus score in extension alignment [-1]\n");
    // err_printf("                            set as <= 0 to disable end bounus\n");
    err_printf("  Input/Output:\n");
    err_printf("    -l --in-list            input file is a list of sequence file names [False]\n");
    err_printf("                            each line is one sequence file containing a set of sequences\n");
    err_printf("                            which will be aligned by abPOA to generate a consensus sequence\n");
    err_printf("    -o --output   FILE      ouput to FILE [stdout]\n");
    err_printf("    -r --result   INT       output result mode [%d]\n", ABPOA_OUT_CONS);
    // err_printf("                            %d: consensus (FASTA format), %d: MSA (PIR format), %d: both 0 & 1\n", ABPOA_OUT_CONS, ABPOA_OUT_MSA, ABPOA_OUT_BOTH);
    err_printf("                            - %d: consensus (FASTA format)\n", ABPOA_OUT_CONS);
    err_printf("                            - %d: MSA (PIR format)\n", ABPOA_OUT_MSA);
    err_printf("                            - %d: both 0 & 1\n", ABPOA_OUT_BOTH);
    err_printf("    -g --out-pog  FILE      dump final alignment graph to FILE (.pdf/.png) [Null]\n\n");

    err_printf("    -h --help               print this help usage information\n");
    err_printf("    -v --version            show version number\n");

    err_printf("  Parameters under development:\n");
    err_printf("    -a --cons-alg INT       algorithm to use for consensus calling [%d]\n", ABPOA_HB);
    // err_printf("                            %d: heaviest bundling, %d: heaviest column\n", ABPOA_HB, ABPOA_HC);
    err_printf("                            - %d: heaviest bundling\n", ABPOA_HB);
    err_printf("                            - %d: heaviest in column\n", ABPOA_HC);
    err_printf("    -d --diploid            input data is diploid [False]\n");
    err_printf("                            -a/--cons-alg will be set as %d when input is diploid\n", ABPOA_HC);
    err_printf("                            and at most two consensus sequences will be generated\n");
    err_printf("    -q --min-freq FLOAT     min frequency of each consensus for diploid input [%.2f]\n", DIPLOID_MIN_FREQ);

    err_printf("\n");
    return 1;
}

int abpoa_read_seq(kseq_t *read_seq, int chunk_read_n)
{
    kseq_t *s = read_seq;
    int n = 0;
    while (kseq_read(s+n) >= 0) {
        n++;
        if (n >= chunk_read_n) break;
    }
    return n;
}

void print_graph(abpoa_t *ab, FILE *file) {
	abpoa_graph_t *abg = ab->abg;
	int node_n = abg->node_n;
	int i;
	for (i = 0; i < node_n - 1; i++) {
		fprintf(file, "%d", abg->node[i].base);
		/* fprintf(file, "node_id:%d,char:%d\n", i, abg->node[i].base); */
	}
	fprintf(file, "\n");
} 

#define abpoa_core(read_fn) {   \
    gzFile readfp = xzopen(read_fn, "r"); kstream_t *fs = ks_init(readfp);  \
    for (i = 0; i < CHUNK_READ_N; ++i) read_seq[i].f = fs;  \
    /* progressively partial order alignment */     \
    n_seqs = 0,  tot_n = 0, read_id = 0; \
	/* reset graph for a new input file   */\
    abpoa_reset_graph(ab, abpt, bseq_m);    \
    while ((n_seqs = abpoa_read_seq(read_seq, CHUNK_READ_N)) != 0) {    \
		char file_name_graph_node[] = "graph_node";\
		FILE *file_graph_node = fopen(file_name_graph_node, "w");\
        for (i = 0; i < n_seqs; ++i) {  \
            kseq_t *seq = read_seq + i; \
            int seq_l = seq->seq.l; \
            if (seq_l <= 0) err_fatal("read_seq", "Unexpected read length: %d (%s)", seq_l, seq->name.s);   \
            char *seq1 = seq->seq.s;    \
			/* fprintf(stderr, "seq(%d): %s\n", seq_l, seq1);    */\
			fprintf(stderr, "seq(%d)\n", seq_l);   \
            if (seq_l > bseq_m) {   \
                bseq_m = seq_l; kroundup32(bseq_m); \
                bseq = (uint8_t*)_err_realloc(bseq, bseq_m * sizeof(uint8_t));  \
            }   \
            for (j = 0; j < seq_l; ++j) bseq[j] = nst_nt4_table[(int)(seq1[j])];    \
            abpoa_res_t res; res.graph_cigar=0; res.n_cigar=0;    \
            abpoa_align_sequence_to_graph(ab, abpt, bseq, seq_l, &res); \
            abpoa_add_graph_alignment(ab, abpt, bseq, seq_l, res.n_cigar, res.graph_cigar, read_id++, tot_n+n_seqs);     \
			fprintf(file_graph_node, "graph :%d\n", i);\
			print_graph(ab, file_graph_node);\
            if (res.n_cigar) free(res.graph_cigar); \
        }   \
		fclose(file_graph_node);\
        tot_n += n_seqs;    \
    }   \
    /* generate consensus from graph */ \
    if (abpt->out_cons) {   \
        abpoa_generate_consensus(ab, abpt, tot_n, stdout, NULL, NULL, NULL); \
    }   \
    /* generate multiple sequence alignment */  \
    if (abpt->out_msa) {  \
        uint8_t **msa_seq; int msa_l;   \
        /* abpoa_generate_rc_msa(ab, tot_n, stdout, NULL, NULL);   */ \
        abpoa_generate_rc_msa(ab, tot_n, stdout, &msa_seq, &msa_l);   \
        if (msa_l) {    \
            for (i = 0; i < tot_n; ++i) free(msa_seq[i]); free(msa_seq);    \
        }   \
    }   \
    /* generate dot plot */     \
    if (abpt->out_pog) {    \
        abpoa_dump_pog(ab, abpt);  \
    }   \
    ks_destroy(fs); gzclose(readfp);    \
}

int abpoa_main(const char *list_fn, int in_list, abpoa_para_t *abpt){
    kseq_t *read_seq = (kseq_t*)calloc(CHUNK_READ_N, sizeof(kseq_t));
    int bseq_m = 1024; uint8_t *bseq = (uint8_t*)_err_malloc(bseq_m * sizeof(uint8_t));
    int i, j, n_seqs, tot_n, read_id;
    // TODO abpoa_init for each input file ???
    abpoa_t *ab = abpoa_init();
	/* cudaEvent_t start, end; */
	/* checkCudaErrors(cudaEventCreate(&start)); */
	/* checkCudaErrors(cudaEventCreate(&end)); */
	/* checkCudaErrors(cudaEventRecord(start, 0)); */

	checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    if (in_list) { // input file list
        FILE *list_fp = fopen(list_fn, "r"); char read_fn[1024];
        while (fgets(read_fn, sizeof(read_fn), list_fp)) {
            read_fn[strlen(read_fn)-1] = '\0';
            abpoa_core(read_fn);
        }
        fclose(list_fp);
    } else { // input file
        abpoa_core(list_fn);
    }


	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
	fprintf(stderr, "time: %lf ms\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	/* checkCudaErrors(cudaEventRecord(end, 0)); */
	/* float elapsedTime; */
	/* checkCudaErrors(cudaEventSynchronize(end)); */
	/* checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end)); */
	/* fprintf(stderr, "time: %lf ms\n", elapsedTime); */
	/* checkCudaErrors(cudaEventDestroy(start)); */
	/* checkCudaErrors(cudaEventDestroy(end)); */
    free(bseq);
    for (i = 0; i < CHUNK_READ_N; ++i) { free((read_seq+i)->name.s); free((read_seq+i)->comment.s); free((read_seq+i)->seq.s); free((read_seq+i)->qual.s); } free(read_seq); 
    abpoa_free(ab, abpt);
    return 0;
}

int main(int argc, char **argv) {
    int c, m, in_list=0; char *s; abpoa_para_t *abpt = abpoa_init_para();
    while ((c = getopt_long(argc, argv, "m:M:X:O:E:b:f:z:e:lo:r:g:a:dq:hv", abpoa_long_opt, NULL)) >= 0) {
        switch(c)
        {
            case 'm': m = atoi(optarg);
                      if (m != ABPOA_GLOBAL_MODE && m != ABPOA_EXTEND_MODE && m != ABPOA_LOCAL_MODE) { 
                          err_printf("Unknown alignment mode: %d.\n", m); return abpoa_usage();
                      } abpt->align_mode=m; break;
            case 'M': abpt->match = atoi(optarg); break;
            case 'X': abpt->mismatch = atoi(optarg); break;
            case 'O': abpt->gap_open1 = strtol(optarg, &s, 10); if (*s == ',') abpt->gap_open2 = strtol(s+1, &s, 10); break;
            case 'E': abpt->gap_ext1 = strtol(optarg, &s, 10); if (*s == ',') abpt->gap_ext2 = strtol(s+1, &s, 10); break;

            case 'b': abpt->wb = atoi(optarg); break;
            case 'f': abpt->wf = atof(optarg); break;
            case 'z': abpt->zdrop = atoi(optarg); break;
            case 'e': abpt->end_bonus= atoi(optarg); break;

            case 'l': in_list = 1; break;
            case 'o': if (strcmp(optarg, "-") != 0) {
                          if (freopen(optarg, "wb", stdout) == NULL)
                              err_fatal(__func__, "Failed to open the output file %s", optarg);
                      } break;
            case 'r': if (atoi(optarg) == ABPOA_OUT_CONS) abpt->out_cons = 1, abpt->out_msa = 0;
                      else if (atoi(optarg) == ABPOA_OUT_MSA) abpt->out_cons = 0, abpt->out_msa = 1;
                      else if (atoi(optarg) == ABPOA_OUT_BOTH) abpt->out_cons = abpt->out_msa = 1;
                      else err_printf("Error: unknown output result mode: %s.\n", optarg);
                      break;
            case 'g': abpt->out_pog= optarg; break;

            case 'a': abpt->cons_agrm = atoi(optarg); break;
            case 'd': abpt->is_diploid = 1; break; 
            case 'q': abpt->min_freq = atof(optarg); break;

            case 'h': return abpoa_usage();
            case 'v': printf("%s\n", VERSION); goto End; break;
            default:
                      err_printf("Error: unknown option: %s.\n", optarg);
                      return abpoa_usage();
                      break;
        }
    }

    if (argc - optind != 1) return abpoa_usage();

    abpoa_post_set_para(abpt);
    abpoa_main(argv[optind], in_list, abpt);

End:
    abpoa_free_para(abpt);
    return 0;
}
