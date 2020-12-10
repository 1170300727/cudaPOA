#include <stdio.h>
#include <stdlib.h>

void CUDA_SHIFT_RIGHTi32(int32_t *array, int32_t length, int32_t number){
	if (number > length - 1) 
		fprintf(stderr, "shift number is to large!\n");
	// int32_t temp = *(array + length - 1);
	int i;
	for (i = 0; i < number; i++){
		array[i] = 0;
	}
	for (i = number; i < length; i++){
		array[i] = array[i - number];
	}
}

void CUDA_SHIFT_LEFTi32(int32_t *array, int32_t length, int32_t number){
	if (number > length - 1) 
		fprintf(stderr, "shift number is to large!\n");
	// int32_t temp = *(array + length - 1);
	int i;
	for (i = length - number - 1; i < length - 1; i++){
		array[i] = 0;
	}
	for (i = 0; i < number; i++){
		array[i] = array[i + number];
	}
}

int32_t* CUDA_ORi32(int32_t *array_a, int32_t *array_b, int32_t length){
	int i;
	int32_t *re = (int32_t *)malloc(sizeof(int32_t) * length);
	for(i = 0; i < length; i++){
		re[i] = array_a[i] | array_b[i];
	}
	return re;
}
