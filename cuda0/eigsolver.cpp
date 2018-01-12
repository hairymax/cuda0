
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cublas_v2.h"
#include "cuComplex.h"
#include "cusolverDn.h"

#define ERROR_OPEN_FILE - 3

const int N = 396;
const int arr_size = N*N;

// ôóíęöč˙ ń÷čňűâŕíč˙ ýëĺěĺíňîâ ęîěďëĺęńíîăî âĺęňîđŕ čç áčíŕđíîăî ôŕéëŕ
void readComplexArrayFromFile(cuDoubleComplex * arr, const char * fname) {
	FILE *infile = NULL;
	infile = fopen(fname, "rb");
	if (infile == NULL) {
		printf("Error opening file %s", fname);
		_getch();
		exit(ERROR_OPEN_FILE);
	}
	
	for (int i = 0; i < arr_size; i++) {
		fread(&arr[i].x, sizeof(double), 1, infile);
		fread(&arr[i].y, sizeof(double), 1, infile);
		//printf("[%d,%d]:    Re = %26.18e,   Im = %26.18e\n", i/N+1, i%N+1, cuCreal(arr[i]), cuCimag(arr[i]));
	}
	fclose(infile); 
}

// Çŕďčńü ýëĺěĺíňîâ ńîáńňâĺííîăî âĺęňîđŕ, ńîîňâĺňńâóţůĺăî îďđĺäĺë¸ííîé ýíĺđăčč, â ňĺęńňîâűé ôâéë
void writeComplexArrayToFile(cuDoubleComplex * Varr, double * lambda, const int index, const char * fpath) {
	char ind_str[3]; itoa(index, ind_str, 10);
	char fname[50];
	strcpy(fname, fpath);
	strcat(fname, ind_str);
	strcat(fname, ".txt");
	
	FILE *outfile = NULL;
	outfile = fopen(fname, "w+");
	if (outfile == NULL) {
		printf("Error opening file %s", fname);
		_getch();
		exit(ERROR_OPEN_FILE);
	}
	fprintf(outfile, "Eigen vector for energy value = %26.18e\n\n", lambda[index-1]);
	for (int i = 0; i < N; i++) {
		fprintf(outfile, "%d\t%26.18e\t%26.18e\n", 
			i + 1, Varr[(index-1)*N+i].x, Varr[(index-1)*N+i].y);
	}
	fclose(outfile);
}

int main()
{	
	// ńîçäŕ¸ě ńîáűňč˙ äë˙ ďîäń÷¸ňŕ âđĺěĺíč âű÷čńëĺíč˙
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	// řŕă 1: ń÷čňűâŕĺě äŕííűĺ čç ôŕéëîâ
	printf("1. Reading data from files\n");
	cuDoubleComplex * A = new cuDoubleComplex[arr_size];
	cuDoubleComplex * B = new cuDoubleComplex[arr_size];

	// ÷ňĺíčĺ ěŕňđčö čç ôŕéëîâ
	readComplexArrayFromFile(A, "A.dat");
	readComplexArrayFromFile(B, "B.dat");
	
	cusolverDnHandle_t cusolverH = NULL;  // óęŕçŕňĺëü íŕ ńîäĺđćčěîĺ áčáëčîňĺęč cuSolverDN.
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; // đĺçóëüňŕň âűďîëĺíč˙ ôóíęöčč
	cudaError_t cudaStat1 = cudaSuccess;  // ńňŕňóń îďĺđŕöčé
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	int info_gpu = 0;   // 0 - ýňî őîđîřî
		
	cuDoubleComplex * V = new cuDoubleComplex[arr_size]; // Ńîáńňâĺííűĺ âĺęňîđű
	double W[N];     // Ńîáńňâĺííűĺ çíŕ÷ĺíč˙

	cuDoubleComplex *d_A = NULL;
	cuDoubleComplex *d_B = NULL;
	double *d_W = NULL;
	int *devInfo = NULL;
	cuDoubleComplex *d_work = NULL;
	int  lwork = 0;
	//printf("\nBEFORE hegvd: info_gpu = %d\n", info_gpu);
	
	// Âűçîâ đĺřŕňĺë˙

	// Ńîçäŕ¸ě óęŕçŕňĺëü íŕ cusolver/cublas
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// řŕă 2: ęîďčđîâŕíčĺ ěŕňđčö A č B íŕ âčäĺîęŕđňó
	printf("2. Copying matricies A and B to device\n");
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * arr_size);
	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(cuDoubleComplex) * arr_size);
	cudaStat3 = cudaMalloc((void**)&d_W, sizeof(double) * N);
	cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * arr_size, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, B, sizeof(cuDoubleComplex) * arr_size, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	// řŕă 3: çŕďđîń äë˙ đŕáî÷ĺăî ďđîńňđŕíńňâŕ ęîěŕíäű hegvd
	printf("3. Query working space of Zhegvd\n");
	cusolverEigMode_t jobz  = CUSOLVER_EIG_MODE_VECTOR; // VECTOR - Âű÷čńë˙ňü ŃÇ č ŃÂ, NOVECTOR - ňîëüęî ŃÇ.
	cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_LOWER;
	cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;      // đĺřŕĺňń˙ çŕäŕ÷ŕ ňčďŕ A*x = lambda*B*x
	cusolver_status = cusolverDnZhegvd_bufferSize( //cusolverDnDsyevd_bufferSize(
		cusolverH, itype, jobz,	uplo,
		N, d_A, N, d_B, N, d_W,
		&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	assert(cudaSuccess == cudaStat1);

	// řŕă 4: Âű÷čńëĺíčĺ
	printf("4. Calculating\n");
	cusolver_status = cusolverDnZhegvd( //Dsyevd(
		cusolverH, itype, jobz,	uplo,
		N, d_A, N, d_B, N, d_W,
		d_work,	lwork, devInfo);
	
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	// ęîďčđóĺě ďîëó÷ĺííűĺ äŕííűĺ ń âčäĺîęŕđňű íŕ őîńň
	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*N, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(cuDoubleComplex)*arr_size, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	//printf("AFTER hegvd: info_gpu = %d\n\n", info_gpu);
	assert(0 == info_gpu);

	// Îďđĺäĺë˙ĺě âđĺě˙, ďîíŕäîáčâřĺĺń˙ äë˙ đŕń÷¸ňîâ
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\nTime for calculating eigenvalues: %f ms\n\n", time);
		
	// Çŕďčńűâŕĺě ńîáńňâĺííűĺ çíŕ÷ĺíč˙ â ôŕéë
	FILE *outfile = NULL;
	outfile = fopen("eigenvalues.txt", "w+");
	if (outfile == NULL) {
		printf("Error opening file eigenvalues.txt");
		_getch();
		exit(ERROR_OPEN_FILE);
	}
	for (int i = 0; i < N; i++) {
		fprintf(outfile, "E[%d] = %26.18e\n", i + 1, W[i]);
	}
	fprintf(outfile, "\nCalculating time : %f ms\n", time);
	fclose(outfile);
	printf("Eigenvalues are written to a file eigenvalues.txt:\n\n");

	// Çŕďčńűâŕĺě ńîáńňâĺííűé âĺęňîđ äë˙ ýíĺđăčč ń çŕäŕííűě íîěĺđîě (ňđĺňčé ŕđăóěĺíň)
	writeComplexArrayToFile(V, W, 1, "vector");

	// îńâîáîćäŕĺě ďŕě˙ňü
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_A);
	if (d_W) cudaFree(d_W);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);

	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();

	return 0;
}
