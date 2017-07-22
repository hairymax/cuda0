
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cublas_v2.h"
#include "cuComplex.h"
#include "cusolverDn.h"

//#include "helper_cuda.h"
//#include "helper_cusolver.h"

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}

int main()
{
	cusolverDnHandle_t cusolverH = NULL;  // указатель на содержимое библиотеки cuSolverDN.
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; // результат выполения функции
	cudaError_t cudaStat1 = cudaSuccess;  // статус операций над
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	const int m = 3;
	const int lda = m;
	
   /*       | 3.5 0.5 0 |
	*   A = | 0.5 3.5 0 |
	*       | 0   0   2 |
	*/
	/*
	double A[lda*m] = { 3.5, 0.5, 0,
		                0.5, 3.5, 0, 
		                0,   0,   2 };
	//double lambda[m] = { 1.0, 3.0, 4.0, 6.0 };

	double V[lda*m]; // Собственные векторы
	double W[m];     // Собственные значения

	double *d_A = NULL;
	double *d_W = NULL;
	int *devInfo = NULL;
	double *d_work = NULL;
	int  lwork = 0;

	printf("A = (matlab base-1)\n");
	printMatrix(m, m, A, lda, "A");
	printf("=====\n");
	*/
	
	int info_gpu = 0;

	cuDoubleComplex A[lda*m];
	for (int i = 0; i < lda; i++) {
		for (int j = 0; j < m; j++) {
			A[i*m+j] = make_cuDoubleComplex((double)(i+j+5)/10, (double)(i-j)/10);
		}
	}
	cuDoubleComplex V[lda*m]; // Собственные векторы
	double W[m];     // Собственные значения

	cuDoubleComplex *d_A = NULL;
	double *d_W = NULL;
	int *devInfo = NULL;
	cuDoubleComplex *d_work = NULL;
	int  lwork = 0;
	 
	for (int i = 0; i < lda; i++) {
		for (int j = 0; j < m; j++) {
			printf("A[%d]:    Re = %f,  Im = %f \n", i*m+j, cuCreal(A[i*m + j]), cuCimag(A[i*m + j]));
		}
	}
	// Вызов решателя

	// шаг 1: create cusolver/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// шаг 2: копирование матриц A и B на видеокарту
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * lda * m);
	cudaStat2 = cudaMalloc((void**)&d_W, sizeof(double) * m);
	cudaStat3 = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	// шаг 3: запрос для рабочего пространства команды syevd
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR; // Вычислять только собственные значения.
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
	cusolver_status = cusolverDnZheevd_bufferSize( //cusolverDnDsyevd_bufferSize(
		cusolverH,
		jobz,
		uplo,
		m,
		d_A,
		lda,
		d_W,
		&lwork);

	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	assert(cudaSuccess == cudaStat1);

	// шаг 4: Вычисление
	cusolver_status = cusolverDnZheevd( //Dsyevd(
		cusolverH,
		jobz,
		uplo,
		m,
		d_A,
		lda,
		d_W,
		d_work,
		lwork,
		devInfo);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(cuDoubleComplex)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	printf("after syevd: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);

	printf("eigenvalue, ascending order\n");
	for (int i = 0; i < m; i++) {
		printf("W[%d] = %E\n", i + 1, W[i]);
	}

	
	/*
	printf("V = (matlab base-1)\n");
	printMatrix(m, m, V, lda, "V");
	printf("=====\n");

	// шаг 4: проверка
	double lambda_sup = 0;
	for (int i = 0; i < m; i++) {
		double error = fabs(lambda[i] - W[i]);
		lambda_sup = (lambda_sup > error) ? lambda_sup : error;
	}
	printf("|lambda - W| = %E\n", lambda_sup);
	*/
	// free resources
	if (d_A) cudaFree(d_A);
	if (d_W) cudaFree(d_W);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);

	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();

	return 0;
}
