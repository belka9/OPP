#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<time.h>
#include<sys/time.h>

#define NUM_DIMS 2
int main(int argc, char **argv)
{
int M, N, K;
int sizeM[3];
M = 4;
K = M;
N = M;

MPI_Init(&argc, &argv);
int threadCount;
int threadRank;
MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
MPI_Comm_rank(MPI_COMM_WORLD, &threadRank);


int dims[NUM_DIMS] = {0, 0};
MPI_Dims_create(threadCount, NUM_DIMS, dims);

if (threadRank == 0) {
	fprintf(stderr, "dims[0] = %d, dims[1] = %d\n", dims[0], dims[1]);
}

double *A = new double[M * N];
double *B = new double[N * K];
double *C = new double[M * K];

if (threadRank == 0)
{
	sizeM[0] = M;
	sizeM[1] = N;
	sizeM[2] = K;

	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			A[N*i + j] = 1;
			}
	}

	for (int j = 0; j < N; ++j) {
		for (int k = 0; k < K; ++k) {
			B[K*j + k] = 1;
			}
		}
}
double startTime = MPI_Wtime();

MPI_Bcast(sizeM, 3, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

int periods[2] = { 0 };

MPI_Comm comm_2D;
MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm_2D);

int threadCoords[2];
MPI_Comm_rank(comm_2D, &threadRank);
MPI_Cart_coords(comm_2D, threadRank, NUM_DIMS, threadCoords);

MPI_Comm comm_1D[2];
int remains[2];

for (int i = 0; i < 2; i++) {
	for (int j = 0; j < 2; j++) {
		remains[j] = (i == j);
	}
	MPI_Cart_sub(comm_2D, remains, &comm_1D[i]);
}

int sizeAA, sizeBB;
sizeAA = sizeM[0] / dims[0]; 
sizeBB = sizeM[2] / dims[1];

double *AA, *BB, *CC;
AA = new double[sizeAA * sizeM[1]];
BB = new double[sizeM[1] * sizeBB];
CC = new double[sizeAA * sizeBB];


int *countc, *dispc, *countb, *dispb;
MPI_Datatype column, matrix;

//column
if (threadRank == 0) {

MPI_Type_vector(sizeM[1], sizeBB, sizeM[2], MPI_DOUBLE, &column);

MPI_Type_create_resized(column, 0, sizeBB * sizeof(double), &column);

MPI_Type_commit(&column);

//mini-matrix in С
MPI_Type_vector(sizeAA, sizeBB, sizeM[2], MPI_DOUBLE, &matrix);
MPI_Type_create_resized(matrix, 0, sizeBB * sizeof(double), &matrix);
MPI_Type_commit(&matrix);

dispb = new int[dims[1]];
countb = new int[dims[1]];
for (int j = 0; j < dims[1]; j++) {
	dispb[j] = j;
	countb[j] = 1;
}

dispc = new int[dims[0] * dims[1]];
countc = new int[dims[0] * dims[1]];
for (int i = 0; i < dims[0]; i++) {
	for (int j = 0; j < dims[1]; j++) {
		dispc[i * dims[1] + j] = (i * dims[1] * sizeAA + j);
		countc[i * dims[1] + j] = 1;
		}
	}
}
if (threadCoords[1] == 0) 
MPI_Scatter(A, sizeAA * sizeM[1], MPI_DOUBLE, AA, sizeAA * sizeM[1], MPI_DOUBLE, 0, comm_1D[0]);

if (threadCoords[0] == 0) 
MPI_Scatterv(B, countb, dispb, column, BB, sizeM[1] * sizeBB, MPI_DOUBLE, 0, comm_1D[1]);

MPI_Bcast(AA, sizeAA * sizeM[1], MPI_DOUBLE, 0, comm_1D[1]);

MPI_Bcast(BB, sizeM[1] * sizeBB, MPI_DOUBLE, 0, comm_1D[0]);

for (int i = 0; i < sizeAA; i++) {
	for (int j = 0; j < sizeBB; j++) {
		for (int k = 0; k < sizeM[1]; k++) {
			CC[sizeBB * i + j] = CC[sizeBB * i + j] + AA[sizeM[1] * i + k] * BB[sizeBB * k + j];
		}
	}
}

MPI_Gatherv(CC, sizeAA * sizeBB, MPI_DOUBLE, C, countc, dispc, matrix, 0, comm_2D);

delete[] AA;
delete[] BB;
delete[] CC;
MPI_Comm_free(&comm_2D);
for (int i = 0; i < 2; i++) {
	MPI_Comm_free(&comm_1D[i]);
}

if (threadRank == 0) {

delete[] countc;
delete[] dispc;
delete[] countb;
delete[] dispb;
MPI_Type_free(&column);
MPI_Type_free(&matrix);
}

double finishTime = MPI_Wtime();


if (threadRank == 0) {
int test;
int flag;
for (int i = 0; i < M; ++i) {
	for (int j = 0; j < K; ++j) {
		if( C[sizeBB * i + j] = N)
		// printf("%lf\n", C[sizeBB * i + j]); 
			test++;
	}
}
	if(test = M*K) flag = 1;
	fprintf(stderr, "Time:%lf\nTest:%d\n", finishTime - startTime, flag);
}

delete[] A;
delete[] B;
delete[] C;

MPI_Finalize();
return 0;
}
