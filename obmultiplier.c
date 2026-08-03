//gcc -O3 -I. obmultiplier.c -o ob -L. -l:libopenblas.so.0 -lpthread
/*
 * mygemm : a minimal program that reads two matrices from an input file
 *          and computes C = A * B using OpenBLAS (cblas_dgemm), with timing.
 *
 * Input file format (plain text):
 *   line 1 : m n k            (dimensions)
 *   next   : m*k doubles, the A matrix in ROW-MAJOR order
 *   next   : k*n doubles, the B matrix in ROW-MAJOR order
 *
 * Usage:
 *   mygemm.exe < input.in
 *   mygemm.exe input.in
 *
 * Timing: warmup + repeats (OPENBLAS_LOOPS, default 3), reports best time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

struct Matrix { 
    size_t rows; 
    size_t cols; 
    double * data; 
};


static double *matrix_at(struct Matrix *m, size_t row, size_t col) {
    if (m == NULL || m->data == NULL || row >= m->rows || col >= m->cols) {
        return NULL;
    }
    return &m->data[row * m->cols + col];
}

static int matrix_init(struct Matrix *m, size_t rows, size_t cols) {
    size_t count;

    if (m == NULL || rows == 0 || cols == 0) {
        return -1;
    }

    if (rows > ((size_t)-1) / cols) {
        return -1;
    }
    count = rows * cols;

    m->data = (double *)calloc(count, sizeof(double));
    if (m->data == NULL) {
        m->rows = 0;
        m->cols = 0;
        return -1;
    }

    m->rows = rows;
    m->cols = cols;
    return 0;
}

static int matrix_read(struct Matrix *m, FILE *in) {
    size_t i;
    size_t j;

    if (m == NULL || in == NULL || m->data == NULL) {
        return -1;
    }

    for (i = 0; i < m->rows; i++) {
        for (j = 0; j < m->cols; j++) {
            double *slot = matrix_at(m, i, j);
            if (slot == NULL) {
                return -1;
            }
            if (fscanf(in, "%lf", slot) != 1) {   //读入一个浮点数到slot位置
                return -1;
            }
        }
    }
    return 0;
}

static void matrix_destroy(struct Matrix *m) {
    if (m == NULL) {
        return;
    }

    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char *argv[]) {
    const char *input_path = "input/in.txt";
    FILE *input_file;
    struct Matrix l = {0, 0, NULL};
    struct Matrix r = {0, 0, NULL};
    struct Matrix rs = {0, 0, NULL};
    size_t a_rows;
    size_t a_cols;
    size_t b_rows;
    size_t b_cols;
  
    input_file = fopen(input_path, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Cannot open %s.\n", input_path);
        return 1;
    }

    if (fscanf(input_file, "%zu %zu", &a_rows, &a_cols) != 2) {
        fprintf(stderr, "Invalid matrix A size.\n");
        matrix_destroy(&l);
        fclose(input_file);
        return 1;
    }
    if (matrix_init(&l, a_rows, a_cols) != 0) {
        fprintf(stderr, "Cannot allocate matrix A.\n");
        matrix_destroy(&l);
        fclose(input_file);
        return 1;
    }
    if (matrix_read(&l, input_file) != 0) {
        fprintf(stderr, "Invalid matrix A input.\n");
        matrix_destroy(&l);
        fclose(input_file);
        return 1;
    }

    if (fscanf(input_file, "%zu %zu", &b_rows, &b_cols) != 2) {
        fprintf(stderr, "Invalid matrix B size.\n");
        matrix_destroy(&r);
        fclose(input_file);
        return 1;
    }
    if (matrix_init(&r, b_rows, b_cols) != 0) {
        fprintf(stderr, "Cannot allocate matrix B.\n");
        matrix_destroy(&r);
        fclose(input_file);
        return 1;
    }
    if (matrix_read(&r, input_file) != 0) {
        fprintf(stderr, "Invalid matrix B input.\n");
        matrix_destroy(&r);
        fclose(input_file);
        return 1;
    }

    fclose(input_file);

    if (a_cols != b_rows) {
        fprintf(stderr, "Matrix sizes are incompatible: A is %zux%zu, B is %zux%zu.\n", a_rows, a_cols, b_rows, b_cols);
        matrix_destroy(&l);
        matrix_destroy(&r);
        return 1;
    }

    double *a = l.data;
    double *b = r.data;
    if (matrix_init(&rs, a_rows, b_cols) != 0) {
        fprintf(stderr, "Cannot allocate matrix C.\n");
        matrix_destroy(&l);
        matrix_destroy(&r);
        return 1;
    }
    double *c = rs.data;
    if (!c) {
        fprintf(stderr, "out of memory\n"); 
        return 1;
    }

    double alpha = 1.0, beta = 0.0;

    /* warmup: force library init, thread pool, page faults */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols, a_cols, alpha, a, a_cols, b, b_cols, beta, c, b_cols);

    long long startTime = now_ns();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols, a_cols, alpha, a, a_cols, b, b_cols, beta, c, b_cols);
    long long endTime = now_ns();
    long long duration = endTime - startTime;
    printf("duration = %lldns\n", duration);

    // double flops = 2.0 * (double)a_rows * b_cols * a_cols;
    // fprintf(stderr, "C = A*B, m=%d n=%d k=%d, loops=%d\n", a_rows, b_cols, a_cols, loops);
    // fprintf(stderr, "best time : %.6f s\n", best);
    // fprintf(stderr, "perf      : %.3f GFLOPs\n", flops / best / 1e9);

    /* verification checksum: C[0][0] and a couple of others */
    // fprintf(stderr, "C[0][0]=%.6f  C[m-1][n-1]=%.6f  sum|C|=%.6f\n",
    //         c[0], c[(size_t)a_rows * b_cols - 1],
    //         c[0] + c[(size_t)a_rows * b_cols - 1] + c[(size_t)a_rows * b_cols / 2]);

    matrix_destroy(&l);
    matrix_destroy(&r);
    matrix_destroy(&rs);
    return 0;
}
