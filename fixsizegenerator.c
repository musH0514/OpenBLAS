#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define MAKE_INPUT_DIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MAKE_INPUT_DIR(path) mkdir((path), 0755)
#endif

static int is_supported_size(size_t n) {
    return n == 16 || n == 64 || n == 128 || n == 256 || n == 1024 || n == 8192 || n == 4096 || n == 16384 || n == 65536;
}

static float random_float(void) {
    float scale = (float)rand() / (float)RAND_MAX;
    return scale * 200.0f - 100.0f;
}

static int write_matrix(FILE *out, size_t rows, size_t cols) {
    size_t i;
    size_t count;

    if (out == NULL || rows == 0 || cols == 0) {
        return -1;
    }

    count = rows * cols;
    for (i = 0; i < count; ++i) {
        if (i != 0) {
            fputc(' ', out);
        }
        fprintf(out, "%.6f", random_float());
    }
    fputc('\n', out);
    return 0;
}

int main(int argc, char **argv) {
    FILE *out;
    size_t n;
    unsigned int seed;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <size> [seed]\n", argv[0]);
        return 1;
    }

    n = (size_t)strtoull(argv[1], NULL, 10);
    if (!is_supported_size(n)) {
        fprintf(
            stderr,
            "Unsupported size %zu. Use one of: 16, 64, 128, 256, 1024, 4096, 16384, 65536.\n",
            n
        );
        return 1;
    }

    if (argc >= 3) {
        seed = (unsigned int)strtoul(argv[2], NULL, 10);
    } else {
        seed = (unsigned int)time(NULL) ^ (unsigned int)clock();
    }
    srand(seed);

    MAKE_INPUT_DIR("input");

    out = fopen("input/in.txt", "w");
    if (out == NULL) {
        fprintf(stderr, "Cannot open input/in.txt for writing.\n");
        return 1;
    }

    fprintf(out, "%zu %zu\n", n, n);
    if (write_matrix(out, n, n) != 0) {
        fclose(out);
        fprintf(stderr, "Failed to write matrix A.\n");
        return 1;
    }

    fprintf(out, "%zu %zu\n", n, n);
    if (write_matrix(out, n, n) != 0) {
        fclose(out);
        fprintf(stderr, "Failed to write matrix B.\n");
        return 1;
    }

    fclose(out);
    printf("seed=%u A=%zux%zu B=%zux%zu\n", seed, n, n, n, n);
    return 0;
}
