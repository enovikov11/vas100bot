#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/sha256.cuh"

#define SEED_SIZE 1024
#define MIN_EXPONENT -200
#define TIMES 100000

#define LINE_SIZE 64
#define MAX_COUNT 100000

__global__ void task(int *exps, long long *ids, char *lines1, char *lines2, char *lines3, long long linesCount, long long run, int cores)
{
    // 100 = 12 * 8 + 4
    const int bits_count_8[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    const int bits_count_4[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    long long i = index + cores * TIMES * run;
    char seed[SEED_SIZE];
    BYTE hash[32];
    exps[index] = MIN_EXPONENT;

    for (int times = 0; times < TIMES; times++)
    {
        int len = 0;

        long long ii = i;

        long long pos1 = ii % linesCount;
        ii /= linesCount;
        long long pos2 = ii % linesCount;
        ii /= linesCount;
        long long pos3 = ii % linesCount;
        ii /= linesCount;

        if (ii > 0)
        {
            return;
        }

        char *str;

        str = lines1 + pos1 * LINE_SIZE;
        while (*str != '\0')
        {
            seed[len++] = *str;
            str += 1;
        }

        seed[len++] = '\n';

        str = lines2 + pos2 * LINE_SIZE;
        while (*str != '\0')
        {
            seed[len++] = *str;
            str += 1;
        }

        seed[len++] = '\n';

        str = lines3 + pos3 * LINE_SIZE;
        while (*str != '\0')
        {
            seed[len++] = *str;
            str += 1;
        }

        SHA256_CTX ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, (const BYTE *)seed, len);
        sha256_final(&ctx, hash);

        int ones = bits_count_8[hash[0]] + bits_count_8[hash[1]] + bits_count_8[hash[2]] + bits_count_8[hash[3]] +
                   bits_count_8[hash[4]] + bits_count_8[hash[5]] + bits_count_8[hash[6]] + bits_count_8[hash[7]] +
                   bits_count_8[hash[8]] + bits_count_8[hash[9]] + bits_count_8[hash[10]] + bits_count_8[hash[11]] +
                   bits_count_4[hash[12]];

        int exp = ones - 2 * (100 - ones);

        if (exp > exps[index])
        {
            exps[index] = exp;
            ids[index] = i;
        }

        i += cores;
    }
}

int read(char *line)
{
    size_t len = 0;
    int charRead;

    while (len < LINE_SIZE - 1 && (charRead = getchar()) != EOF && charRead != '\0' && charRead != '\n')
        *(line + len++) = charRead;

    if (len == 0)
        return -1;

    *(line + len) = '\0';
    return 0;
}

void print_seed(char *seed)
{
    while (*seed != '\0')
    {
        if (*seed == '\n')
        {
            putchar('\\');
            putchar('n');
        }
        else if (*seed == '"')
        {
            putchar('\\');
            putchar('"');
        }
        else
        {
            putchar(*seed);
        }

        seed++;
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock;

    long long linesCount = 0;
    char *lines1 = (char *)malloc(MAX_COUNT * LINE_SIZE * sizeof(char));
    char *lines2 = (char *)malloc(MAX_COUNT * LINE_SIZE * sizeof(char));
    char *lines3 = (char *)malloc(MAX_COUNT * LINE_SIZE * sizeof(char));

    int *exps, *d_exps;
    long long *ids, *d_ids;
    char *d_lines1, *d_lines2, *d_lines3;

    while (linesCount < MAX_COUNT)
    {
        if (read(lines1 + linesCount * LINE_SIZE) == -1)
            break;

        if (read(lines2 + linesCount * LINE_SIZE) == -1)
            break;

        if (read(lines3 + linesCount * LINE_SIZE) == -1)
            break;

        linesCount++;
    }

    exps = (int *)malloc(cores * sizeof(int));
    cudaMalloc(&d_exps, cores * sizeof(int));

    ids = (long long *)malloc(cores * sizeof(long long));
    cudaMalloc(&d_ids, cores * sizeof(long long));

    cudaMalloc(&d_lines1, MAX_COUNT * LINE_SIZE * sizeof(char));
    cudaMalloc(&d_lines2, MAX_COUNT * LINE_SIZE * sizeof(char));
    cudaMalloc(&d_lines3, MAX_COUNT * LINE_SIZE * sizeof(char));

    cudaMemcpy(d_lines1, lines1, MAX_COUNT * LINE_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lines2, lines2, MAX_COUNT * LINE_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lines3, lines3, MAX_COUNT * LINE_SIZE * sizeof(char), cudaMemcpyHostToDevice);

    long long lineCount3 = linesCount * linesCount * linesCount;

    for (long long run = 0; run * cores * TIMES < lineCount3; run++)
    {
        task<<<prop.maxBlocksPerMultiProcessor, prop.maxThreadsPerBlock>>>(d_exps, d_ids, d_lines1, d_lines2, d_lines3, linesCount, run, cores);

        cudaDeviceSynchronize();
        cudaMemcpy(exps, d_exps, cores * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids, d_ids, cores * sizeof(long long), cudaMemcpyDeviceToHost);

        int max_exp = MIN_EXPONENT;
        long long max_i = 0;

        for (int i = 0; i < cores; i++)
        {
            if (exps[i] > max_exp)
            {
                max_exp = exps[i];
                max_i = ids[i];
            }
        }

        struct timespec currentTime;
        clock_gettime(CLOCK_REALTIME, &currentTime);

        printf("{\"exponent\": %d, \"hashesProcessed\": %ld, \"unixtime\": %ld, \"worker\": 0, \"seed\": \"",
               max_exp, (long int)(run + 1) * cores * TIMES, currentTime.tv_sec);

        char seed[SEED_SIZE];
        long long ii = max_i;

        int pos1 = ii % linesCount;
        ii /= linesCount;
        int pos2 = ii % linesCount;
        ii /= linesCount;
        int pos3 = ii % linesCount;

        snprintf(seed, SEED_SIZE, "%s\n%s\n%s",
                 lines1 + pos1 * LINE_SIZE,
                 lines2 + pos2 * LINE_SIZE,
                 lines3 + pos3 * LINE_SIZE);

        print_seed(seed);

        printf("\"}\n");
        fflush(stdout);
    }

    return 0;
}
