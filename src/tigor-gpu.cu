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

__global__ void task(char *seeds, int *exps, long long run, int cores)
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
        seed[len++] = 't';
        seed[len++] = 'i';
        seed[len++] = 'g';
        seed[len++] = 'o';
        seed[len++] = 'r';
        seed[len++] = ' ';

        long long ii = i, isize = 0;

        while (ii > 0)
        {
            ii /= 10;
            isize++;
        }

        long long jsize = isize;

        ii = i;
        while (ii > 0)
        {
            seed[len + isize - 1] = '0' + (ii % 10);
            ii /= 10;
            isize--;
        }

        len += jsize;

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

            for (int i = 0; i < len; i++)
            {
                seeds[SEED_SIZE * index + i] = seed[i];
            }
            seeds[SEED_SIZE * index + len] = '\0';
        }

        i += cores;
    }
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

    char *seeds, *d_seeds;
    int *exps, *d_exps;

    seeds = (char *)malloc(cores * SEED_SIZE * sizeof(char));
    exps = (int *)malloc(cores * sizeof(int));

    cudaMalloc(&d_seeds, cores * SEED_SIZE * sizeof(char));
    cudaMalloc(&d_exps, cores * sizeof(int));

    for (long long run = 0;; run++)
    {
        task<<<prop.maxBlocksPerMultiProcessor, prop.maxThreadsPerBlock>>>(d_seeds, d_exps, run, cores);

        cudaDeviceSynchronize();
        cudaMemcpy(seeds, d_seeds, cores * SEED_SIZE * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(exps, d_exps, cores * sizeof(int), cudaMemcpyDeviceToHost);

        int max_exp = exps[0], max_i = 0;

        for (int i = 1; i < cores; i++)
        {
            if (exps[i] > max_exp)
            {
                max_exp = exps[i];
                max_i = i;
            }
        }

        struct timespec currentTime;
        clock_gettime(CLOCK_REALTIME, &currentTime);

        printf("{\"exponent\": %d, \"hashesProcessed\": %ld, \"unixtime\": %ld, \"worker\": 0, \"seed\": \"",
               max_exp, (long int)(run + 1) * cores * TIMES, currentTime.tv_sec);
        print_seed(seeds + max_i * SEED_SIZE);
        printf("\"}\n");
        fflush(stdout);
    }

    return 0;
}
