#include <stdio.h>
#include <time.h>
#include <string.h>
#include "../lib/sha256.c"

#define SEED_SIZE 1024
#define MIN_EXPONENT -200

#ifndef NUM_CORES
#define NUM_CORES 1
#endif

// 100 = 12 * 8 + 4
int bits_count_8[256];
int bits_count_4[256];

long hashesCalced = 0, lastReportedAt = 0;
int max_exponent = MIN_EXPONENT;
unsigned char maxseed[SEED_SIZE];

void init_calc()
{
    for (int i = 0; i < 256; i++)
    {
        bits_count_8[i] = 0;
        for (int j = 0; j < 8; j++)
        {
            if ((1 << j) & i)
            {
                bits_count_8[i]++;
            }
        }

        bits_count_4[i] = 0;
        for (int j = 4; j < 8; j++)
        {
            if ((1 << j) & i)
            {
                bits_count_4[i]++;
            }
        }
    }
}

int calc_exponent(unsigned char *seed)
{
    SHA256_CTX ctx;
    BYTE hash[32];

    sha256_init(&ctx);
    sha256_update(&ctx, seed, strlen((char *)seed));
    sha256_final(&ctx, hash);

    int ones = bits_count_8[hash[0]] + bits_count_8[hash[1]] + bits_count_8[hash[2]] + bits_count_8[hash[3]] +
               bits_count_8[hash[4]] + bits_count_8[hash[5]] + bits_count_8[hash[6]] + bits_count_8[hash[7]] +
               bits_count_8[hash[8]] + bits_count_8[hash[9]] + bits_count_8[hash[10]] + bits_count_8[hash[11]] +
               bits_count_4[hash[12]];

    return ones - 2 * (100 - ones);
}

void print_seed(unsigned char *seed)
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

void report_result(int worker_id)
{
    struct timespec currentTime;
    clock_gettime(CLOCK_REALTIME, &currentTime);

    printf("{\"exponent\": %d, \"hashesProcessed\": %ld, \"unixtime\": %ld, \"worker\": %d, \"seed\": \"",
           max_exponent, hashesCalced, currentTime.tv_sec, worker_id);
    print_seed(maxseed);
    printf("\"}\n");
    fflush(stdout);

    lastReportedAt = currentTime.tv_sec;
    max_exponent = MIN_EXPONENT;
}

void try_seed(unsigned char *seed, int worker_id)
{
    int exponent = calc_exponent(seed);
    hashesCalced++;

    if (exponent > max_exponent)
    {
        max_exponent = exponent;
        memcpy(maxseed, seed, SEED_SIZE);
    }

    struct timespec currentTime;
    clock_gettime(CLOCK_REALTIME, &currentTime);

    if (currentTime.tv_sec > lastReportedAt)
    {
        report_result(worker_id);
    }
}
