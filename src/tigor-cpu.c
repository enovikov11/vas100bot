#include "miner.c"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    init_calc();
    unsigned char seed[SEED_SIZE];
    long long d = 0;

    if (argc > 1)
    {
        d = atoi(argv[1]);
    }

    char *fmt = "tigor %d";

    if (argc > 2)
    {
        fmt = argv[2];
    }

    for (int i = 0; i < NUM_CORES; i++)
    {
        pid_t pid = fork();

        if (pid > 0)
        {
            d += i;
            while (1)
            {
                sprintf((char *)seed, fmt, d);
                try_seed(seed, (int)pid);
                d += NUM_CORES;
            }
        }
    }
}
