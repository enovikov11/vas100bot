#include "miner.c"

#define LINE_SIZE 64
#define MAX_COUNT 100000

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

int main()
{
    init_calc();
    int linesCount = 0;

    char seed[SEED_SIZE];
    char *lines1 = malloc(MAX_COUNT * LINE_SIZE * sizeof(char));
    char *lines2 = malloc(MAX_COUNT * LINE_SIZE * sizeof(char));
    char *lines3 = malloc(MAX_COUNT * LINE_SIZE * sizeof(char));

    while (linesCount < MAX_COUNT)
    {
        if (read(lines1 + linesCount * LINE_SIZE) == -1)
            break;

        if (read(lines2 + linesCount * LINE_SIZE) == -1)
            break;

        if (read(lines3 + linesCount * LINE_SIZE) == -1)
            break;

        linesCount++;

        for (int i = 0; i < linesCount; i++)
        {
            for (int j = 0; j < linesCount; j++)
            {
                int k = 0;
                if (i < linesCount - 1 || j < linesCount - 1)
                {
                    k = linesCount - 1;
                }

                for (; k < linesCount; k++)
                {
                    snprintf(seed, SEED_SIZE, "%s\n%s\n%s",
                             lines1 + i * LINE_SIZE,
                             lines2 + j * LINE_SIZE,
                             lines3 + k * LINE_SIZE);
                    try_seed((unsigned char *)seed, 0);
                }
            }
        }
    }

    report_result(0);

    free(lines1);
    free(lines2);
    free(lines3);

    return 0;
}
