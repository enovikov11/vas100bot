# Симулятор игры Василия

Есть игровой автомат, который подбрасывает монетку 100 раз, умоножая ставку в x2 или в x0.25 раз. Хоть у игры матожидание больше 1, реально выиграть в нее практически невозможно.

## Телеграм бот @vas100bot

Это симулятор игры Василия, ты ему текст, он берет от него SHA-256 и первые 100 бит это и есть монтека.

### Запуск

В папке `bot` лежит докер, в .env кладем `API_KEY=` и запускаем `docker-compose build` + `docker-compose up -d`

## Майнер

Перебирает хеши в поисках такого, который дает больше всего выигрыша.

### Хешрейты tigor воркера

CPU
- Apple Silicon M1 (8 workers): 45 MH/s (5.6 MH/s/core)
- AMD Ryzen 5 3600 (12 workers): 55 MH/s (4.6 MH/s/core)
- Intel Ice Lake (96 workers): 230 MH/s (2.4 MH/s/core)
- Intel Broadwell (32 workers): 31 MH/s (1.0 MH/s/core)
- Intel Cascade Lake (80 workers): 71 MH/s (0.9 MH/s/core)

GPU
- Nvidia RTX 3070 LHR (16 384 workers): 600 MH/s (38 KH/s/core)