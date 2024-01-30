import argparse
import aiohttp
import asyncio
import sys


async def fetch_haiku(session):
    async with session.get("http://nevmenandr.net/cgi-bin/haiku.html") as resp:
        text = await resp.text()
        haiku = "\n".join(text.split("\n")[119:122]) \
            .replace("</span></td></tr>", "") \
            .replace('<tr><td></td><td><span style="color: #363636; font: normal 1.8em/1.36 Georgia">', "")
        return haiku + "\0"


async def main():
    with open("./data/haikus.txt", "r") as file:
        sys.stdout.write(file.read())

    if args.offline:
        return

    async with aiohttp.ClientSession() as session:
        with open("./data/haikus.txt", "a") as file:
            while True:
                try:
                    haiku = await fetch_haiku(session)

                    sys.stdout.write(haiku)
                    sys.stdout.flush()

                    file.write(haiku)
                    file.flush()
                except Exception:
                    continue

parser = argparse.ArgumentParser(description="Haiku")
parser.add_argument('--offline', action='store_true', help='Do not fetch new')
args = parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())
