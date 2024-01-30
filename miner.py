#!/usr/bin/python3

import multiprocessing
import subprocess
import argparse
import time
import json
import sys
import os


def get_args():
    parser = argparse.ArgumentParser(description="SHA-256 custom miner")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--show-eta', action='store_true')
    parser.add_argument('--start', type=str, default='0',
                        help='Starting point')
    parser.add_argument('--fmt', type=str, default='tigor %d',
                        help='sprintf format for one number')
    parser.add_argument('solver', choices=[
                        'deadbeef', 'haiku', 'tigor', 'wordlist'])
    return parser.parse_args()


def get_eta_table():
    pascal = [1]

    for _ in range(101):
        pascal = [x + y for x, y in zip([0] + pascal, pascal + [0])]

    total = sum(pascal)

    return [(lvl, total / sum(pascal[67 + (lvl - 1) // 3:])) for lvl in range(1, 70, 3)]


def build(binary, source, gpu):
    print(f"Building {binary}")

    if gpu:
        command = ["nvcc", "-o", binary, source]
    else:
        is_arm = "ARM" in os.uname().machine.upper()
        arch_flags = [
            "-march=armv8-a+crypto"] if is_arm else ["-msse4.1", "-msha"]
        command = ["gcc", "-Ofast", f"-DNUM_CORES={multiprocessing.cpu_count()}"] + \
            arch_flags + ["-o", binary, source]

    print(" ".join(command))
    subprocess.run(command)


def get_stdin(solver):
    if solver == "haiku":
        return subprocess.Popen([sys.executable, "./src/haiku.py", "--offline"], stdout=subprocess.PIPE).stdout
    return None


def clear_line():
    sys.stdout.write(f"\033[{1}A")
    sys.stdout.write('\r\033[K')


def fmt_hashes(n):
    if n > 1e12:
        return f"{n / 1e12:.2f} TH"
    if n > 1e9:
        return f"{n / 1e9:.2f} GH"
    if n > 1e6:
        return f"{n / 1e6:.2f} MH"
    if n > 1e3:
        return f"{n / 1e3:.2f} KH"
    return f"{n:.2f} H"


def fmt_sec(s):
    return f"{s}s"


def get_eta(caclced, rate):
    if rate == 0:
        return "avg ETA unknown"

    for lvl, hashes in eta_table:
        if hashes > caclced:
            return f"avg ETA {int((hashes - caclced) / rate)}s to lvl {lvl}"
    return "avg ETA unknown"


def get_run_command(binary, args):
    if args.solver == "tigor" and not args.gpu:
        return [binary, args.start, args.fmt]
    return [binary]


def run(command, stdin=None):
    try:
        logfile = open(f"./logs/{int(time.time())}.log", 'a', encoding='utf-8')
        print(" ".join(command))
        process = subprocess.Popen(
            command, stdin=stdin, stdout=subprocess.PIPE, text=True)

        started_at = int(time.time())
        max_exponent = -200
        last_reported_at = 0
        found = None
        hashes_processed = {}

        print("Starting")

        def report(found):
            total_hashes = sum(hashes_processed.values())
            total_time = int(time.time()) - started_at

            rate = fmt_hashes(0 if total_time ==
                              0 else total_hashes / total_time)
            running = fmt_sec(total_time)
            processed = fmt_hashes(total_hashes)
            eta = ""
            if args.show_eta:
                eta = ", " + ("avg ETA unknown" if total_time ==
                              0 else get_eta(total_hashes, total_hashes / total_time))

            clear_line()

            if found:
                print(
                    f"Found lvl {found[0]} after {running} at {processed}, seed: {found[1]}")
                found = None

            print(
                f"Hashrate {rate}/s, processed {processed}, running {running}{eta}")
            sys.stdout.flush()

        for line in process.stdout:
            logfile.write(line)
            logfile.flush()
            data = json.loads(line)

            hashes_processed[data['worker']] = data['hashesProcessed']
            if data['exponent'] > max_exponent:
                max_exponent = data['exponent']
                found = (data['exponent'], json.dumps(
                    data['seed'], ensure_ascii=False))

            now = int(time.time())
            if now > last_reported_at:
                last_reported_at = now
                report(found)
                found = None

        if data['exponent'] > max_exponent:
            max_exponent = data['exponent']
            found = (data['exponent'], json.dumps(
                data['seed'], ensure_ascii=False))

        report(found)

    except KeyboardInterrupt:
        pass


subprocess.run(["mkdir", "-p", "./bin/", "./logs/"])

args = get_args()
eta_table = get_eta_table()

solver = args.solver + ("-gpu" if args.gpu else "-cpu")
binary = f"./bin/{solver}"
source = f"./src/{solver}" + (".cu" if args.gpu else ".c")

build(binary, source, args.gpu)
stdin = get_stdin(args.solver)
run(get_run_command(binary, args), stdin)
