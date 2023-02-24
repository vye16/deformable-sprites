import os
from datetime import datetime
import subprocess
import multiprocessing as mp
from concurrent import futures

import argparse


"""
Script for launching multiple optimization jobs.
Specify either through a job file (-f), example usage:

    python launch.py --gpus 0 1 --seqs bear dog

or listing sequences (--seqs), example usage:

    python launch.py --gpus 0 1 -f job_specs/davis_val.txt

"""


def run(gpus, log_file, seq, overwrite=False):
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    worker_id = cur_proc._identity[0] - 1  # 1-indexed processes
    gpu = gpus[worker_id % len(gpus)]
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} python run_opt.py "
    )
    if seq is not None:
        cmd = f"{cmd} data.seq={seq}"
    if overwrite:
        cmd = f"{cmd} overwrite=True"

    print(f"LOGGING TO {log_file}")
    cmd = f"{cmd} > {log_file} 2>&1"
    print(cmd)
    subprocess.call(cmd, shell=True)


def main(args):
    seqs = args.seqs
    if seqs is None:
        seqs = [None]
    if args.job_file is not None:
        with open(args.job_file, "r") as f:
            seqs = [args.strip() for args in f.readlines()]
    print(seqs)

    os.makedirs(args.log_dir, exist_ok=True)
    job_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as exe:
        for i, seq in enumerate(seqs):
            log_file = f"{args.log_dir}/{job_name}_{i:03d}.log"
            exe.submit(
                run,
                args.gpus,
                log_file,
                seq,
                args.opt,
                args.vis,
                args.overwrite,
                args.argstr,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="*", default=[8])
    parser.add_argument("-f", "--job_file", default=None)
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("-y", "--overwrite", action="store_true")
    parser.add_argument("-l", "--log_dir", default="_launch_logs")
    args = parser.parse_args()

    main(args)
