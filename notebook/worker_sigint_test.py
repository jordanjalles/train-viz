"""CPU test: a kernel interrupt (SIGINT to the process group) kills DataLoader workers unless
the worker_init_fn tells them to ignore it.

    python worker_sigint_test.py UNIFY_NOTEBOOK.ipynb

Uses the notebook's seed_imagenet_worker cell, as pushed and with the two added lines.
"""
import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]
src = next(c for c in cells if c.lstrip().startswith("def seed_imagenet_worker"))
ADD = ("    # A kernel interrupt must stop the training loop, not the workers.\n"
       "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n")
head = "def seed_imagenet_worker(worker_id):\n"
assert src.count(head) == 1
fixed_src = "import signal\n\n" + src.replace(head, head + ADD)


def trial(source, label):
    g = dict(torch=torch, random=random, np=np)
    exec(source, g)
    loader = DataLoader(TensorDataset(torch.rand(64, 3, 8, 8), torch.zeros(64)), batch_size=8, num_workers=2,
                        persistent_workers=True, multiprocessing_context="fork", worker_init_fn=g["seed_imagenet_worker"])
    assert len([b for b in loader]) == 8
    pids = [w.pid for w in loader._iterator._workers]
    try:
        for pid in pids:
            os.kill(pid, signal.SIGINT)           # what the workers receive when the kernel is interrupted
        time.sleep(1.5)
    except RuntimeError:
        pass
    alive = [w.is_alive() for w in loader._iterator._workers]
    try:
        n = len([b for b in loader])
        result = f"next pass over the loader gives {n} batches"
    except RuntimeError as e:
        result = "next use fails: " + str(e).split(")")[-1].strip()
    print(f"{label}: workers alive after the interrupt: {alive} | {result}")
    return alive, result


a, ra = trial(src, "as pushed       ")
b, rb = trial(fixed_src, "with the 2 lines")
assert not any(a) and "exited unexpectedly" in ra
assert all(b) and "8 batches" in rb
print("ALL OK")
