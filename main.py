#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
from har_config import args

from servers.serverhar import Server

warnings.simplefilter("ignore")
torch.manual_seed(0)

def print_device(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def run(args):

    name = 'mmfedavg_k=7'
    results_root = os.path.join('results', name)

    if not os.path.exists(results_root):
        os.makedirs(results_root)

    print_device(args)
    server = Server(args)
    acc_list = server.train()


if __name__ == "__main__":
    total_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    run(args)


