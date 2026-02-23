"""
Computational performance benchmarking.

Compares inference time and VRAM usage between sparse and dense
architectures across different resolutions and devices.
"""

import os
import time
import gc
import numpy as np
import torch
import tqdm
import pickle as pk
from itertools import islice
from torch.utils.data import DataLoader
from dataset import SparseDataset
from utils import (
    ini_argparse,
    collate_sparse_minkowski,
    replace_depthwise_with_channelwise,
    arrange_sparse_minkowski,
    arrange_truth,
)
from model import MinkUNetConvNeXtV2, DenseUNetConvNeXtV2


def setup_dataset(args, stage2, min_hu, max_hu, dataset_path, dataset_name):
    args.stage2 = stage2
    args.min_hu = min_hu
    args.max_hu = max_hu
    args.dataset_path = dataset_path
    args.dataset_name = dataset_name
    return SparseDataset(args)


def create_dataloader(dataset):
    return DataLoader(
        dataset, batch_size=1, num_workers=1,
        pin_memory=True, persistent_workers=True,
        collate_fn=collate_sparse_minkowski, shuffle=False,
    )


def _arrange_batch(batch, device, sparse):
    batch_input = arrange_sparse_minkowski(batch, device)
    batch_target = arrange_truth(batch, device)
    if not sparse:
        batch_input = batch_input.dense()[0]
        batch_target = batch_target.dense()[0]
    return batch_input, batch_target


def measure_inference(model, loader, device, sparse,
                      warmup_iters=5, use_gpu=False):
    model.to(device)

    print("  Warm-up...")
    with torch.no_grad():
        for batch in tqdm.tqdm(islice(loader, warmup_iters), total=warmup_iters):
            batch_input, batch_target = _arrange_batch(batch, device, sparse)
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            _ = model(batch_input, batch_target)
            if use_gpu:
                torch.cuda.synchronize()

    print("  Measuring...")
    idx_list, times = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, total=len(loader)):
            batch_input, batch_target = _arrange_batch(batch, device, sparse)
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            start = time.perf_counter()
            _ = model(batch_input, batch_target)
            if use_gpu:
                torch.cuda.synchronize()
            end = time.perf_counter()

            idx_list.append(batch["idx"][0])
            times.append(end - start)

    return np.array(idx_list), np.array(times)


def run_tests(net, loader, device, label, sparse, use_gpu=False):
    idx, times = measure_inference(net, loader, device, sparse, use_gpu=use_gpu)
    print(f"  {label}: {times.mean():.4f} +/- {times.std():.4f} s")
    return {"idx": idx, "times": times}


def save_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as fd:
        pk.dump(results, fd)


def main():
    parser = ini_argparse()
    args = parser.parse_args()
    nb_gpus = len(args.gpus)
    gpus = ", ".join(str(g) for g in args.gpus)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    dataset_lowres = setup_dataset(args, False, -53.4, 283.2, "/scratch2/salonso/medical/medical_data/{}/*", "kits23_large_processed")
    dataset_highres = setup_dataset(args, True, -300, 500, "/scratch2/salonso/medical/medical_data/{}/*", "kits23_processed_highres11_all")

    print(len(dataset_lowres), len(dataset_highres))

    loader_lowres = create_dataloader(dataset_lowres)
    loader_highres = create_dataloader(dataset_highres)

    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sparse:
        net_cw = replace_depthwise_with_channelwise(
            MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
        )
        net_dw = MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
        net_cw.eval()
        net_dw.eval()

        results = {}
        #results["low_res_channelwise_cpu"] = run_tests(
        #    net_cw, loader_lowres, device_cpu, "Low-res channelwise [CPU]", True)
        results["low_res_channelwise_gpu"] = run_tests(
            net_cw, loader_lowres, device_gpu, "Low-res channelwise [GPU]", True, use_gpu=True)
        results["low_res_depthwise_gpu"] = run_tests(
            net_dw, loader_lowres, device_gpu, "Low-res depthwise [GPU]", True, use_gpu=True)
        save_results(results, "performance/lowres_sparse.pkl")

        results = {}
        results["high_res_channelwise_cpu"] = run_tests(
            net_cw, loader_highres, device_cpu, "High-res channelwise [CPU]", True)
        results["high_res_channelwise_gpu"] = run_tests(
            net_cw, loader_highres, device_gpu, "High-res channelwise [GPU]", True, use_gpu=True)
        results["high_res_depthwise_gpu"] = run_tests(
            net_dw, loader_highres, device_gpu, "High-res depthwise [GPU]", True, use_gpu=True)
        save_results(results, "performance/highres_sparse.pkl")

    else:
        net_dense = DenseUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
        net_dense.eval()

        results = {}
        results["low_res_dense_cpu"] = run_tests(
            net_dense, loader_lowres, device_cpu, "Low-res dense [CPU]", False)
        results["low_res_dense_gpu"] = run_tests(
            net_dense, loader_lowres, device_gpu, "Low-res dense [GPU]", False, use_gpu=True)
        save_results(results, "performance/lowres_dense.pkl")

        results = {}
        results["high_res_dense_cpu"] = run_tests(
            net_dense, loader_highres, device_cpu, "High-res dense [CPU]", False)
        results["high_res_dense_gpu"] = run_tests(
            net_dense, loader_highres, device_gpu, "High-res dense [GPU]", False, use_gpu=True)
        save_results(results, "performance/highres_dense.pkl")


if __name__ == "__main__":
    main()
