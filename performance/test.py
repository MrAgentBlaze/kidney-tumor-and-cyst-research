import os
import sys
import time
import gc
import psutil
import resource
import pickle as pk
import numpy as np
import torch
import tqdm
from itertools import islice
from torch.utils.data import DataLoader
from dataset import SparseDataset
from utils import (
    ini_argparse,
    get_k_fold_data_loaders,
    collate_sparse_minkowski,
    configure_matplotlib,
    replace_depthwise_with_channelwise,
    arrange_sparse_minkowski,
    arrange_truth
)
from model import MinkUNetConvNeXtV2, DenseUNetConvNeXtV2

def setup_dataset(args, stage2, min_hu, max_hu, dataset_name):
    args.stage2 = stage2
    args.min_hu = min_hu
    args.max_hu = max_hu
    args.dataset_name = dataset_name
    return SparseDataset(args)

def create_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_sparse_minkowski,
        shuffle=False,
    )

def _arrange_batch(batch, device, sparse):
    batch_input = arrange_sparse_minkowski(batch, device)
    batch_target = arrange_truth(batch, device)
    if not sparse:
        batch_input = batch_input.dense()[0]
        batch_target = batch_target.dense()[0]
    return batch_input, batch_target

def measure_inference(model, loader, device, sparse, warmup_iters=5, disable=False, use_gpu=False):
    model.to(device)
    process = psutil.Process()
    memory_stats = {"usage": [], "peak": []} if not use_gpu else {"active": [], "reserved": [], "peak": []}
    
    print("Warmup...")
    with torch.no_grad():
        for batch in tqdm.tqdm(islice(loader, warmup_iters), total=warmup_iters, disable=disable):
            batch_input, batch_target = _arrange_batch(batch, device, sparse)
            gc.collect()  # Ensure old objects are cleared
            if use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            _ = model(batch_input, batch_target)
            torch.cuda.synchronize() if use_gpu else None
    print("Done!")
    
    print("Main test...")
    idx, times = [], []
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), disable=disable):
            batch_input, batch_target = _arrange_batch(batch, device, sparse)
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            
            start_time = time.perf_counter()
            _ = model(batch_input, batch_target)
            torch.cuda.synchronize() if use_gpu else None
            end_time = time.perf_counter()
            
            idx.append(batch['idx'][0])
            times.append(end_time - start_time)
    print("Done!")
    
    return np.array(idx), np.array(times), memory_stats

def run_tests(net, loader, device, label, sparse, use_gpu=False):
    idx, times, memory_usages = measure_inference(net, loader, device, sparse, use_gpu=use_gpu)
    time_mean, time_std = times.mean(), times.std()
    print(f"{label} Inference Time: {time_mean:.6f} ± {time_std:.6f} seconds")
    return {"idx": idx, "times": times, "memory_usages": memory_usages}

def save_results(results, filename):
    with open(filename, "wb") as fd:
        pk.dump(results, fd)

def main():
    parser = ini_argparse()
    args = parser.parse_args()
    nb_gpus = len(args.gpus)
    gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])

    # Manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    dataset_lowres = setup_dataset(args, False, -53.4, 283.2, "kits23_large_processed")
    dataset_highres = setup_dataset(args, True, -300, 500, "kits23_processed_highres11_all")
    
    loader_lowres = create_dataloader(dataset_lowres)
    loader_highres = create_dataloader(dataset_highres)
    
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    final_results = {}
    
    if args.sparse:
        net_channelwise = replace_depthwise_with_channelwise(MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args))
        net_depthwise = MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
        net_channelwise.eval()
        net_depthwise.eval()
        
        final_results["low_res_channelwise_cpu"] = run_tests(net_channelwise, loader_lowres, device_cpu, "Low-res: Channelwise [CPU]", args.sparse)
        final_results["low_res_channelwise_gpu"] = run_tests(net_channelwise, loader_lowres, device_gpu, "Low-res: Channelwise [GPU]", args.sparse, use_gpu=True)
        final_results["low_res_depthwise_gpu"] = run_tests(net_depthwise, loader_lowres, device_gpu, "Low-res: Depthwise [GPU]", args.sparse, use_gpu=True)
        
        save_results(final_results, "performance/lowres_sparse.pkl")
        
        final_results = {}
        final_results["high_res_channelwise_cpu"] = run_tests(net_channelwise, loader_highres, device_cpu, "High-res: Channelwise [CPU]", args.sparse)
        final_results["high_res_channelwise_gpu"] = run_tests(net_channelwise, loader_highres, device_gpu, "High-res: Channelwise [GPU]", args.sparse, use_gpu=True)
        final_results["high_res_depthwise_gpu"] = run_tests(net_depthwise, loader_highres, device_gpu, "High-res: Depthwise [GPU]", args.sparse, use_gpu=True)
        
        save_results(final_results, "performance/highres_sparse.pkl")
    else:
        net_dense = DenseUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
        net_dense.eval()
        
        final_results["low_res_dense_cpu"] = run_tests(net_dense, loader_lowres, device_cpu, "Low-res: Dense [CPU]", args.sparse)
        final_results["low_res_dense_gpu"] = run_tests(net_dense, loader_lowres, device_gpu, "Low-res: Dense [GPU]", args.sparse, use_gpu=True)
        save_results(final_results, "performance/lowres_dense.pkl")
        
        final_results = {}
        final_results["high_res_dense_cpu"] = run_tests(net_dense, loader_highres, device_cpu, "High-res: Dense [CPU]", args.sparse)
        final_results["high_res_dense_gpu"] = run_tests(net_dense, loader_highres, device_gpu, "High-res: Dense [GPU]", args.sparse, use_gpu=True)
        save_results(final_results, "performance/highres_dense.pkl")

if __name__ == "__main__":
    main()
