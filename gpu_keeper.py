#!/usr/bin/env python3
"""
GPU Keeper — 空闲时占用 GPU 防止被调度系统回收

功能:
  1. 在每张 GPU 上分配指定比例的显存（默认 60%）
  2. 周期性执行矩阵运算保持 GPU 利用率（默认 ~15-25%）
  3. 自动检测是否有"真正的任务"在跑，如果有则自动让出
  4. 支持信号优雅退出（Ctrl+C / kill）
  5. 支持只占用指定 GPU

用法:
  python gpu_keeper.py                        # 占用所有 GPU
  python gpu_keeper.py --gpus 0,1,2,3         # 只占用指定 GPU
  python gpu_keeper.py --mem-fraction 0.5      # 占用 50% 显存
  python gpu_keeper.py --util-target 20        # 目标利用率 20%
  python gpu_keeper.py --yield-threshold 5000  # 当其他进程占用超过 5GB 时让出
  nohup python gpu_keeper.py &                 # 后台运行
"""

import argparse
import os
import signal
import sys
import time
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# 全局控制
# ---------------------------------------------------------------------------
_STOP = threading.Event()


def _handle_signal(signum, frame):
    print(f"\n[{_now()}] Received signal {signum}, shutting down gracefully...")
    _STOP.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 检查其他用户/进程是否在使用 GPU
# ---------------------------------------------------------------------------
def get_other_gpu_usage_mb(gpu_id: int, my_pid: int) -> float:
    """通过 nvidia-smi 查询该 GPU 上非本进程的显存占用 (MiB)"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=10
        )
        total = 0.0
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            pid = int(parts[0].strip())
            mem = float(parts[-1].strip())
            if pid != my_pid:
                total += mem
        return total
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 单卡占用 worker
# ---------------------------------------------------------------------------
def gpu_worker(
    gpu_id: int,
    mem_fraction: float,
    util_target: int,
    yield_threshold_mb: float,
    check_interval: float,
):
    """在一张 GPU 上持续占用显存和算力"""
    import torch

    device = torch.device(f"cuda:{gpu_id}")
    my_pid = os.getpid()

    # ---- 1. 分配显存 ----
    torch.cuda.set_device(device)
    total_mem = torch.cuda.get_device_properties(device).total_mem
    alloc_bytes = int(total_mem * mem_fraction)
    alloc_gb = alloc_bytes / (1024 ** 3)

    print(f"[{_now()}] GPU {gpu_id}: Allocating {alloc_gb:.1f} GB ({mem_fraction*100:.0f}% of {total_mem/(1024**3):.1f} GB)")

    # 用若干大 tensor 占满目标显存
    tensors = []
    chunk_size = 256 * 1024 * 1024 // 4  # 256 MB per chunk (float32)
    allocated = 0
    try:
        while allocated < alloc_bytes:
            remaining = alloc_bytes - allocated
            elems = min(chunk_size, remaining // 4)
            if elems <= 0:
                break
            t = torch.randn(elems, device=device, dtype=torch.float32)
            tensors.append(t)
            allocated += elems * 4
    except torch.cuda.OutOfMemoryError:
        print(f"[{_now()}] GPU {gpu_id}: OOM during allocation, using what we got ({allocated/(1024**3):.1f} GB)")

    print(f"[{_now()}] GPU {gpu_id}: Allocated {allocated/(1024**3):.1f} GB in {len(tensors)} chunks")

    # ---- 2. 准备计算用的矩阵 ----
    # 通过调整矩阵大小和 sleep 控制利用率
    mat_size = 2048
    mat_a = torch.randn(mat_size, mat_size, device=device, dtype=torch.float16)
    mat_b = torch.randn(mat_size, mat_size, device=device, dtype=torch.float16)

    # 根据目标利用率调整 duty cycle
    # compute_time / (compute_time + sleep_time) ≈ util_target / 100
    compute_ms = 50   # 每轮计算约 50ms
    if util_target > 0:
        sleep_ms = compute_ms * (100 - util_target) / util_target
    else:
        sleep_ms = 1000

    paused = False
    last_check = 0.0
    cycle_count = 0

    print(f"[{_now()}] GPU {gpu_id}: Running (target util ~{util_target}%, "
          f"yield if others use >{yield_threshold_mb/1024:.1f} GB)")

    # ---- 3. 主循环 ----
    while not _STOP.is_set():
        now = time.monotonic()

        # 定期检查是否需要让出
        if now - last_check > check_interval:
            last_check = now
            other_usage = get_other_gpu_usage_mb(gpu_id, my_pid)
            if other_usage > yield_threshold_mb:
                if not paused:
                    print(f"[{_now()}] GPU {gpu_id}: Other processes using {other_usage:.0f} MiB, YIELDING compute...")
                    paused = True
            else:
                if paused:
                    print(f"[{_now()}] GPU {gpu_id}: Other processes gone ({other_usage:.0f} MiB), RESUMING...")
                    paused = False

        if paused:
            # 让出算力但保留显存占用
            _STOP.wait(2.0)
            continue

        # 做一小轮矩阵乘法
        try:
            for _ in range(5):
                _ = torch.mm(mat_a, mat_b)
            torch.cuda.synchronize(device)
        except Exception as e:
            print(f"[{_now()}] GPU {gpu_id}: Compute error: {e}")
            _STOP.wait(5.0)
            continue

        cycle_count += 1
        if cycle_count % 1000 == 0:
            print(f"[{_now()}] GPU {gpu_id}: heartbeat — {cycle_count} cycles done")

        # sleep 控制利用率
        _STOP.wait(sleep_ms / 1000.0)

    # ---- 4. 清理 ----
    print(f"[{_now()}] GPU {gpu_id}: Cleaning up...")
    del mat_a, mat_b
    for t in tensors:
        del t
    tensors.clear()
    torch.cuda.empty_cache()
    print(f"[{_now()}] GPU {gpu_id}: Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU Keeper — 保持 GPU 占用防止被回收",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="要占用的 GPU 编号，逗号分隔 (默认: 全部)"
    )
    parser.add_argument(
        "--mem-fraction", type=float, default=0.60,
        help="每张 GPU 占用的显存比例 (默认: 0.60 = 60%%)"
    )
    parser.add_argument(
        "--util-target", type=int, default=20,
        help="目标 GPU 利用率百分比 (默认: 20)"
    )
    parser.add_argument(
        "--yield-threshold", type=float, default=5000,
        help="其他进程显存占用超过此值(MiB)时让出算力 (默认: 5000)"
    )
    parser.add_argument(
        "--check-interval", type=float, default=30.0,
        help="检查其他进程的间隔秒数 (默认: 30)"
    )

    args = parser.parse_args()

    # 确定要占用的 GPU
    import torch
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        print("No GPUs found!")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  GPU Keeper")
    print(f"  GPUs:            {gpu_ids}")
    print(f"  Memory fraction: {args.mem_fraction*100:.0f}%")
    print(f"  Util target:     ~{args.util_target}%")
    print(f"  Yield threshold: {args.yield_threshold:.0f} MiB")
    print(f"  Check interval:  {args.check_interval}s")
    print(f"  PID:             {os.getpid()}")
    print(f"  Stop:            Ctrl+C or kill {os.getpid()}")
    print(f"{'='*60}")

    # 每张 GPU 一个线程
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, args.mem_fraction, args.util_target,
                  args.yield_threshold, args.check_interval),
            name=f"gpu-worker-{gpu_id}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)  # 错开启动避免同时 alloc

    # 等待退出
    try:
        while not _STOP.is_set():
            _STOP.wait(1.0)
    except KeyboardInterrupt:
        _STOP.set()

    print(f"\n[{_now()}] Waiting for workers to finish...")
    for t in threads:
        t.join(timeout=10)

    print(f"[{_now()}] All done. Bye!")


if __name__ == "__main__":
    main()
