import os, time
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    warmup = int(os.environ.get("WARMUP", "20"))
    iters  = int(os.environ.get("ITERS", "50"))
    msg_bytes = int(os.environ.get("MSG_BYTES", str(256 * 1024 * 1024)))

    n = msg_bytes // 2  # fp16
    x = torch.empty(n, device="cuda", dtype=torch.float16)
    x.fill_(1.0)

    for _ in range(warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    t1 = time.time()

    dt = (t1 - t0) / iters
    world = dist.get_world_size()
    factor = 2.0 * (world - 1) / world if world > 1 else 0.0
    algbw_Bps = (factor * msg_bytes) / dt
    algbw_GBps = algbw_Bps / 1e9

    if dist.get_rank() == 0:
        print(f"WORLD={world} MSG_MiB={msg_bytes/1024/1024:.0f} dt_s={dt:.6f} algbw_GBps={algbw_GBps:.2f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
