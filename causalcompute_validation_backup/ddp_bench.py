import os, time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

class ToyNet(torch.nn.Module):
    def __init__(self, d=8192, layers=8):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.append(torch.nn.Linear(d, d, bias=False))
            blocks.append(torch.nn.GELU())
        self.net = torch.nn.Sequential(*blocks)
        self.out = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.out(self.net(x))

def main():
    init()
    torch.backends.cudnn.benchmark = True

    d = int(os.environ.get("D", "8192"))
    b = int(os.environ.get("B", "8"))
    layers = int(os.environ.get("LAYERS", "8"))
    steps = int(os.environ.get("STEPS", "250"))
    warmup = int(os.environ.get("WARMUP", "50"))

    model = ToyNet(d=d, layers=layers).cuda()
    ddp = DDP(model, broadcast_buffers=False)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    x = torch.randn(b, d, device="cuda", dtype=torch.float16)
    scaler = torch.cuda.amp.GradScaler()

    times = []
    torch.cuda.synchronize()
    for i in range(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y = ddp(x).float()
            loss = (y * y).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        t1 = time.time()
        if i >= warmup:
            times.append(t1 - t0)

    if dist.get_rank() == 0:
        times.sort()
        med = times[len(times)//2]
        p95 = times[int(len(times)*0.95)]
        print(f"MED_STEP_S {med:.6f}  P95_STEP_S {p95:.6f}  (B={b}, D={d}, LAYERS={layers})")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
