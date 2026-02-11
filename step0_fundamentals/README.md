# LSDT Fundamentals — Physics Layer

This module answers a single question:

> **What must be true in the universe in order to train a model, regardless of how we design the cluster?**

It contains only statements that follow from:
- model size  
- token count  
- deadline  
- numerical representation  
- sustained device and bandwidth capabilities  

No parallelism, no topology, no vendor assumptions.

---

## What This Layer Guarantees

The fundamentals layer is **complete** when it expresses:

1. **Total work required**
   - Total FLOPs in the run  
   - Sustained FLOP/s to meet the deadline  
   - Sustained token/s to meet the deadline  

2. **Information that must exist**
   - Weight bytes  
   - Gradient bytes  
   - Optimizer state bytes  
   - Minimum total model state  

3. **I/O obligations**
   - Dataset streaming bandwidth  
   - Checkpoint size  
   - Checkpoint bandwidth requirement  

4. **Device lower bounds (before any architecture)**
   - From compute capability  
   - From memory capacity  
   - From instantaneous step working set  

5. **Per-step physics**
   - Tokens per step → maximum allowed step time  
   - Bytes that must be exchanged per step  
   - Minimum bandwidth implied by that exchange  

6. **Time-scale ledger**
   - Shortest possible compute time per step  
   - Shortest possible communication time per step  
   - Residual step time left for all other activity  

7. **Long-term wall-clock drains**
   - Minimum checkpoint time from storage bandwidth  
   - Total time spent checkpointing over the run  
   - Fraction of wall clock lost to checkpoints  

---

## What This Layer Does **Not** Assume

Nothing here depends on:

- DP / TP / PP choices  
- all-reduce algorithms (ring, tree, etc.)  
- nodes, racks, GPUs per node  
- placement or sharding strategy  
- overlap schemes  
- filesystem type or vendor  

Every number is derived only from **bytes, FLOPs, and seconds**.

---

## Output Contract

`run_fundamentals(...) → dict`

The module produces a structured Python bundle designed to feed `design.py`.

Sections include:

- `req` – global compute and I/O requirements  
- `device_bounds` – architecture-free device minima  
- `instant` – instantaneous memory requirements  
- `movement` – per-step exchange requirements  
- `timecmp` – compute time scales  
- `comm` – communication time scales  
- `budget` – residual step-time ledger  
- `ckpt` – checkpoint wall-clock impact  

---

## CLI

Return only the bundle:

```bash
python fundamentals.py
```

Show narrative + bundle:

```
python fundamentals.py --debug
```


## Known Simplifications

This module is “physics-oriented” (bytes, FLOPs, seconds) but still uses deliberate abstractions:

- **Step working set is an input:** `B_step_bytes` is provided rather than derived from sequence length, activation checkpointing/recompute strategy, attention variant, etc.
- **Optimizer + update signal are summarized:** optimizer state and per-step update signal are modeled as `bytes/param` (e.g., Adam-like states, gradient representation) rather than a specific optimizer implementation.
- **Communication is topology-agnostic:** per-step movement is modeled as an update-sized signal and does not encode a particular collective algorithm (ring/tree) or overlap behavior.
