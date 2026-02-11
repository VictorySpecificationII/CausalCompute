# LSDT Design — Architecture Layer

Step 0 asked:

> **What must be true in the universe in order to train this model at all?**

Step 1 asks the next question:

> **Given those truths, how could we arrange real machines so they are respected?**

This layer does not create new physics.  
It interprets the physics through **architecture choices**.

---

## What This Layer Introduces

Step 1 adds only what Step 0 intentionally avoided:

- Parallelism choices — **DP / TP / PP**
- Device count and node shape
- Sustained efficiency penalties
- A search for feasible decompositions

It transforms:

physics requirements → realizable cluster design


without altering any quantity derived in Step 0.

---

## What This Layer Must Respect

Step 1 is bound by the Step-0 contract.

It **must not**:

- Recompute FLOPs, bytes, or time budgets  
- Invent new resource requirements  
- Assume any vendor topology

It may only:

- Partition existing quantities  
- Apply efficiency factors  
- Test closure against device limits

---

## Terminology (Aligned with Step 0)

| Step-0 Concept | Meaning in Step 1 |
|----------------|-------------------|
| **Instantaneous working set** (`B_step_bytes`) | Memory that must be resident during one update step |
| **Model state** (`B_state_min_bytes`) | Weights + gradients + optimizer state |
| **Per-step movement** (`B_exchange_per_step_bytes`) | Bytes that must be communicated each step |
| **Step time budget** (`t_step_max_s`) | Maximum allowed wall-clock per step |
| **Device capability** (`meta`) | Sustained FLOP/s, memory, fabric BW |

No new physical quantities appear here.

---

## Design Model

### Memory interpretation


bytes_per_device = B_state_min_bytes / DP
 - B_step_bytes / (TP × PP)


- **State** shards with data parallelism  
- **Instantaneous working set** shards with model partition

### Timing interpretation


t_step = t_compute + t_comm

t_compute = F_step_flop / (G × F_dev × η_compute)
t_comm = t_comm_min_s / η_fabric



Step 0 supplied every term on the right;  
Step 1 only applies **real-world penalties**.

---

## Two Legitimate Questions

### Mode A — Auto-size

> *How many devices must this cluster have at minimum?*

- Start from Step-0 lower bound  
- Increase `G` until a design closes  
- Return the smallest feasible cluster

### Mode B — Fixed size

> *Given a cluster I already own, can it meet the goal?*

- Keep `G` fixed  
- Search only `(dp,tp,pp)` factorizations  
- Report feasibility and margins

---

## Constraints

Any candidate must satisfy:

- `dp × tp × pp = G`  
- per-device memory ≤ device capacity  
- `t_compute + t_comm ≤ t_step_max`

Preference:

- **Auto-size:** minimize `G`, then prefer high DP  
- **Fixed:** maximize headroom, then prefer high DP

---

## Output Contract

`run_design(bundle0, G=None) → dict`

- `feasible` – whether any design closed  
- `solution` – cluster and parallelism (if feasible)  
- `diagnostics` – physics echo and search trace  
- `no_solution_reason` – structured explanation (if not)

---

## What This Layer Does Not Yet Model

Deliberate omissions:

- Collective algorithms (ring/tree/hierarchical)  
- DP vs TP communication asymmetry  
- NVLink vs fabric tiers  
- Overlap and pipeline bubbles  
- Activation checkpointing policies

Those belong to later refinement, not to first closure.

---

## CLI Examples

Auto-size:

```bash
python examples/13b.py
```

Fixed cluster:

```bash
python examples/13b.py --G 128
```

Tune efficiencies:

```bash
python examples/13b.py --eta-compute 0.30 --eta-fabric 0.70
```

Step 0 described necessity.
Step 1 explores possibility.
Everything beyond this becomes engineering.