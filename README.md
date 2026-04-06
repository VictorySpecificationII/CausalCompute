# CausalCompute

Vendor-neutral, first-principles AI training infrastructure sizing.

Supply a YAML brief describing your workload and hardware; get a full cluster design, power budget, network topology, storage pools, bill of materials, and cost model — all derived causally from physics, with no brand templates or magic numbers.

> Predictions have been validated against real H100 InfiniBand training runs (see [`v0/evidence/`](v0/evidence/)).

---

## How it works

Each step only consumes what the previous step proved. Physics flows one way.

| Step | Module | What it produces |
|------|--------|-----------------|
| 0 | `core/fundamentals.py` | Device lower bounds (compute / state / instant), step time budget, checkpoint cost |
| 1 | `core/design.py` | Smallest feasible cluster via DP/TP/PP search; closes time and memory constraints |
| — | `core/analysis.py` | Binding constraint identification, step-time headroom, comm fraction, recommendations |
| 2 | `core/thermals.py` | Per-node power ledger, facility power, airflow / coolant flow rates |
| 3 | `core/network.py` | Leaf/spine switch count, port budget, inter-node fabric BW verification |
| 4 | `core/storage.py` | Dataset and checkpoint pool sizing (BW-bound and capacity-bound) |
| 5 | `core/bom.py` | Bill of materials — GPU, NIC, switch, drive, rack counts; NVLink and GPUDirect checks |
| 6 | `core/cost.py` | CapEx, OpEx, cost per token (amortised and full-capex views) |

---

## Quickstart

```bash
pip install -e ".[dev]"
causalcompute briefs/13b.yaml
```

Narrative (Step 0 causal story only):

```bash
causalcompute briefs/13b.yaml --story
```

Full debug dump of all pipeline dicts:

```bash
causalcompute briefs/13b.yaml --debug
```

Interactive UI:

```bash
streamlit run app/streamlit_app.py
```

---

## Brief format

A brief is a YAML file that is the single source of truth for a sizing run. All quantities are in SI units (bytes, seconds, bytes/s, FLOP/s, watts).

```
briefs/
  template.yaml       — copy this to start a new scenario
  13b.yaml            — worked example: 13B params / 3T tokens / 30 days
  validate_dp8.yaml   — validation against 8-GPU DDP run
  validate_dp16.yaml  — validation against 16-GPU DDP run
```

Key sections:

| Section | Controls |
|---------|----------|
| `workload` | Parameters `P`, tokens `Tok`, deadline `T`, FLOPs/token `c` |
| `state_bytes` | Bytes/param for weights, gradients, optimizer state |
| `io` | Dataset stream BW, checkpoint bytes/param |
| `device` | Sustained FLOP/s, HBM capacity |
| `step` | Instantaneous working set, tokens/step, update signal size |
| `capabilities` | Fabric BW, storage BW, checkpoint cadence |
| `design` | GPU count (or `null` for auto), TP/PP limits, efficiency factors |
| `power_thermals` | Per-GPU/CPU/other power, PUE, cooling mode, rack limits |
| `network` | Switch radix, over-subscription ratio |
| `storage_design` | Drive capacity/BW, dataset replication, checkpoint retention |
| `compute_node` | NVLink BW, GPUDirect RDMA, CPU cores, DRAM, root complex |
| `cost` | GPU unit cost, NIC/switch/drive costs, electricity rate, amortisation period |

Use `null` for any field to get the default. Scientific notation is encouraged (`13e9`, `1e15`).

---

## Running tests

```bash
pytest
pytest --cov=causalcompute
```

Test files mirror the pipeline:

```
tests/
  test_fundamentals.py
  test_design.py
  test_thermals.py
  test_network.py
  test_storage.py
  test_bom.py
  test_bom_nodespec.py
  test_cost.py
  test_analysis.py
```

---

## Repository layout

```
causalcompute/        — installable package
  core/               — pipeline steps (one file per step)
  io/                 — YAML loader and CLI report printer
  app/                — Streamlit UI
briefs/               — example and template YAML briefs
tests/                — pytest suite (~250 tests)
v0/                   — original Steps 0-2 prototype and validation evidence
TODO.md               — production roadmap
pyproject.toml
```

---

## Units

Everything is SI at the boundary. Display formatting is handled by the report layer.

| Quantity | Unit |
|----------|------|
| Compute | FLOP/s |
| Memory / size | bytes |
| Bandwidth | bytes/s |
| Time | seconds |
| Power | watts |
| Cost | USD |
