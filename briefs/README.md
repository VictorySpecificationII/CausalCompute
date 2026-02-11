# Briefs (YAML)

A **brief** is the single source of truth for a sizing run.

`run_012.py` consumes a brief and runs:

- Step 0 — Fundamentals (physics-only)
- Step 1 — Design (DP/TP/PP + efficiencies; closes time + memory)
- Step 2 — Power & Thermals (power ledger + airflow/coolant flow)

Everything is **vendor-agnostic** and expressed in **SI units**.

---

## Files

- `template.yaml` — copy this to start a new scenario
- `13b.yaml` — an example scenario (13B / 3T tokens / 30 days)

---

## Conventions

### Units (SI)

| Quantity | Units |
|---|---|
| FLOPs, FLOP/s | `FLOP`, `FLOP/s` |
| Tokens | `tokens` |
| Time | `seconds` |
| Bandwidth | `bytes/s` |
| Memory / size | `bytes` |
| Power | `watts` |

Non-SI display units (like CFM or L/min) are optional and should stay off by default.

### Use `null` for “auto”

- `design.G: null` means Step 1 **auto-sizes** the smallest feasible GPU count.
- `power_thermals.rack.*: null` means rack sanity checks are **disabled**.

### Scientific notation is encouraged

Examples:

- `13e9` for parameters
- `3e12` for tokens
- `1e15` for sustained FLOP/s per device

---

## How to run

From `first-principles/`:

```bash
pip install pyyaml
python run_012.py briefs/13b.yaml
```

Narrative Step 0 story only:

```bash
python run_012.py briefs/13b.yaml --story
```

Dump the raw dictionaries (Step 0/1/2):

```bash
python run_012.py briefs/13b.yaml --debug
```
