<!-- step2_powerandthermals/README.md -->
# LSDT Power & Thermals — Energy Layer

This module answers the next reality check:

> **If Step 1 proposes a cluster, can the laws of energy support it?**

Step 0 described what the workload demands in bytes, FLOPs, and seconds.  
Step 1 described how we might arrange devices to satisfy those demands.

Step 2 describes what the building must provide:

- electrical power
- heat rejection
- flow (air or liquid)

This layer is not “vendor sizing.”  
It is conservation laws.

---

## What This Layer Computes

### 1) Power ledger
Given a cluster shape (from Step 1):

- GPU power
- CPU power (per node)
- “other” node power (NICs, DRAM, storage, fans, margin)
- total IT power
- facility power via PUE

### 2) Heat ledger
Nearly all IT electrical power becomes heat:

Q̇ [W] ≈ P_IT [W]


### 3) Cooling flow requirement

#### Air
Uses:

Q̇ = ṁ * cp * ΔT
ṁ = ρ * V̇



Outputs airflow as:
- m³/s
- CFM

#### Liquid
Same equation, different fluid properties.

Outputs coolant flow as:
- kg/s
- L/min

### 4) Optional rack sanity
If you provide rack count (or nodes/rack), it reports:
- IT power per rack
- rack power margin vs a rack limit (if provided)

### 5) Optional energy over run
If you provide runtime seconds (`T_run_s`), it computes:
- IT energy (kWh)
- facility energy (kWh)

---

## Inputs

This layer consumes **Step 1** only, via the stable handoff:

- `design_bundle["handoff"]["cluster"]`

You provide a small set of overrideable assumptions:

- `P_gpu_W`
- `P_cpu_W_per_node`
- `P_other_W_per_node`
- `PUE`
- cooling mode (`air` or `liquid`)
- allowable temperature rise (`ΔT`)

Everything is in SI units.

---

## Output Contract

`run_powerandthermals(design_bundle, ...) → dict`

Sections include:

- `cluster` – echoed cluster shape
- `power` – power inputs and ledger (W and kW)
- `heat` – heat load (W and kW)
- `thermals` – airflow or coolant flow results
- `rack` – optional rack sanity output
- `energy` – optional energy over run (kWh)

---

## Known Simplifications

Deliberate abstractions:

- Power is modeled as a **simple ledger**, not per-rail or transient behavior.
- Heat rejection assumes **steady-state** (no thermal mass, no transients).
- Airflow/coolant flow is **minimum required** for a chosen ΔT; real deployments include margin.

This is physics, not procurement.

