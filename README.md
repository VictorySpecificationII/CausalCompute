# CausalCompute — First-Principles Sizing Engine (Steps 0–2)

> **Infrastructure decisions are derivations from physics and workload, not brand templates.**

This repository implements a vendor-neutral method to translate **AI training intent** into **physical requirements** using only SI units and explicit assumptions.

The engine answers:

**“What must exist in the real world to finish this training run on time?”**

without assuming:
- a topology  
- a vendor  
- a rack layout  
- a product line  

---

## What the engine does

### Step 0 — Fundamentals (physics only)

From a brief:

- model size  
- tokens  
- deadline  
- algorithmic step size  

we derive **absolute invariants**:

- sustained FLOP/s required  
- minimum model-state bytes  
- dataset & checkpoint bandwidth  
- update payload per step  
- maximum allowed step time  

👉 No cluster design is assumed here — only reality.

---

### Step 1 — Design closure

Introduce architecture choices:

- DP / TP / PP factorization  
- real efficiencies (η_compute, η_fabric)  
- communication model

and solve:

- memory per device  
- step time = compute + comm  
- smallest feasible GPU count (or test a fixed G)

👉 This is the **causal bridge** from physics → architecture.

---

### Step 2 — Power & Thermals

From the feasible design we compute:

- IT and facility power  
- heat production  
- airflow or coolant flow  
- optional rack sanity  
- energy over the run

👉 This is the handoff toward mechanical & electrical design.

---

## Quick start

Install dependency:

```bash
pip install pyyaml
````

Run a brief:

```bash
python run_012.py briefs/13b.yaml
```

Narrative Step-0 explanation:

```bash
python run_012.py briefs/13b.yaml --story
```

Full debug bundles:

```bash
python run_012.py briefs/13b.yaml --debug
```

---

## Repository structure

```
first-principles/
├── briefs/
│   ├── template.yaml     ← how to describe a workload
│   ├── 13b.yaml          ← example scenario
│   └── README.md
│
├── run_012.py            ← engine entrypoint (Steps 0→1→2)
│
├── step0_fundamentals/   ← physics, no topology
├── step1_design/         ← DP/TP/PP closure
└── step2_powerandthermals/
```

A **brief** is the contract between ML intent and infrastructure sizing.

---

## Design philosophy

### 1) Causality flows one way

```
Workload → FLOPs → Time → Memory → Communication
         → Power → Heat → Flow → Space
```

Nothing is guessed from brands.

---

### 2) Everything in SI

* bytes, seconds, bytes/s
* FLOP, FLOP/s
* watts, kg/s, m³/s

Non-SI (CFM, LPM) are display only.

---

### 3) Explicit assumptions

Efficiencies and policies are parameters:

* η_compute — sustained vs peak math
* η_fabric — real collectives vs line rate
* Tok_per_step — algorithmic choice
* ΔT — mechanical design envelope

Change the brief → the physical answer changes.

---

### 4) Topology is a decision, not an input

Step 0 does **not** assume:

* nodes
* racks
* networks

Step 1 introduces them only when required to close time and memory.

---

## What this is (and is not)

**This is:**

* a reference sizing engine
* a digital-twin seed
* a communication bridge between ML, EE, and ME

**This is not:**

* a vendor selector
* a BOM generator
* a CFD tool
* a scheduler

Those come later — after physics is satisfied.

---

## Example questions it can answer

* “How many GPUs must exist at minimum?”
* “Is this deadline even possible?”
* “What coolant flow is implied by the workload?”
* “How much power does the facility need to commit?”
* “What bandwidth must the fabric expose before topology?”

---

## Extending beyond Step 2

Future layers can consume the Step-2 handoff:

* Step 3 — Networking topology
* Step 4 — Storage design
* Step 5 — Facilities zoning & transients
* Digital twin supervision

---

## Author

Built from the viewpoint of someone who has lived in:

* HPC & distributed systems
* control systems & thermodynamics
* motorsport engineering

Treating datacenters as **thermodynamic machines**, not SKU catalogs.
