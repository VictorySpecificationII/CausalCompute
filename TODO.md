# CausalCompute — Production Readiness TODO

Items are grouped by category and roughly ordered by impact within each group.
Status: [ ] not started  [~] partial  [x] done

---

## 1. Physics & Model Fidelity

The core physics is validated but covers a narrow slice of real training configurations.

- [ ] **ZeRO stage modeling** — currently assumes full model replica per DP rank. ZeRO-1/2/3 shards optimizer state, gradients, and weights across DP; changes both memory and comm volume significantly.
- [ ] **Pipeline bubble overhead** — PP > 1 incurs a bubble fraction of `(pp-1)/(pp * micro_steps)`. Currently not deducted from useful compute time; Step 1 timing is optimistic for PP > 1.
- [ ] **TP communication modeled explicitly** — intra-node TP comm is flagged via NVLink check but not included in `t_step_s`. For PCIe nodes or large TP degrees, this is a real bottleneck.
- [ ] **Compute/comm overlap** — `comm_exposed_fraction` exists as a knob but defaults to 1.0 (no overlap). Real frameworks (Megatron-LM, FSDP) overlap backward pass with gradient allreduce. Model the overlap properly.
- [ ] **Activation checkpointing** — recomputing activations trades compute for memory. Affects `B_step_bytes` (step working set) and `F_step_flop` (extra forward pass). Currently user must set these manually.
- [ ] **Sequence parallelism (SP)** — extends TP to sequence dimension; common in long-context training. Affects memory and comm model.
- [ ] **Expert parallelism (MoE)** — mixture-of-experts models have a different FLOPs/token ratio and require all-to-all expert routing comm. The `c=6` FLOPs/param assumption does not apply.
- [ ] **FP8 / quantized training** — FP8 forward with BF16 optimizer changes `b_w`, `b_g`, and effective compute throughput. Should be expressible through existing dataclasses but needs validation.
- [ ] **Gradient compression** — TopK sparsification, PowerSGD, etc. reduce `b_update_per_param`. Currently modeled as a scalar knob but not validated.

---

## 2. Validation

Currently validated on a single hardware configuration (Nebius H100 SXM, InfiniBand, DP=8 and DP=16).

- [ ] **Broader hardware coverage** — validate against A100 SXM, H200, MI300X, TPU v4/v5. Each has different `F_dev_sust_flop_s`, `B_dev_mem_bytes`, NVLink topology.
- [ ] **Multiple cluster topologies** — validate network step against real leaf/spine deployments (different radix, oversubscription ratios).
- [ ] **Storage throughput validation** — compare Step 4 drive counts against real storage cluster benchmarks (Lustre, GPFS, WekaFS).
- [ ] **Regression test dataset** — current regression suite has 2 data points (DP=8, DP=16 on one cluster). Need ≥10 data points across different models and hardware.
- [ ] **Cross-validate against published training reports** — GPT-3, Llama, Mistral, Falcon training reports contain cluster sizes and throughput numbers that can be back-tested.

---

## 3. Cost Model

Without cost, you can answer "will it fit?" but not "is it worth building?"

- [x] **CapEx model** — unit cost inputs per component type: GPU, compute node (chassis + CPU + DRAM), NIC, leaf switch, spine switch, cable, storage node, drive, rack. Deliberately vendor-agnostic (price per TFLOP/s, price per TB, etc.).
- [x] **OpEx model** — electricity cost per kWh × facility power × run duration = energy cost. Already have `E_facility_kWh` from Step 2; just need price input.
- [x] **Cost per token** — total CapEx amortized over cluster lifetime + OpEx over run, divided by `Tok`. The number that matters for LLM economics. Two views: amortised and full-capex.
- [ ] **TCO / ROI summary** — total cost of ownership for the training run. Useful for build-vs-buy and on-prem vs. cloud comparisons.
- [ ] **Cloud spot pricing comparison** — optional: given GPU-hours required and a spot price input, compute cloud equivalent cost.

---

## 4. Sensitivity Analysis

Users need to know which constraints are binding, not just whether a design is feasible.

- [x] **Bottleneck identification** — after Step 1, explicitly report which bound is active (compute, memory, fabric, or step time). "You are memory-bound; adding GPUs won't help until you shard more aggressively."
- [ ] **Parameter sweep** — given a brief, sweep one parameter (e.g. GPU count, TP degree, oversubscription) and return a table of results. Expose in CLI as `--sweep param=start:stop:step`.
- [ ] **What-if deltas** — "if I upgrade NICs from 200 Gb/s to 400 Gb/s, what changes?" Requires running the pipeline twice and diffing the output.
- [ ] **Binding constraint visualization** — Streamlit: radar/spider chart showing how close each constraint is to its limit.

---

## 5. Failure & Recovery

A 30-day run on 1000 GPUs is not a single uninterrupted job.

- [ ] **Checkpoint recovery time** — Step 4 computes checkpoint write time but not read-back time. Recovery = read `S_ckpt` bytes at storage read BW + reload to GPU HBM.
- [ ] **MTBF-based failure model** — GPU MTBF ~100k hours. For N GPUs over T seconds, expected failures = `N * T / MTBF`. Combine with checkpoint interval to estimate expected wasted compute.
- [ ] **Effective training time** — `T_effective = T_run + T_ckpt_total + T_restart_expected`. Currently only `T_ckpt_total` is computed.
- [ ] **Availability / uptime target** — given a target uptime (e.g. 95%), derive required checkpoint frequency and storage redundancy.

---

## 6. Network

- [ ] **3-tier fat-tree sizing** — Step 3 flags when 2-tier is insufficient but does not size the 3-tier alternative. Add `run_network_3tier()` or extend `run_network()` to handle it.
- [ ] **RoCE vs. InfiniBand differences** — RoCE has retransmission overhead under congestion; IB has credit-based flow control. Model as a derating factor on effective BW (e.g. `roce_efficiency: float = 0.85`).
- [ ] **Collective algorithm selection** — ring allreduce is optimal for large messages and balanced topologies. For small messages or rail-optimised topologies, recursive halving or tree reduction may be faster. The comm model should flag when ring is suboptimal.
- [ ] **In-network compute** — SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) offloads allreduce to the switch. Reduces fabric traffic by up to 2×. Model as an optional flag.

---

## 7. Storage

- [ ] **Dataset preprocessing throughput** — tokenization is CPU-bound. For large datasets, the preprocessing pipeline (reading raw text → tokenized tensors) can be the real bottleneck, not the training loop. Model tokenization throughput vs. training token rate.
- [ ] **Streaming vs. cached access** — Step 4 assumes sequential read BW. Object stores (S3, GCS) have different latency/throughput characteristics vs. parallel filesystems (Lustre, GPFS, WekaFS). Add a storage backend type input.
- [ ] **Checkpointing to object store** — cloud training often checkpoints to S3/GCS, not a local storage cluster. Different BW characteristics and no drive count.

---

## 8. Software Engineering

- [ ] **JSON / CSV output** — `--output json` flag in CLI. Machine-readable output for integration with other tools (Terraform, Ansible, procurement systems).
- [ ] **Brief schema versioning** — as fields are added/removed, old briefs should still load with sensible defaults and a deprecation warning, not a crash.
- [ ] **Brief validation** — beyond type checking: catch physically impossible inputs (e.g. `eta_compute > 1.0`, `oversubscription < 1.0`, `T = 0`). Most are already caught; do a full audit.
- [ ] **Structured logging** — replace `print()` in CLI with proper logging. Useful for debugging and for capturing run traces.
- [ ] **Public Python API** — `from causalcompute import size_cluster` as a clean single-call interface. Useful for embedding in notebooks or other tools.
- [ ] **Example briefs** — add pre-built briefs for common configurations: 7B, 13B, 70B, 405B models; H100 / A100 / MI300X hardware; IB vs. RoCE fabric.

---

## 9. Packaging & Distribution

- [ ] **Publish to PyPI** — `pip install causalcompute`. Currently only installable from source.
- [ ] **Docker image** — self-contained image with CLI and Streamlit UI. Removes Python environment setup friction.
- [ ] **CI/CD pipeline** — run tests on every push (GitHub Actions). Currently tests are only run manually.
- [ ] **Versioned releases** — semantic versioning, changelog, GitHub releases. Required for any production use.
- [ ] **pyproject.toml audit** — pin dependency versions, add `python_requires`, ensure `extras_require` is complete.

---

## 10. UI / UX

- [ ] **Side-by-side comparison** — compare two briefs (e.g. H100 vs. A100, IB vs. RoCE) in a single Streamlit view. The most-requested feature for any sizing tool.
- [ ] **Export to PDF / CSV** — download the full report as a formatted PDF or the BoM as a CSV for procurement.
- [ ] **Saved sessions / shareable URLs** — Streamlit's session state is ephemeral. Add URL query-param encoding of key parameters so a configuration can be shared via link.
- [ ] **Binding constraint dashboard** — visual indicator of which step is the active bottleneck (compute / memory / fabric / storage / network).
- [ ] **Replace Streamlit with a proper frontend** — Streamlit is fine for prototyping but has limitations (no custom layout, no multi-page state, no auth). For a production tool, consider FastAPI backend + React frontend, or at minimum a multi-page Streamlit app.

---

## 11. Documentation

- [ ] **Physics assumptions document** — write down every formula, every constant, every simplification. "c=6 FLOPs/param-token assumes dense transformer with no activation checkpointing" etc. Required for anyone to trust the tool.
- [ ] **User guide** — how to fill in a brief, what each field means, common pitfalls.
- [ ] **API reference** — auto-generated from docstrings (Sphinx or mkdocs).
- [ ] **Validation report** — publish the Nebius regression results and methodology. This is what makes the tool credible to someone new.
- [ ] **Architecture decision records (ADRs)** — document why the causal pipeline pattern, why vendor-agnostic, why ring allreduce as the default comm model.

---

## Priority order for a v1.0

If this had to ship as a real product, the order would be:

1. ~~Cost model (CapEx + cost/token)~~ ✓ — `core/cost.py`, two accounting views, 27 tests
2. ~~Bottleneck identification~~ ✓ — `core/analysis.py`, binding constraint + headroom + recommendations, 30 tests
3. ZeRO stage modeling — most production training uses ZeRO; without it the memory model is wrong for large models
4. JSON output + PyPI publish — distribution and integration
5. CI/CD + versioned releases — minimum viable open-source project hygiene
6. Cross-validation against published training reports — credibility
7. Side-by-side comparison in UI — highest-impact UX feature
8. 3-tier fat-tree sizing — the 2-tier feasibility warning with no fix is a dead end
9. Checkpoint recovery + failure model — a 30-day run without failure modeling is incomplete
10. Documentation — physics assumptions doc is the single most important credibility signal
