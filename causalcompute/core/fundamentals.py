"""
Step 0 — Fundamentals.

Pure physics: no cluster topology, no parallelism assumptions.
All inputs and outputs are in SI units.

The causal chain established here:
  Workload → FLOPs → token rate → step time budget
           → state bytes → device lower bounds
           → instantaneous memory → checkpoint overhead
"""
from __future__ import annotations

from math import ceil

from .types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
    Device,
    FabricCapability,
    IO,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    Workload,
)


def run_fundamentals(
    workload: Workload,
    state: StateBytes,
    io: IO,
    device: Device,
    step: StepWorkingSet,
    schedule: StepSchedule,
    update: AlgorithmStepFacts,
    fabric: FabricCapability,
    storage: StorageCapability,
    checkpoint: CheckpointPolicy,
) -> dict:
    """
    Derive all topology-agnostic invariants from the workload brief.

    Returns a bundle dict consumed by run_design() (Step 1).
    Keys are stable across versions; Step 1 must not recompute anything here.
    """
    w, st, dev = workload, state, device

    # ------------------------------------------------------------------
    # Compute requirements
    # ------------------------------------------------------------------
    F_total = w.c * w.P * w.Tok       # total FLOPs for the run [FLOP]
    F_req = F_total / w.T             # required sustained throughput [FLOP/s]
    R_tok = w.Tok / w.T               # required token rate [tokens/s]

    # ------------------------------------------------------------------
    # Model-state memory (single logical copy, lower bounds)
    # ------------------------------------------------------------------
    B_weights = st.b_w * w.P
    B_grads = st.b_g * w.P
    B_opt = st.b_opt * w.P
    B_state = B_weights + B_grads + B_opt

    # ------------------------------------------------------------------
    # I/O contracts
    # ------------------------------------------------------------------
    BW_dataset = R_tok * io.b_tok * io.A_io   # dataset read BW [bytes/s]
    B_dataset_total = w.Tok * io.b_tok        # total dataset size [bytes]
    S_ckpt = w.P * io.b_ckpt                  # checkpoint size [bytes]
    BW_ckpt_req = S_ckpt / io.t_ckpt_max      # min checkpoint BW [bytes/s]

    # ------------------------------------------------------------------
    # Device lower bounds (no placement, no parallelism)
    # ------------------------------------------------------------------
    N_compute = ceil(F_req / dev.F_dev_sust_flop_s)
    N_state = ceil(B_state / dev.B_dev_mem_bytes)
    B_instant = B_state + step.B_step_bytes
    N_instant = ceil(B_instant / dev.B_dev_mem_bytes)
    N_min = max(N_compute, N_state, N_instant)

    # ------------------------------------------------------------------
    # Step timing (algorithmic, no parallelism)
    # ------------------------------------------------------------------
    Tok_per_step = schedule.Tok_per_step
    t_step_max = Tok_per_step / R_tok          # deadline per step [s]
    F_step = w.c * w.P * Tok_per_step         # FLOPs per step
    N_steps = w.Tok / Tok_per_step
    t_compute_min = F_step / (N_min * dev.F_dev_sust_flop_s)

    # ------------------------------------------------------------------
    # Update payload (global, algorithmic — topology assigned in Step 1)
    # ------------------------------------------------------------------
    B_update = update.k_update * w.P * update.b_update_per_param

    # ------------------------------------------------------------------
    # Checkpoint timing
    # ------------------------------------------------------------------
    t_ckpt = S_ckpt / storage.BW_ckpt_sust_Bps
    N_ckpt = ceil(w.T / checkpoint.seconds_per_ckpt)
    T_ckpt_total = N_ckpt * t_ckpt

    return {
        "req": {
            "F_total_flop": F_total,
            "F_req_flop_s": F_req,
            "R_tok_req_tok_s": R_tok,
            "B_weights_min_bytes": B_weights,
            "B_grads_min_bytes": B_grads,
            "B_opt_min_bytes": B_opt,
            "B_state_min_bytes": B_state,
            "BW_dataset_plan_Bps": BW_dataset,
            "B_dataset_total_bytes": B_dataset_total,
            "S_ckpt_bytes": S_ckpt,
            "BW_ckpt_req_Bps": BW_ckpt_req,
        },
        "instant": {
            "B_step_bytes": step.B_step_bytes,
            "B_instant_min_bytes": B_instant,
        },
        "device_bounds": {
            "N_compute_lower_bound": float(N_compute),
            "N_state_memory_lower_bound": float(N_state),
            "N_instant_device_lower_bound": float(N_instant),
            "N_min_lower_bound": float(N_min),
        },
        "movement": {
            "Tok_per_step": Tok_per_step,
            "t_step_max_s": t_step_max,
            "B_update_total_bytes_per_step": B_update,
            "BW_update_global_min_Bps": B_update / t_step_max,
        },
        "stepfacts": {
            "N_steps": N_steps,
            "F_step_flop": F_step,
        },
        "timecmp": {
            "t_step_max_s": t_step_max,
            "F_step_flop": F_step,
            "t_step_compute_min_s": t_compute_min,
            "N_guess_devices": float(N_min),
        },
        "ckpt": {
            "S_ckpt_bytes": S_ckpt,
            "BW_ckpt_sust_Bps": storage.BW_ckpt_sust_Bps,
            "seconds_per_ckpt": checkpoint.seconds_per_ckpt,
            "t_ckpt_min_s": t_ckpt,
            "N_ckpt": float(N_ckpt),
            "T_ckpt_total_min_s": T_ckpt_total,
            "ckpt_fraction_of_run": T_ckpt_total / w.T,
        },
        # Capabilities forwarded to Step 1 — do not recompute there
        "meta": {
            "F_dev_sust_flop_s": float(dev.F_dev_sust_flop_s),
            "B_dev_mem_bytes": float(dev.B_dev_mem_bytes),
            "BW_fabric_node_sust_Bps": float(fabric.BW_node_sust_Bps),
        },
    }
