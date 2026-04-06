# first-principles/step0_fundamentals/fundamentals.py
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pprint import pprint
import argparse


# =============================================================================
# Inputs (SI units)
# =============================================================================

@dataclass(frozen=True)
class Workload:
    P: float          # parameters [params]
    Tok: float        # total training tokens [tokens]
    T: float          # deadline [seconds]
    c: float = 6.0    # FLOPs per param-token (dense transformer ~6)


@dataclass(frozen=True)
class StateBytes:
    """
    Lower-bound information that must exist at least once somewhere to train.

    b_w   : weight bytes per param
    b_g   : gradient bytes per param (materialized at least transiently)
    b_opt : optimizer state bytes per param (optimizer-dependent; Adam-like is large)
    """
    b_w: float = 2.0
    b_g: float = 2.0
    b_opt: float = 8.0


@dataclass(frozen=True)
class IO:
    b_tok: float = 2.0        # bytes/token for dataset stream
    A_io: float = 1.3         # dataset bandwidth headroom multiplier
    b_ckpt: float = 2.0       # bytes/param written per checkpoint (often weights only)
    t_ckpt_max: float = 300.0 # max allowed checkpoint time [seconds]


# =============================================================================
# Fundamentals (no cluster assumptions)
# =============================================================================

def fundamentals(w: Workload, st: StateBytes, io: IO) -> dict[str, float]:
    """
    Returns absolute workload requirements in SI units.
    """
    # Compute (unambiguous)
    F_total = w.c * w.P * w.Tok          # [FLOP]
    F_req = F_total / w.T               # [FLOP/s]
    R_tok_req = w.Tok / w.T             # [tokens/s]

    # Model-state information (lower bounds, single-copy)
    B_weights = st.b_w * w.P            # [bytes]
    B_grads = st.b_g * w.P              # [bytes]
    B_opt = st.b_opt * w.P              # [bytes]
    B_state_min = B_weights + B_grads + B_opt

    # I/O contracts
    BW_dataset_ideal = R_tok_req * io.b_tok      # [bytes/s]
    BW_dataset_plan = BW_dataset_ideal * io.A_io

    S_ckpt = w.P * io.b_ckpt                     # [bytes]
    BW_ckpt_req = S_ckpt / io.t_ckpt_max         # [bytes/s]

    return {
        # Compute
        "F_total_flop": F_total,
        "F_req_flop_s": F_req,
        "R_tok_req_tok_s": R_tok_req,

        # Model-state (lower bounds)
        "B_weights_min_bytes": B_weights,
        "B_grads_min_bytes": B_grads,
        "B_opt_min_bytes": B_opt,
        "B_state_min_bytes": B_state_min,

        # I/O
        "BW_dataset_ideal_Bps": BW_dataset_ideal,
        "BW_dataset_plan_Bps": BW_dataset_plan,
        "S_ckpt_bytes": S_ckpt,
        "BW_ckpt_req_Bps": BW_ckpt_req,
    }


# =============================================================================
# Next-layer primitives (still topology-agnostic)
# =============================================================================

@dataclass(frozen=True)
class Device:
    """
    One unit of silicon, described in SI units:
    - F_dev_sust_flop_s : sustained compute capability [FLOP/s]
    - B_dev_mem_bytes   : on-device memory capacity [bytes]
    """
    F_dev_sust_flop_s: float
    B_dev_mem_bytes: float


def device_count_lower_bounds(req: dict[str, float], dev: Device) -> dict[str, float]:
    """
    Pure lower bounds (still no parallelism or cluster design).
    """
    N_compute = ceil(req["F_req_flop_s"] / dev.F_dev_sust_flop_s)
    N_state = ceil(req["B_state_min_bytes"] / dev.B_dev_mem_bytes)
    return {
        "N_compute_lower_bound": float(N_compute),
        "N_state_memory_lower_bound": float(N_state),
        "N_min_lower_bound": float(max(N_compute, N_state)),
    }


@dataclass(frozen=True)
class StepWorkingSet:
    """
    Instantaneous working set needed to execute ONE update step (activations/temps/etc).
    This is global, not per-device.
    """
    B_step_bytes: float


def instantaneous_memory_requirements(req: dict[str, float], step: StepWorkingSet) -> dict[str, float]:
    B_instant_min = req["B_state_min_bytes"] + step.B_step_bytes
    return {
        "B_step_bytes": step.B_step_bytes,
        "B_instant_min_bytes": B_instant_min,
    }


def instantaneous_device_lower_bound(B_instant_min_bytes: float, dev: Device) -> float:
    """Device lower bound to hold instantaneous requirement somewhere."""
    return float(ceil(B_instant_min_bytes / dev.B_dev_mem_bytes))


@dataclass(frozen=True)
class StepSchedule:
    """
    How many tokens are processed per update step (global).
    Algorithm/policy input, not hardware.
    """
    Tok_per_step: float  # [tokens/step]


@dataclass(frozen=True)
class AlgorithmStepFacts:
    """
    Per-step update signal size model (topology-agnostic).
    """
    b_update_per_param: float  # [bytes/param]
    k_update: float = 1.0      # multiplier >= 1 for margin


def derive_update_payload_per_step(P: float, facts: AlgorithmStepFacts) -> float:
    """
    Step-0 invariant: how many bytes of update payload participate per step globally.
    This is NOT a fabric requirement; it is "information moved/combined".
    """
    return facts.k_update * P * facts.b_update_per_param


def movement_facts_per_step(
    req: dict[str, float],
    sched: StepSchedule,
    B_update_total_bytes_per_step: float,
) -> dict[str, float]:
    """
    Step-0 movement facts (topology/parallelism-independent).

    We expose:
      - t_step_max_s from token rate
      - global update payload per step (bytes/step)
      - a *global* lower-bound rate (bytes/s) = payload / t_step_max, useful for narrative only
    """
    t_step_max = sched.Tok_per_step / req["R_tok_req_tok_s"]      # [s/step]
    BW_update_global_min = B_update_total_bytes_per_step / t_step_max  # [bytes/s] global lower bound (NOT per-node)
    return {
        "Tok_per_step": sched.Tok_per_step,
        "t_step_max_s": t_step_max,
        "B_update_total_bytes_per_step": B_update_total_bytes_per_step,
        "BW_update_global_min_Bps": BW_update_global_min,
    }


# =============================================================================
# steps, per-step compute, and step time-scales
# =============================================================================

def step_compute_facts(w: Workload, sched: StepSchedule) -> dict[str, float]:
    N_steps = w.Tok / sched.Tok_per_step
    F_step = w.c * w.P * sched.Tok_per_step
    return {
        "N_steps": N_steps,
        "F_step_flop": F_step,
    }


def step_time_consistency(
    req: dict[str, float],
    w: Workload,
    dev: Device,
    sched: StepSchedule,
    N_devices: float,
) -> dict[str, float]:
    t_step_max = sched.Tok_per_step / req["R_tok_req_tok_s"]  # [s/step]
    F_step = w.c * w.P * sched.Tok_per_step                   # [FLOP/step]
    t_step_compute_min = F_step / (N_devices * dev.F_dev_sust_flop_s)  # [s/step]
    N_devices_time_only = float(ceil(F_step / (t_step_max * dev.F_dev_sust_flop_s)))
    return {
        "t_step_max_s": t_step_max,
        "F_step_flop": F_step,
        "t_step_compute_min_s": t_step_compute_min,
        "N_devices_time_only": N_devices_time_only,
    }


# =============================================================================
# Fabric abstraction stays as a capability meta-number for Step 1+
# =============================================================================

@dataclass(frozen=True)
class FabricCapability:
    """
    Topology-agnostic fabric abstraction used by Step 1.

    IMPORTANT CONTRACT (v1):
      - BW_node_sust_Bps is the sustained payload bandwidth available per *node*
        for inter-node collectives (after protocol/stack effects are applied later via eta_fabric).

    Step 1 computes inter-node bytes per *node per step* and divides by this value.
    """
    BW_node_sust_Bps: float  # [bytes/s]


# =============================================================================
# Residual step-time budget after compute + (comm to be computed in Step 1)
# =============================================================================

def step_time_budget(t_step_max_s: float, t_step_compute_min_s: float, t_comm_s: float) -> dict[str, float]:
    t_residual = t_step_max_s - t_step_compute_min_s - t_comm_s
    residual_fraction = t_residual / t_step_max_s
    accounted_fraction = (t_step_compute_min_s + t_comm_s) / t_step_max_s
    return {
        "t_residual_s": t_residual,
        "residual_fraction_of_step": residual_fraction,
        "accounted_fraction_of_step": accounted_fraction,
    }


# =============================================================================
# Checkpoint time-scale from storage BW + wall-clock policy
# =============================================================================

@dataclass(frozen=True)
class StorageCapability:
    """Checkpoint storage abstraction: sustained write bandwidth available for checkpoint writes."""
    BW_ckpt_sust_Bps: float  # [bytes/s]


@dataclass(frozen=True)
class CheckpointPolicy:
    """Wall-clock checkpointing policy (physics-oriented)."""
    seconds_per_ckpt: float  # [s/checkpoint]


def checkpoint_time_scales(
    req: dict[str, float],
    w: Workload,
    storage: StorageCapability,
    policy: CheckpointPolicy,
) -> dict[str, float]:
    S_ckpt = req["S_ckpt_bytes"]                      # [bytes]
    t_ckpt_min = S_ckpt / storage.BW_ckpt_sust_Bps         # [s/ckpt]
    N_ckpt = float(ceil(w.T / policy.seconds_per_ckpt))
    T_ckpt_total_min = N_ckpt * t_ckpt_min
    ckpt_fraction_of_run = T_ckpt_total_min / w.T
    return {
        "S_ckpt_bytes": S_ckpt,
        "BW_ckpt_sust_Bps": storage.BW_ckpt_sust_Bps,
        "seconds_per_ckpt": policy.seconds_per_ckpt,
        "t_ckpt_min_s": t_ckpt_min,
        "N_ckpt": N_ckpt,
        "T_ckpt_total_min_s": T_ckpt_total_min,
        "ckpt_fraction_of_run": ckpt_fraction_of_run,
    }


# =============================================================================
# Step 0 addition: meta (capabilities/invariants for Step 1)
# =============================================================================

def _build_meta(dev: Device, fabric: FabricCapability) -> dict[str, float]:
    """
    meta = capabilities/invariants that Step 1 (design layer) is allowed to use.
    Still physics: we only expose device + fabric capability numbers.
    """
    return {
        "F_dev_sust_flop_s": float(dev.F_dev_sust_flop_s),
        "B_dev_mem_bytes": float(dev.B_dev_mem_bytes),
        # Interpreted downstream as per-rank sustained payload capability for collectives
        "BW_fabric_node_sust_Bps": float(fabric.BW_node_sust_Bps),
    }


# =============================================================================
# Bundle runner (output for design.py)
# =============================================================================

def run_fundamentals(
    w: Workload,
    st: StateBytes,
    io: IO,
    dev: Device,
    step: StepWorkingSet,
    sched: StepSchedule,
    facts: AlgorithmStepFacts,
    fabric: FabricCapability,
    storage: StorageCapability,
    ckpt_policy: CheckpointPolicy,
) -> dict[str, dict[str, float]]:
    """
    Single entrypoint for fundamentals.

    Returns a python dict bundle with stable keys so design.py can consume it.
    """
    req = fundamentals(w, st, io)

    bounds = device_count_lower_bounds(req, dev)
    inst = instantaneous_memory_requirements(req, step)
    N_instant = instantaneous_device_lower_bound(inst["B_instant_min_bytes"], dev)

    # Strongest "must exist" device count so far (still not placement)
    N_guess = float(max(bounds["N_min_lower_bound"], N_instant))

    B_update_total = derive_update_payload_per_step(w.P, facts)
    mv = movement_facts_per_step(req, sched, B_update_total)

    stepfacts = step_compute_facts(w, sched)
    timecmp = step_time_consistency(req, w, dev, sched, N_devices=N_guess)

    ckpt = checkpoint_time_scales(req, w, storage, ckpt_policy)

    return {
        "req": req,
        "device_bounds": bounds,
        "instant": {**inst, "N_instant_device_lower_bound": float(N_instant)},
        "movement": mv,
        "stepfacts": stepfacts,
        "timecmp": {**timecmp, "N_guess_devices": float(N_guess)},
        "ckpt": ckpt,

        # capabilities/invariants for Step 1+
        "meta": _build_meta(dev=dev, fabric=fabric),
    }


# =============================================================================
# Debug story renderer (kept, but comm is now instantiated in Step 1)
# =============================================================================

def si(x: float) -> str:
    return f"{x:.6e}"


def print_report(
    w: Workload,
    st: StateBytes,
    io: IO,
    dev: Device,
    step: StepWorkingSet,
    sched: StepSchedule,
    facts: AlgorithmStepFacts,
    fabric: FabricCapability,
    storage: StorageCapability,
    ckpt_policy: CheckpointPolicy,
) -> None:
    b = run_fundamentals(w, st, io, dev, step, sched, facts, fabric, storage, ckpt_policy)

    req = b["req"]
    bounds = b["device_bounds"]
    inst = b["instant"]
    mv = b["movement"]
    stepfacts = b["stepfacts"]
    timecmp = b["timecmp"]
    ckpt = b["ckpt"]

    print("I want to run a Large Scale Distributed Training workload.\n")

    print(
        f"I want to train a model of {si(w.P)} parameters "
        f"using {si(w.Tok)} tokens in {si(w.T)} seconds:\n"
    )

    print(
        "FUNDAMENTAL QUESTION: What physical resources must exist in total "
        f"to train a model of {si(w.P)} parameters using {si(w.Tok)} tokens "
        f"in {si(w.T)} seconds?\n"
    )

    print(
        f"• Total compute required: {si(req['F_total_flop'])} FLOP\n"
        f"• Sustained compute required: {si(req['F_req_flop_s'])} FLOP/s\n"
        f"• Sustained token rate required: {si(req['R_tok_req_tok_s'])} tokens/s\n"
    )

    print(
        "Model-state information lower bounds (single copy):\n"
        f"• Weights:   {si(req['B_weights_min_bytes'])} B\n"
        f"• Gradients: {si(req['B_grads_min_bytes'])} B\n"
        f"• Opt state: {si(req['B_opt_min_bytes'])} B\n"
        f"• Total:     {si(req['B_state_min_bytes'])} B\n"
    )

    print(
        "I/O contracts:\n"
        f"• Dataset BW (planned): {si(req['BW_dataset_plan_Bps'])} B/s\n"
        f"• Checkpoint size:      {si(req['S_ckpt_bytes'])} B\n"
        f"• Checkpoint BW req:    {si(req['BW_ckpt_req_Bps'])} B/s\n"
    )

    print(
        "NEXT FUNDAMENTAL QUESTION: If one unit of silicon can sustain "
        f"{si(dev.F_dev_sust_flop_s)} FLOP/s and has {si(dev.B_dev_mem_bytes)} B memory, "
        "how many such units must exist?\n"
    )

    print(
        f"• Compute lower bound devices: {si(bounds['N_compute_lower_bound'])}\n"
        f"• State-memory lower bound:    {si(bounds['N_state_memory_lower_bound'])}\n"
        f"• Min devices (max of above):  {si(bounds['N_min_lower_bound'])}\n"
    )

    print(
        "NEXT FUNDAMENTAL QUESTION: What must be simultaneously resident to execute ONE step?\n"
        f"• Step working set (input):    {si(inst['B_step_bytes'])} B\n"
        f"• Instantaneous total minimum: {si(inst['B_instant_min_bytes'])} B\n"
        f"• Instantaneous device LB:     {si(inst['N_instant_device_lower_bound'])} devices\n"
    )

    print(
        "NEXT FUNDAMENTAL QUESTION: What update payload exists per step (global, algorithmic)?\n"
        f"• Tokens/step (input):                 {si(mv['Tok_per_step'])}\n"
        f"• Max allowed step time:               {si(mv['t_step_max_s'])} s/step\n"
        f"• Update payload per step (global):    {si(mv['B_update_total_bytes_per_step'])} B/step\n"
        f"• Global lower-bound rate (narrative): {si(mv['BW_update_global_min_Bps'])} B/s\n"
        "\n"
        "NOTE: Turning this payload into a fabric requirement requires choosing DP/TP/PP and a collective model (Step 1).\n"
    )

    print(
        "NEXT FUNDAMENTAL QUESTION: How many update steps exist, how much compute occurs in one step?\n"
    )

    print(
        f"• Total steps in run:            {si(stepfacts['N_steps'])} steps\n"
        f"• Compute per step:              {si(stepfacts['F_step_flop'])} FLOP/step\n"
        f"• Using N =                      {si(timecmp['N_guess_devices'])} devices (strongest lower bound)\n"
        f"• Shortest possible compute time: {si(timecmp['t_step_compute_min_s'])} s/step\n"
        f"• Step time allowed by deadline: {si(timecmp['t_step_max_s'])} s/step\n"
        f"• Devices implied by time alone: {si(timecmp['N_devices_time_only'])} devices\n"
    )

    print(
        "NEXT FUNDAMENTAL QUESTION: If checkpointing is a periodic write of S_ckpt bytes, "
        "and storage can sustain BW_sust, what wall-clock time does checkpointing consume?\n"
    )

    print(
        f"• Wall-clock ckpt period (in):   {si(ckpt['seconds_per_ckpt'])} s\n"
        f"• Minimum checkpoint time:       {si(ckpt['t_ckpt_min_s'])} s/ckpt\n"
        f"• Checkpoints over run:          {si(ckpt['N_ckpt'])}\n"
        f"• Total ckpt time (min):         {si(ckpt['T_ckpt_total_min_s'])} s\n"
        f"• Fraction of run in ckpt:       {si(ckpt['ckpt_fraction_of_run'])} (dimensionless)\n"
    )


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSDT fundamentals (topology-agnostic)")

    p.add_argument(
        "--debug",
        action="store_true",
        help="Print the narrative story in addition to the output bundle.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Example inputs (placeholders)
    w = Workload(P=1e9, Tok=3e11, T=604800)
    st = StateBytes()
    io = IO()

    dev = Device(F_dev_sust_flop_s=1.0e15, B_dev_mem_bytes=8.0e10)
    step = StepWorkingSet(B_step_bytes=5.0e10)
    sched = StepSchedule(Tok_per_step=1.0e7)
    facts = AlgorithmStepFacts(b_update_per_param=st.b_g, k_update=1.0)

    fabric = FabricCapability(BW_sust_Bps=1.0e11)
    storage = StorageCapability(BW_sust_Bps=5.0e9)
    ckpt_policy = CheckpointPolicy(seconds_per_ckpt=1800.0)

    if args.debug:
        print_report(w, st, io, dev, step, sched, facts, fabric, storage, ckpt_policy)
        print("\n---\nDEBUG BUNDLE (python dict)\n---")
        pprint(run_fundamentals(w, st, io, dev, step, sched, facts, fabric, storage, ckpt_policy))
    else:
        pprint(run_fundamentals(w, st, io, dev, step, sched, facts, fabric, storage, ckpt_policy))

