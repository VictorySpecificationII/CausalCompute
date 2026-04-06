"""
Bottleneck identification — post-Step-1 analysis.

No new physics, no new user inputs.  Reads bundle0 (Step 0) and design
(Step 1) to answer: "what is the active constraint that drove the cluster
size, and how tight is it?"

Four dimensions are analysed:

  1. Device lower bounds (compute / state memory / instantaneous memory)
     The binding constraint is whichever lower bound equals N_min and
     therefore forced the cluster to be as large as it is.

  2. Step time budget
     How much of the per-step deadline is consumed by compute and
     communication, and how much headroom remains.

  3. Memory utilisation
     Actual bytes per device vs. device capacity.

  4. Communication fraction
     What fraction of step time is communication overhead — the leading
     indicator of fabric saturation.

Returns a dict consumed by report.py, cli.py, and streamlit_app.py.
"""
from __future__ import annotations

from typing import Any

# Thresholds for generating recommendations
_STEP_TIME_TIGHT_PCT = 10.0   # headroom below this → step time warning
_COMM_HEAVY_PCT = 35.0        # comm above this → fabric warning
_MEM_TIGHT_PCT = 85.0         # memory utilisation above this → memory warning


def run_bottleneck(
    bundle0: dict[str, Any],
    design: dict[str, Any],
) -> dict[str, Any]:
    """
    Identify the active bottleneck from Step 0 and Step 1 results.

    Parameters
    ----------
    bundle0 : dict
        Output of run_fundamentals().
    design : dict
        Output of run_design().  Must be feasible.

    Returns
    -------
    dict with sections: binding, device_bounds, step_time, memory,
    communication, recommendations
    """
    if not design.get("feasible"):
        raise ValueError("run_bottleneck requires a feasible design")

    sol = design["solution"]
    cl = sol["cluster"]
    tim = sol["timing"]
    mem = sol["memory"]

    bounds = bundle0["device_bounds"]
    N_compute = bounds["N_compute_lower_bound"]
    N_state   = bounds["N_state_memory_lower_bound"]
    N_instant = bounds["N_instant_device_lower_bound"]
    N_min     = bounds["N_min_lower_bound"]
    G         = cl["G"]

    # ------------------------------------------------------------------
    # 1. Device lower bounds — which one is binding?
    # ------------------------------------------------------------------
    # Headroom_x = G / N_bound.  The bound closest to G (smallest headroom)
    # is the one that forced the cluster size.

    compute_hx  = G / N_compute  if N_compute  > 0 else float("inf")
    state_hx    = G / N_state    if N_state    > 0 else float("inf")
    instant_hx  = G / N_instant  if N_instant  > 0 else float("inf")

    # N_min is the maximum of the three lower bounds; binding = the one that equals N_min
    bound_map = {
        "compute":  (N_compute,  compute_hx,  "Compute throughput"),
        "state":    (N_state,    state_hx,    "Model-state memory"),
        "instant":  (N_instant,  instant_hx,  "Instantaneous memory"),
    }

    # The binding bound has the minimum headroom (closest to 1.0×)
    binding_key = min(bound_map, key=lambda k: bound_map[k][1])
    binding_label = bound_map[binding_key][2]

    device_bounds = {
        k: {
            "N": v[0],
            "headroom_x": v[1],
            "is_binding": (k == binding_key),
            "label": v[2],
        }
        for k, v in bound_map.items()
    }
    device_bounds["N_min"] = N_min
    device_bounds["G_actual"] = G
    device_bounds["G_over_N_min"] = G / N_min if N_min > 0 else float("inf")

    # ------------------------------------------------------------------
    # 2. Step time
    # ------------------------------------------------------------------
    t_step     = tim["t_step_s"]
    t_step_max = tim["t_step_max_s"]
    t_compute  = tim["t_compute_s"]
    t_comm     = tim["t_comm_s"]
    headroom_s = tim["headroom_s"]

    headroom_pct      = headroom_s / t_step_max * 100 if t_step_max > 0 else 0.0
    compute_pct       = t_compute  / t_step     * 100 if t_step     > 0 else 0.0
    comm_pct          = t_comm     / t_step     * 100 if t_step     > 0 else 0.0

    step_time = {
        "t_step_s":         t_step,
        "t_step_max_s":     t_step_max,
        "t_compute_s":      t_compute,
        "t_comm_s":         t_comm,
        "headroom_s":       headroom_s,
        "headroom_pct":     headroom_pct,
        "compute_pct":      compute_pct,
        "comm_pct":         comm_pct,
        "step_time_tight":  headroom_pct < _STEP_TIME_TIGHT_PCT,
    }

    # ------------------------------------------------------------------
    # 3. Memory utilisation
    # ------------------------------------------------------------------
    mem_used     = mem["bytes_per_device"]
    mem_cap      = mem["B_dev_mem_bytes"]
    mem_util_pct = mem_used / mem_cap * 100 if mem_cap > 0 else 0.0
    mem_headroom = mem_cap - mem_used

    memory = {
        "bytes_per_device":  mem_used,
        "B_dev_mem_bytes":   mem_cap,
        "utilization_pct":   mem_util_pct,
        "headroom_bytes":    mem_headroom,
        "memory_tight":      mem_util_pct > _MEM_TIGHT_PCT,
    }

    # ------------------------------------------------------------------
    # 4. Communication burden
    # ------------------------------------------------------------------
    comm_heavy = comm_pct > _COMM_HEAVY_PCT

    communication = {
        "t_comm_s":    t_comm,
        "comm_pct":    comm_pct,
        "comm_heavy":  comm_heavy,
        "model":       sol["parallelism"].get("comm_model", "ring_allreduce_dp_only"),
    }

    # ------------------------------------------------------------------
    # 5. Recommendations
    # ------------------------------------------------------------------
    recs: list[str] = []

    if binding_key == "instant":
        recs.append(
            "Memory-bound (instantaneous): step working set + model state is the "
            "limiting factor. Increasing TP or PP reduces per-device footprint."
        )
    elif binding_key == "state":
        recs.append(
            "Memory-bound (model state): weights + gradients + optimizer state is "
            "the limiting factor. Consider ZeRO sharding or mixed-precision to reduce "
            "bytes/param."
        )
    elif binding_key == "compute":
        recs.append(
            "Compute-bound: device FLOP/s is the limiting factor. The cluster is "
            "sized by throughput, not memory. This is the ideal regime."
        )

    if step_time["step_time_tight"]:
        recs.append(
            f"Step time is tight ({headroom_pct:.1f}% headroom). Any increase in "
            "communication overhead (larger batch, more PP stages, slower fabric) "
            "will miss the deadline."
        )

    if comm_heavy:
        recs.append(
            f"Communication is {comm_pct:.0f}% of step time — consider gradient "
            "compression, reducing DP degree, or increasing fabric BW."
        )

    if memory["memory_tight"] and binding_key != "instant":
        recs.append(
            f"Memory utilisation is high ({mem_util_pct:.0f}%). "
            "Little room for activation growth or larger micro-batches."
        )

    if device_bounds["G_over_N_min"] > 2.0:
        recs.append(
            f"Cluster is {device_bounds['G_over_N_min']:.1f}× above the device "
            "lower bound — parallelism overhead or feasibility constraints inflated "
            "the cluster size. Review TP/PP/PP choices."
        )

    return {
        "binding": {
            "key":   binding_key,
            "label": binding_label,
        },
        "device_bounds":  device_bounds,
        "step_time":      step_time,
        "memory":         memory,
        "communication":  communication,
        "recommendations": recs,
    }
