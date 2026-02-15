# first-principles/briefs/loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from step0_fundamentals.fundamentals import (
    Workload, StateBytes, IO,
    Device, StepWorkingSet, StepSchedule,
    AlgorithmStepFacts, FabricCapability,
    StorageCapability, CheckpointPolicy,
)

from step1_design.design import DesignInputs
from step2_powerandthermals.powerandthermals import (
    PowerInputs, ThermalInputs, AirCoolingInputs, LiquidCoolingInputs, RackInputs
)


def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


def _as_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Expected a number for '{name}', got: {x!r}")


def _as_int(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception:
        raise ValueError(f"Expected an int for '{name}', got: {x!r}")


@dataclass(frozen=True)
class Brief012:
    # Step 0 inputs
    w: Workload
    st: StateBytes
    io: IO
    dev: Device
    step: StepWorkingSet
    sched: StepSchedule
    facts: AlgorithmStepFacts
    fabric: FabricCapability
    storage: StorageCapability
    ckpt_policy: CheckpointPolicy

    # Step 1 inputs
    design_inputs: DesignInputs
    design_G: Optional[int]

    # Step 2 inputs
    pt_power: PowerInputs
    pt_thermals: ThermalInputs
    pt_rack: RackInputs


def load_brief_yaml(path: str) -> Brief012:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Install it with: pip install pyyaml\n"
            f"Cannot load YAML brief: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict):
        raise ValueError("Brief YAML must parse to a mapping/dict at the top level.")

    # ---------------------------------------------------------------------
    # Workload
    # ---------------------------------------------------------------------
    wl = _req(doc, "workload")
    w = Workload(
        P=_as_float(_req(wl, "P"), "workload.P"),
        Tok=_as_float(_req(wl, "Tok"), "workload.Tok"),
        T=_as_float(_req(wl, "T"), "workload.T"),
        c=_as_float(wl.get("c", 6.0), "workload.c"),
    )

    # ---------------------------------------------------------------------
    # Numerics
    # ---------------------------------------------------------------------
    sb = doc.get("state_bytes", {}) or {}
    st = StateBytes(
        b_w=_as_float(sb.get("b_w", 2.0), "state_bytes.b_w"),
        b_g=_as_float(sb.get("b_g", 2.0), "state_bytes.b_g"),
        b_opt=_as_float(sb.get("b_opt", 8.0), "state_bytes.b_opt"),
    )

    io0 = doc.get("io", {}) or {}
    io = IO(
        b_tok=_as_float(io0.get("b_tok", 2.0), "io.b_tok"),
        A_io=_as_float(io0.get("A_io", 1.3), "io.A_io"),
        b_ckpt=_as_float(io0.get("b_ckpt", 2.0), "io.b_ckpt"),
        t_ckpt_max=_as_float(io0.get("t_ckpt_max", 300.0), "io.t_ckpt_max"),
    )

    # ---------------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------------
    dv = _req(doc, "device")
    dev = Device(
        F_dev_sust_flop_s=_as_float(_req(dv, "F_dev_sust_flop_s"), "device.F_dev_sust_flop_s"),
        B_dev_mem_bytes=_as_float(_req(dv, "B_dev_mem_bytes"), "device.B_dev_mem_bytes"),
    )

    # ---------------------------------------------------------------------
    # Step working set + schedule + update facts
    # ---------------------------------------------------------------------
    stp = _req(doc, "step")
    step = StepWorkingSet(B_step_bytes=_as_float(_req(stp, "B_step_bytes"), "step.B_step_bytes"))
    sched = StepSchedule(Tok_per_step=_as_float(_req(stp, "Tok_per_step"), "step.Tok_per_step"))

    upd = stp.get("update", {}) or {}
    facts = AlgorithmStepFacts(
        b_update_per_param=_as_float(_req(upd, "b_update_per_param"), "step.update.b_update_per_param"),
        k_update=_as_float(upd.get("k_update", 1.0), "step.update.k_update"),
    )

    # ---------------------------------------------------------------------
    # Capabilities (Step 0 informational knobs)
    # ---------------------------------------------------------------------
    caps = doc.get("capabilities", {}) or {}

    fab = caps.get("fabric", {}) or {}
    fabric = FabricCapability(
        BW_node_sust_Bps=_as_float(
            fab.get("BW_node_sust_Bps", 0.0),
            "capabilities.fabric.BW_node_sust_Bps",
        )
    )

    stc = caps.get("storage", {}) or {}
    storage = StorageCapability(
        BW_ckpt_sust_Bps=_as_float(
            stc.get("BW_ckpt_sust_Bps", 0.0),
            "capabilities.storage.BW_ckpt_sust_Bps",
        )
    )

    ck = caps.get("checkpoint_policy", {}) or {}
    ckpt_policy = CheckpointPolicy(
        seconds_per_ckpt=_as_float(
            ck.get("seconds_per_ckpt", 3600.0),
            "capabilities.checkpoint_policy.seconds_per_ckpt",
        )
    )

    # ---------------------------------------------------------------------
    # Step 1 design knobs
    # ---------------------------------------------------------------------
    d1 = doc.get("design", {}) or {}

    design_G_raw = d1.get("G", None)
    if design_G_raw is None:
        design_G = None
    else:
        # allow YAML "null" (parsed as None) or explicit int
        design_G = _as_int(design_G_raw, "design.G")

    design_inputs = DesignInputs(
        gpus_per_node=_as_int(d1.get("gpus_per_node", 8), "design.gpus_per_node"),
        eta_compute=_as_float(d1.get("eta_compute", 0.35), "design.eta_compute"),
        eta_fabric=_as_float(d1.get("eta_fabric", 0.80), "design.eta_fabric"),
        tp_max=_as_int(d1.get("tp_max", 16), "design.tp_max"),
        pp_max=_as_int(d1.get("pp_max", 16), "design.pp_max"),
        g_max_multiplier=_as_int(d1.get("g_max_multiplier", 8), "design.g_max_multiplier"),
        comm_model=str(d1.get("comm_model", "ring_allreduce_dp_only")),
        comm_exposed_fraction=_as_float(d1.get("comm_exposed_fraction", 1.0),
                                    "design.comm_exposed_fraction"),
    )

    # ---------------------------------------------------------------------
    # Step 2 power & thermals knobs
    # ---------------------------------------------------------------------
    pt = doc.get("power_thermals", {}) or {}

    pwr = pt.get("power", {}) or {}
    pt_power = PowerInputs(
        P_gpu_W=_as_float(pwr.get("P_gpu_W", 700.0), "power_thermals.power.P_gpu_W"),
        P_cpu_W_per_node=_as_float(pwr.get("P_cpu_W_per_node", 250.0), "power_thermals.power.P_cpu_W_per_node"),
        P_other_W_per_node=_as_float(pwr.get("P_other_W_per_node", 300.0), "power_thermals.power.P_other_W_per_node"),
        PUE=_as_float(pwr.get("PUE", 1.30), "power_thermals.power.PUE"),
    )

    cool = pt.get("cooling", {}) or {}
    mode = str(cool.get("mode", "air")).strip()
    pt_thermals = ThermalInputs(
        mode=mode,  # Step 2 validates allowed values
        air=AirCoolingInputs(
            deltaT_C=_as_float(cool.get("deltaT_air_C", 15.0), "power_thermals.cooling.deltaT_air_C")
        ),
        liquid=LiquidCoolingInputs(
            deltaT_C=_as_float(cool.get("deltaT_liquid_C", 7.0), "power_thermals.cooling.deltaT_liquid_C")
        ),
    )

    rk = pt.get("rack", {}) or {}
    pt_rack = RackInputs(
        nodes_per_rack=None if rk.get("nodes_per_rack", None) is None else _as_int(rk["nodes_per_rack"], "power_thermals.rack.nodes_per_rack"),
        rack_power_limit_W=None if rk.get("rack_power_limit_W", None) is None else _as_float(rk["rack_power_limit_W"], "power_thermals.rack.rack_power_limit_W"),
        racks=None if rk.get("racks", None) is None else _as_int(rk["racks"], "power_thermals.rack.racks"),
    )

    return Brief012(
        w=w,
        st=st,
        io=io,
        dev=dev,
        step=step,
        sched=sched,
        facts=facts,
        fabric=fabric,
        storage=storage,
        ckpt_policy=ckpt_policy,
        design_inputs=design_inputs,
        design_G=design_G,
        pt_power=pt_power,
        pt_thermals=pt_thermals,
        pt_rack=pt_rack,
    )

