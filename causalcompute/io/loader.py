"""YAML brief loader — converts a brief YAML file into a typed Brief object."""
from __future__ import annotations

from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

from ..core.types import (
    AlgorithmStepFacts,
    AirCoolingInputs,
    Brief,
    CheckpointPolicy,
    DesignInputs,
    Device,
    FabricCapability,
    IO,
    LiquidCoolingInputs,
    ComputeNodeSpec,
    CostInputs,
    NetworkInputs,
    PowerInputs,
    StorageInputs,
    RackInputs,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    ThermalInputs,
    Workload,
)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _get(d: dict, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _req(d: dict, key: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise KeyError(f"Brief is missing required field: {key!r}")
    return d[key]


def _float(val: Any, name: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        raise ValueError(f"Expected a number for {name!r}, got {val!r}")


def _int(val: Any, name: str) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        raise ValueError(f"Expected an integer for {name!r}, got {val!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_brief(path: str) -> Brief:
    """
    Parse a YAML brief file and return a fully validated Brief object.

    Raises
    ------
    RuntimeError
        If PyYAML is not installed.
    KeyError
        If a required field is missing.
    ValueError
        If a field has the wrong type or an out-of-range value.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Run: pip install pyyaml\n"
            f"Cannot load brief: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict):
        raise ValueError("Brief YAML must be a mapping at the top level.")

    return _parse(doc)


def load_brief_dict(d: dict) -> Brief:
    """Parse a Brief from a plain Python dict (useful for tests and the UI)."""
    return _parse(d)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse(doc: dict) -> Brief:
    # -- Workload ----------------------------------------------------------
    wl = _req(doc, "workload")
    workload = Workload(
        P=_float(_req(wl, "P"), "workload.P"),
        Tok=_float(_req(wl, "Tok"), "workload.Tok"),
        T=_float(_req(wl, "T"), "workload.T"),
        c=_float(_get(wl, "c", 6.0), "workload.c"),
    )

    # -- State bytes -------------------------------------------------------
    sb = doc.get("state_bytes") or {}
    state = StateBytes(
        b_w=_float(_get(sb, "b_w", 2.0), "state_bytes.b_w"),
        b_g=_float(_get(sb, "b_g", 2.0), "state_bytes.b_g"),
        b_opt=_float(_get(sb, "b_opt", 8.0), "state_bytes.b_opt"),
    )

    # -- I/O ---------------------------------------------------------------
    io_raw = doc.get("io") or {}
    io = IO(
        b_tok=_float(_get(io_raw, "b_tok", 2.0), "io.b_tok"),
        A_io=_float(_get(io_raw, "A_io", 1.3), "io.A_io"),
        b_ckpt=_float(_get(io_raw, "b_ckpt", 2.0), "io.b_ckpt"),
        t_ckpt_max=_float(_get(io_raw, "t_ckpt_max", 300.0), "io.t_ckpt_max"),
    )

    # -- Device ------------------------------------------------------------
    dv = _req(doc, "device")
    device = Device(
        F_dev_sust_flop_s=_float(_req(dv, "F_dev_sust_flop_s"), "device.F_dev_sust_flop_s"),
        B_dev_mem_bytes=_float(_req(dv, "B_dev_mem_bytes"), "device.B_dev_mem_bytes"),
    )

    # -- Step & schedule ---------------------------------------------------
    stp = _req(doc, "step")
    step = StepWorkingSet(
        B_step_bytes=_float(_req(stp, "B_step_bytes"), "step.B_step_bytes")
    )
    schedule = StepSchedule(
        Tok_per_step=_float(_req(stp, "Tok_per_step"), "step.Tok_per_step")
    )

    upd = _get(stp, "update") or {}
    update = AlgorithmStepFacts(
        b_update_per_param=_float(
            _req(upd, "b_update_per_param"), "step.update.b_update_per_param"
        ),
        k_update=_float(_get(upd, "k_update", 1.0), "step.update.k_update"),
    )

    # -- Capabilities (fabric, storage, checkpoint) ------------------------
    caps = doc.get("capabilities") or {}

    fab = _get(caps, "fabric") or {}
    fabric = FabricCapability(
        BW_node_sust_Bps=_float(_get(fab, "BW_node_sust_Bps", 0.0), "capabilities.fabric.BW_node_sust_Bps")
    )

    sto = _get(caps, "storage") or {}
    storage = StorageCapability(
        BW_ckpt_sust_Bps=_float(_get(sto, "BW_ckpt_sust_Bps", 0.0), "capabilities.storage.BW_ckpt_sust_Bps")
    )

    ck = _get(caps, "checkpoint_policy") or {}
    checkpoint = CheckpointPolicy(
        seconds_per_ckpt=_float(_get(ck, "seconds_per_ckpt", 3600.0), "capabilities.checkpoint_policy.seconds_per_ckpt")
    )

    # -- Design (Step 1 knobs) ---------------------------------------------
    d1 = doc.get("design") or {}

    raw_G = _get(d1, "G", None)
    design_G: Optional[int] = None if raw_G is None else _int(raw_G, "design.G")

    design = DesignInputs(
        gpus_per_node=_int(_get(d1, "gpus_per_node", 8), "design.gpus_per_node"),
        eta_compute=_float(_get(d1, "eta_compute", 0.35), "design.eta_compute"),
        eta_fabric=_float(_get(d1, "eta_fabric", 0.80), "design.eta_fabric"),
        tp_max=_int(_get(d1, "tp_max", 16), "design.tp_max"),
        pp_max=_int(_get(d1, "pp_max", 16), "design.pp_max"),
        g_max_multiplier=_int(_get(d1, "g_max_multiplier", 8), "design.g_max_multiplier"),
        comm_model=str(_get(d1, "comm_model", "ring_allreduce_dp_only")),
        comm_exposed_fraction=_float(_get(d1, "comm_exposed_fraction", 1.0), "design.comm_exposed_fraction"),
    )

    # -- Power & thermals (Step 2 knobs) -----------------------------------
    pt = doc.get("power_thermals") or {}

    pwr = _get(pt, "power") or {}
    power = PowerInputs(
        P_gpu_W=_float(_get(pwr, "P_gpu_W", 700.0), "power_thermals.power.P_gpu_W"),
        P_cpu_W_per_node=_float(_get(pwr, "P_cpu_W_per_node", 250.0), "power_thermals.power.P_cpu_W_per_node"),
        P_other_W_per_node=_float(_get(pwr, "P_other_W_per_node", 300.0), "power_thermals.power.P_other_W_per_node"),
        PUE=_float(_get(pwr, "PUE", 1.30), "power_thermals.power.PUE"),
    )

    cool = _get(pt, "cooling") or {}
    thermals = ThermalInputs(
        mode=str(_get(cool, "mode", "air")),
        air=AirCoolingInputs(
            deltaT_C=_float(_get(cool, "deltaT_air_C", 15.0), "power_thermals.cooling.deltaT_air_C")
        ),
        liquid=LiquidCoolingInputs(
            deltaT_C=_float(_get(cool, "deltaT_liquid_C", 7.0), "power_thermals.cooling.deltaT_liquid_C")
        ),
    )

    rk = _get(pt, "rack") or {}
    raw_npr = _get(rk, "nodes_per_rack", None)
    raw_lim = _get(rk, "rack_power_limit_W", None)
    raw_racks = _get(rk, "racks", None)
    rack = RackInputs(
        nodes_per_rack=None if raw_npr is None else _int(raw_npr, "power_thermals.rack.nodes_per_rack"),
        rack_power_limit_W=None if raw_lim is None else _float(raw_lim, "power_thermals.rack.rack_power_limit_W"),
        racks=None if raw_racks is None else _int(raw_racks, "power_thermals.rack.racks"),
    )

    # -- Network (Step 3) — fully optional, defaults to HPC IB rail-optimised --
    nw = doc.get("network") or {}
    network = NetworkInputs(
        nics_per_node=_int(_get(nw, "nics_per_node", 8), "network.nics_per_node"),
        switch_radix=_int(_get(nw, "switch_radix", 64), "network.switch_radix"),
        port_bw_Bps=_float(_get(nw, "port_bw_Bps", 5.0e10), "network.port_bw_Bps"),
        oversubscription=_float(_get(nw, "oversubscription", 1.0), "network.oversubscription"),
        rail_optimised=bool(_get(nw, "rail_optimised", True)),
    )

    # -- Compute node spec (Step 5) — fully optional, defaults to H100 SXM HGX --
    cn = doc.get("compute_node") or {}
    node_spec = ComputeNodeSpec(
        intra_node_bw_Bps=_float(_get(cn, "intra_node_bw_Bps", 9.0e11), "compute_node.intra_node_bw_Bps"),
        gpudirect_rdma=bool(_get(cn, "gpudirect_rdma", True)),
        cpu_cores_per_node=_int(_get(cn, "cpu_cores_per_node", 128), "compute_node.cpu_cores_per_node"),
        dram_bytes_per_node=_float(_get(cn, "dram_bytes_per_node", 2.0e12), "compute_node.dram_bytes_per_node"),
        root_complex=str(_get(cn, "root_complex", "dual_socket")),
    )

    # -- Cost model (Step 6) — fully optional, defaults to ~2024 ballpark prices --
    co = doc.get("cost") or {}
    cost_inputs = CostInputs(
        gpu_unit_cost=_float(_get(co, "gpu_unit_cost", 30_000.0), "cost.gpu_unit_cost"),
        node_chassis_cost=_float(_get(co, "node_chassis_cost", 15_000.0), "cost.node_chassis_cost"),
        nic_unit_cost=_float(_get(co, "nic_unit_cost", 2_500.0), "cost.nic_unit_cost"),
        switch_unit_cost=_float(_get(co, "switch_unit_cost", 40_000.0), "cost.switch_unit_cost"),
        cable_unit_cost=_float(_get(co, "cable_unit_cost", 100.0), "cost.cable_unit_cost"),
        storage_node_cost=_float(_get(co, "storage_node_cost", 20_000.0), "cost.storage_node_cost"),
        drive_unit_cost=_float(_get(co, "drive_unit_cost", 2_500.0), "cost.drive_unit_cost"),
        rack_unit_cost=_float(_get(co, "rack_unit_cost", 3_000.0), "cost.rack_unit_cost"),
        electricity_usd_kwh=_float(_get(co, "electricity_usd_kwh", 0.07), "cost.electricity_usd_kwh"),
        capex_amortization_years=_float(_get(co, "capex_amortization_years", 3.0), "cost.capex_amortization_years"),
    )

    # -- Storage design (Step 4) — fully optional, defaults to NVMe all-flash --
    sd = doc.get("storage_design") or {}
    storage_inputs = StorageInputs(
        drive_bw_seq_Bps=_float(_get(sd, "drive_bw_seq_Bps", 7.0e9), "storage_design.drive_bw_seq_Bps"),
        drive_capacity_bytes=_float(_get(sd, "drive_capacity_bytes", 7.68e12), "storage_design.drive_capacity_bytes"),
        drives_per_storage_node=_int(_get(sd, "drives_per_storage_node", 24), "storage_design.drives_per_storage_node"),
        storage_net_bw_Bps=_float(_get(sd, "storage_net_bw_Bps", 2.5e10), "storage_design.storage_net_bw_Bps"),
        dataset_replication=_int(_get(sd, "dataset_replication", 1), "storage_design.dataset_replication"),
        ckpt_replication=_int(_get(sd, "ckpt_replication", 2), "storage_design.ckpt_replication"),
        ckpt_keep_count=_int(_get(sd, "ckpt_keep_count", 3), "storage_design.ckpt_keep_count"),
    )

    return Brief(
        workload=workload,
        state=state,
        io=io,
        device=device,
        step=step,
        schedule=schedule,
        update=update,
        fabric=fabric,
        storage=storage,
        checkpoint=checkpoint,
        design=design,
        design_G=design_G,
        power=power,
        thermals=thermals,
        rack=rack,
        network=network,
        storage_inputs=storage_inputs,
        node_spec=node_spec,
        cost=cost_inputs,
    )
