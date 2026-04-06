"""
Step 2 — Power & Thermals.

Consumes the Step 1 design bundle and derives:
  - IT power ledger (GPU + CPU + other)
  - Facility power (IT × PUE)
  - Heat load (≈ P_IT at steady state)
  - Cooling mass flow and volumetric flow (air or liquid)
  - Optional rack sanity check
  - Optional energy over the run

All outputs are in SI units (W, kg/s, m³/s).
Display-unit conversions (CFM, LPM) are provided separately for UI use.
"""
from __future__ import annotations

from math import ceil
from typing import Any, Optional

from .types import AirCoolingInputs, LiquidCoolingInputs, PowerInputs, RackInputs, ThermalInputs

_CFM_PER_M3_S = 2118.88       # 1 m³/s = 2118.88 ft³/min
_LPM_PER_M3_S = 60_000.0      # 1 m³/s = 60 000 L/min


def _extract_cluster(design_bundle: dict[str, Any]) -> tuple[int, int, int]:
    """Return (G, nodes, gpus_per_node) from a Step-1 bundle. Raises on infeasible."""
    if not design_bundle.get("feasible", False):
        raise ValueError(
            "Design is infeasible; cannot compute power and thermals. "
            "Check Step 1 diagnostics for hints."
        )
    cluster = design_bundle["handoff"]["cluster"]
    if cluster is None:
        raise ValueError("Step 1 handoff is missing cluster data.")
    G = int(cluster["G"])
    nodes = int(cluster["nodes"])
    gpn = int(cluster["gpus_per_node"])
    if G <= 0 or nodes <= 0 or gpn <= 0:
        raise ValueError(f"Invalid cluster values: G={G}, nodes={nodes}, gpus_per_node={gpn}")
    return G, nodes, gpn


def _cooling_air(Q_W: float, a: AirCoolingInputs) -> dict:
    m_dot = Q_W / (a.cp_J_kgK * a.deltaT_C)     # [kg/s]
    V_dot = m_dot / a.rho_kg_m3                  # [m³/s]
    return {
        "mode": "air",
        "deltaT_C": a.deltaT_C,
        "mass_flow_kg_s": m_dot,
        "vol_flow_m3_s": V_dot,
        "vol_flow_CFM": V_dot * _CFM_PER_M3_S,   # display only
    }


def _cooling_liquid(Q_W: float, l: LiquidCoolingInputs) -> dict:
    m_dot = Q_W / (l.cp_J_kgK * l.deltaT_C)     # [kg/s]
    V_dot = m_dot / l.rho_kg_m3                  # [m³/s]
    return {
        "mode": "liquid",
        "deltaT_C": l.deltaT_C,
        "mass_flow_kg_s": m_dot,
        "vol_flow_m3_s": V_dot,
        "vol_flow_LPM": V_dot * _LPM_PER_M3_S,  # display only
    }


def run_thermals(
    design_bundle: dict[str, Any],
    *,
    power: PowerInputs = PowerInputs(),
    thermals: ThermalInputs = ThermalInputs(),
    rack: RackInputs = RackInputs(),
    T_run_s: Optional[float] = None,
) -> dict[str, Any]:
    """
    Compute power, heat, and cooling requirements for a Step-1 cluster design.

    Parameters
    ----------
    design_bundle : dict
        Output of run_design().
    power : PowerInputs
    thermals : ThermalInputs
    rack : RackInputs
    T_run_s : float or None
        If provided, total energy consumption over the run is computed.

    Returns
    -------
    dict with keys:
        power       — full power ledger
        heat        — heat load
        cooling     — thermal flow results
        rack        — rack sanity (if configured)
        energy      — energy over run (if T_run_s provided)
        handoff     — stable digest for downstream stages
    """
    # -- Validate inputs -------------------------------------------------
    if power.P_gpu_W <= 0:
        raise ValueError("P_gpu_W must be > 0")
    if power.PUE < 1.0:
        raise ValueError("PUE must be ≥ 1.0")
    if thermals.mode not in ("air", "liquid"):
        raise ValueError("thermals.mode must be 'air' or 'liquid'")
    if thermals.air.deltaT_C <= 0 or thermals.liquid.deltaT_C <= 0:
        raise ValueError("deltaT_C must be > 0")
    if T_run_s is not None and T_run_s <= 0:
        raise ValueError("T_run_s must be > 0 when provided")

    G, nodes, gpn = _extract_cluster(design_bundle)

    # -- Power ledger ----------------------------------------------------
    P_gpus = G * power.P_gpu_W
    P_cpu = nodes * power.P_cpu_W_per_node
    P_other = nodes * power.P_other_W_per_node
    P_IT = P_gpus + P_cpu + P_other
    P_facility = P_IT * power.PUE
    Q_dot = P_IT   # steady-state: all IT power becomes heat [W]

    power_out = {
        "P_gpu_W": power.P_gpu_W,
        "P_cpu_W_per_node": power.P_cpu_W_per_node,
        "P_other_W_per_node": power.P_other_W_per_node,
        "PUE": power.PUE,
        "P_gpus_W": P_gpus,
        "P_cpu_W": P_cpu,
        "P_other_W": P_other,
        "P_IT_W": P_IT,
        "P_IT_kW": P_IT / 1000.0,
        "P_facility_W": P_facility,
        "P_facility_kW": P_facility / 1000.0,
    }

    # -- Cooling ---------------------------------------------------------
    if thermals.mode == "air":
        cooling_out = _cooling_air(Q_dot, thermals.air)
    else:
        cooling_out = _cooling_liquid(Q_dot, thermals.liquid)

    # -- Rack sanity (optional) ------------------------------------------
    rack_out: Optional[dict] = None
    n_racks = rack.racks
    if n_racks is None and rack.nodes_per_rack:
        n_racks = ceil(nodes / rack.nodes_per_rack)

    if n_racks:
        P_per_rack = P_IT / n_racks
        rack_out = {
            "racks": n_racks,
            "nodes_per_rack": rack.nodes_per_rack,
            "P_IT_per_rack_W": P_per_rack,
        }
        if rack.rack_power_limit_W is not None:
            rack_out["rack_power_limit_W"] = rack.rack_power_limit_W
            rack_out["rack_power_ok"] = P_per_rack <= rack.rack_power_limit_W
            rack_out["rack_power_margin_W"] = rack.rack_power_limit_W - P_per_rack

    # -- Energy over run (optional) --------------------------------------
    energy_out: Optional[dict] = None
    if T_run_s is not None:
        energy_out = {
            "T_run_s": T_run_s,
            "E_IT_kWh": (P_IT * T_run_s) / 3_600_000.0,
            "E_facility_kWh": (P_facility * T_run_s) / 3_600_000.0,
        }

    # -- Stable handoff for downstream stages ----------------------------
    handoff = {
        "cluster": {"G": G, "nodes": nodes, "gpus_per_node": gpn},
        "power": {
            "P_IT_W": P_IT,
            "P_facility_W": P_facility,
            "PUE": power.PUE,
        },
        "heat": {"Qdot_W": Q_dot},
        "cooling": {
            "mode": cooling_out["mode"],
            "mass_flow_kg_s": cooling_out["mass_flow_kg_s"],
            "vol_flow_m3_s": cooling_out["vol_flow_m3_s"],
            "deltaT_C": cooling_out["deltaT_C"],
        },
        "rack": rack_out,
        "energy": energy_out,
    }

    return {
        "cluster": {"G": G, "nodes": nodes, "gpus_per_node": gpn},
        "power": power_out,
        "heat": {"Qdot_W": Q_dot, "Qdot_kW": Q_dot / 1000.0},
        "cooling": cooling_out,
        "rack": rack_out,
        "energy": energy_out,
        "handoff": handoff,
    }
