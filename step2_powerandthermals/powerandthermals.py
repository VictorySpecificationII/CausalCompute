# step2_powerandthermals/powerandthermals.py
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Literal, Optional


# =============================================================================
# Step 2 — Power & Thermals (energy + flow, vendor-agnostic)
# =============================================================================
#
# Goal: keep the physics in SI.
# - Power: W (and kW as a derived convenience)
# - Heat:  W (and kW)
# - Mass flow: kg/s
# - Volumetric flow: m^3/s
#
# Non-SI display units (CFM, LPM) are optional and OFF by default.
#
# This layer consumes Step 1 "handoff" and produces a stable Step 2 "handoff"
# for downstream layers (e.g., networking, facility planning).
# =============================================================================


CoolingMode = Literal["air", "liquid"]


# =============================================================================
# Inputs
# =============================================================================

@dataclass(frozen=True)
class PowerInputs:
    """
    Power model in watts.

    Per-node "other" is everything that isn't GPU or CPU:
    - DRAM
    - NICs
    - motherboard/BMC
    - local storage
    - fans/pumps
    - margin
    """
    P_gpu_W: float = 700.0
    P_cpu_W_per_node: float = 250.0
    P_other_W_per_node: float = 300.0

    # Facility overhead
    PUE: float = 1.30  # dimensionless


@dataclass(frozen=True)
class AirCoolingInputs:
    """
    Air cooling model.
    ΔT is the allowed temperature rise across the air path (rack / room).
    """
    deltaT_C: float = 15.0

    # Properties of air (overrideable)
    rho_kg_m3: float = 1.20      # kg/m^3
    cp_J_kgK: float = 1005.0     # J/(kg*K)


@dataclass(frozen=True)
class LiquidCoolingInputs:
    """
    Liquid cooling model (water-like by default).
    ΔT is the allowed coolant temperature rise across the cold plate / CDU loop.
    """
    deltaT_C: float = 7.0

    # Properties of liquid (overrideable; water-like defaults)
    rho_kg_m3: float = 1000.0    # kg/m^3
    cp_J_kgK: float = 4186.0     # J/(kg*K)


@dataclass(frozen=True)
class ThermalInputs:
    mode: CoolingMode = "air"
    air: AirCoolingInputs = AirCoolingInputs()
    liquid: LiquidCoolingInputs = LiquidCoolingInputs()


@dataclass(frozen=True)
class RackInputs:
    """
    Optional rack constraints for a quick sanity check.

    If racks is not provided, we infer racks from nodes_per_rack if set.
    """
    nodes_per_rack: Optional[int] = None
    rack_power_limit_W: Optional[float] = None  # e.g. 30000W
    racks: Optional[int] = None                 # explicit override


# =============================================================================
# Conversions (display only; NOT part of the SI contract)
# =============================================================================

_CFM_PER_M3_S = 2118.88         # ft^3/min per (m^3/s)
_LPM_PER_M3_S = 1000.0 * 60.0   # (L/m^3) * (s/min)


# =============================================================================
# Validation
# =============================================================================

def _validate_power_inputs(p: PowerInputs) -> None:
    if p.P_gpu_W <= 0.0:
        raise ValueError("P_gpu_W must be > 0")
    if p.P_cpu_W_per_node < 0.0 or p.P_other_W_per_node < 0.0:
        raise ValueError("Per-node CPU/other power must be >= 0")
    if p.PUE < 1.0:
        raise ValueError("PUE must be >= 1.0")


def _validate_thermal_inputs(t: ThermalInputs) -> None:
    if t.mode not in ("air", "liquid"):
        raise ValueError("mode must be 'air' or 'liquid'")
    if t.air.deltaT_C <= 0.0 or t.liquid.deltaT_C <= 0.0:
        raise ValueError("deltaT_C must be > 0")
    if t.air.rho_kg_m3 <= 0.0 or t.air.cp_J_kgK <= 0.0:
        raise ValueError("air rho/cp must be > 0")
    if t.liquid.rho_kg_m3 <= 0.0 or t.liquid.cp_J_kgK <= 0.0:
        raise ValueError("liquid rho/cp must be > 0")


# =============================================================================
# Step 1 handoff extraction
# =============================================================================

def _extract_cluster(design_bundle: Dict[str, Any]) -> Dict[str, int]:
    if "handoff" not in design_bundle:
        raise KeyError("design_bundle missing 'handoff' (Step 1 handoff contract).")

    h = design_bundle["handoff"]
    cluster = h.get("cluster", None)

    if design_bundle.get("feasible", False) is not True or cluster is None:
        raise ValueError("Design is not feasible or missing cluster handoff; cannot compute power/thermals.")

    G = int(cluster["G"])
    nodes = int(cluster["nodes"])
    gpn = int(cluster["gpus_per_node"])

    if G <= 0 or nodes <= 0 or gpn <= 0:
        raise ValueError("Invalid cluster values in handoff.")

    return {"G": G, "nodes": nodes, "gpus_per_node": gpn}


# =============================================================================
# Core
# =============================================================================

def run_powerandthermals(
    design_bundle: Dict[str, Any],
    *,
    power: PowerInputs = PowerInputs(),
    thermals: ThermalInputs = ThermalInputs(),
    rack: RackInputs = RackInputs(),
    T_run_s: Optional[float] = None,
    include_display_units: bool = False,
) -> Dict[str, Any]:
    """
    Compute power + thermals for a Step-1 design output.

    SI contract:
      - Power: W (kW also provided)
      - Heat:  W (kW also provided)
      - Flows: kg/s and m^3/s

    Non-SI display units (CFM, LPM) are optional (include_display_units=True).

    Produces:
      - bundle2 with full results
      - bundle2["handoff"] as a stable digest for downstream layers
    """
    _validate_power_inputs(power)
    _validate_thermal_inputs(thermals)

    c = _extract_cluster(design_bundle)
    G = c["G"]
    nodes = c["nodes"]
    gpn = c["gpus_per_node"]

    # -------------------------------------------------------------------------
    # Power ledger (IT)
    # -------------------------------------------------------------------------
    P_gpus_W = G * power.P_gpu_W
    P_cpu_W = nodes * power.P_cpu_W_per_node
    P_other_W = nodes * power.P_other_W_per_node

    P_IT_W = P_gpus_W + P_cpu_W + P_other_W
    P_facility_W = P_IT_W * power.PUE

    # Heat (steady state): nearly all IT power becomes heat
    Qdot_W = P_IT_W  # [W]

    # -------------------------------------------------------------------------
    # Thermals: air or liquid (SI outputs)
    # -------------------------------------------------------------------------
    thermal_out: Dict[str, Any]
    handoff_thermals: Dict[str, Any]

    if thermals.mode == "air":
        a = thermals.air
        # Q = m_dot * cp * ΔT  =>  m_dot = Q / (cp*ΔT)
        m_dot_kg_s = Qdot_W / (a.cp_J_kgK * a.deltaT_C)
        # m_dot = rho * V_dot  =>  V_dot = m_dot / rho
        V_dot_m3_s = m_dot_kg_s / a.rho_kg_m3

        results: Dict[str, Any] = {
            "mass_flow_kg_s": float(m_dot_kg_s),
            "vol_flow_m3_s": float(V_dot_m3_s),
        }
        if include_display_units:
            results["vol_flow_CFM_display"] = float(V_dot_m3_s * _CFM_PER_M3_S)

        thermal_out = {
            "mode": "air",
            "assumptions": {
                "deltaT_C": float(a.deltaT_C),
                "rho_kg_m3": float(a.rho_kg_m3),
                "cp_J_kgK": float(a.cp_J_kgK),
            },
            "results": results,
        }

        handoff_thermals = {
            "mode": "air",
            "mass_flow_kg_s": float(m_dot_kg_s),
            "vol_flow_m3_s": float(V_dot_m3_s),
            "deltaT_C": float(a.deltaT_C),
        }

    else:
        l = thermals.liquid
        # Q = m_dot * cp * ΔT  =>  m_dot = Q / (cp*ΔT)
        m_dot_kg_s = Qdot_W / (l.cp_J_kgK * l.deltaT_C)
        # V_dot = m_dot / rho
        V_dot_m3_s = m_dot_kg_s / l.rho_kg_m3

        results = {
            "mass_flow_kg_s": float(m_dot_kg_s),
            "vol_flow_m3_s": float(V_dot_m3_s),
        }
        if include_display_units:
            results["vol_flow_LPM_display"] = float(V_dot_m3_s * _LPM_PER_M3_S)

        thermal_out = {
            "mode": "liquid",
            "assumptions": {
                "deltaT_C": float(l.deltaT_C),
                "rho_kg_m3": float(l.rho_kg_m3),
                "cp_J_kgK": float(l.cp_J_kgK),
            },
            "results": results,
        }

        handoff_thermals = {
            "mode": "liquid",
            "mass_flow_kg_s": float(m_dot_kg_s),
            "vol_flow_m3_s": float(V_dot_m3_s),
            "deltaT_C": float(l.deltaT_C),
        }

    # -------------------------------------------------------------------------
    # Optional rack sanity
    # -------------------------------------------------------------------------
    rack_out: Dict[str, Any] = {"enabled": False}

    racks = rack.racks
    nodes_per_rack = rack.nodes_per_rack

    if racks is None and nodes_per_rack is not None and nodes_per_rack > 0:
        racks = int(ceil(nodes / nodes_per_rack))

    if racks is not None and racks > 0:
        rack_out["enabled"] = True
        rack_out["racks"] = int(racks)
        rack_out["nodes_per_rack_assumed"] = int(nodes_per_rack) if nodes_per_rack else None

        P_IT_per_rack_W = P_IT_W / racks
        rack_out["P_IT_per_rack_W"] = float(P_IT_per_rack_W)

        if rack.rack_power_limit_W is not None:
            rack_out["rack_power_limit_W"] = float(rack.rack_power_limit_W)
            rack_out["rack_power_ok"] = bool(P_IT_per_rack_W <= rack.rack_power_limit_W)
            rack_out["rack_power_margin_W"] = float(rack.rack_power_limit_W - P_IT_per_rack_W)

    # -------------------------------------------------------------------------
    # Optional energy over run
    # -------------------------------------------------------------------------
    energy_out: Optional[Dict[str, Any]] = None
    if T_run_s is not None:
        if T_run_s <= 0.0:
            raise ValueError("T_run_s must be > 0 when provided.")
        # kWh = (W * s) / (1000 * 3600)
        E_IT_kWh = (P_IT_W * T_run_s) / (1000.0 * 3600.0)
        E_facility_kWh = (P_facility_W * T_run_s) / (1000.0 * 3600.0)
        energy_out = {
            "T_run_s": float(T_run_s),
            "E_IT_kWh": float(E_IT_kWh),
            "E_facility_kWh": float(E_facility_kWh),
        }

    # -------------------------------------------------------------------------
    # Stable handoff (digest) for downstream layers
    # -------------------------------------------------------------------------
    handoff: Dict[str, Any] = {
        "cluster": {"G": int(G), "nodes": int(nodes), "gpus_per_node": int(gpn)},
        "power": {"P_IT_W": float(P_IT_W), "P_facility_W": float(P_facility_W), "PUE": float(power.PUE)},
        "heat": {"Qdot_W": float(Qdot_W)},
        "thermals": handoff_thermals,
        "rack": None if not rack_out.get("enabled", False) else rack_out,
        "energy": energy_out,  # None if T_run_s not provided
    }

    return {
        "cluster": {
            "G": int(G),
            "nodes": int(nodes),
            "gpus_per_node": int(gpn),
        },
        "power": {
            "inputs": {
                "P_gpu_W": float(power.P_gpu_W),
                "P_cpu_W_per_node": float(power.P_cpu_W_per_node),
                "P_other_W_per_node": float(power.P_other_W_per_node),
                "PUE": float(power.PUE),
            },
            "ledger_W": {
                "P_gpus_W": float(P_gpus_W),
                "P_cpu_W": float(P_cpu_W),
                "P_other_W": float(P_other_W),
                "P_IT_W": float(P_IT_W),
                "P_facility_W": float(P_facility_W),
            },
            "ledger_kW": {
                "P_IT_kW": float(P_IT_W / 1000.0),
                "P_facility_kW": float(P_facility_W / 1000.0),
            },
        },
        "heat": {
            "Qdot_W": float(Qdot_W),
            "Qdot_kW": float(Qdot_W / 1000.0),
        },
        "thermals": thermal_out,
        "rack": rack_out,
        "energy": energy_out,
        "meta": {
            "si_contract": True,
            "display_units_included": bool(include_display_units),
        },
        "handoff": handoff,
    }

