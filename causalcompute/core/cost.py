"""
Step 6 — Cost Model.

Pure arithmetic on top of the BoM (Step 5) and thermals (Step 2).
No new physics.  Produces:

  capex       — itemised CapEx by component type
  opex        — energy cost for the training run
  run_cost    — what this specific run costs (amortised CapEx + OpEx)
  cost_per_token — the number that drives LLM economics decisions

Two views of run cost are provided:
  amortised   — CapEx is shared across the cluster's lifetime; this run
                pays its proportional share (T_run / amortisation_period)
  full_capex  — the cluster is dedicated to this run; full CapEx allocated

Consumes:
  bom      — Step 5 (component counts)
  thermals — Step 2 (energy over run, optional)
  bundle0  — Step 0 (Tok, T_run_s for cost-per-token, optional)
"""
from __future__ import annotations

from typing import Any, Optional

from .types import CostInputs

_SECONDS_PER_YEAR = 365.25 * 86400.0


def _validate(c: CostInputs) -> None:
    for field, val in [
        ("gpu_unit_cost", c.gpu_unit_cost),
        ("node_chassis_cost", c.node_chassis_cost),
        ("nic_unit_cost", c.nic_unit_cost),
        ("switch_unit_cost", c.switch_unit_cost),
        ("cable_unit_cost", c.cable_unit_cost),
        ("storage_node_cost", c.storage_node_cost),
        ("drive_unit_cost", c.drive_unit_cost),
        ("rack_unit_cost", c.rack_unit_cost),
        ("electricity_usd_kwh", c.electricity_usd_kwh),
    ]:
        if val < 0:
            raise ValueError(f"{field} must be >= 0, got {val}")
    if c.capex_amortization_years <= 0:
        raise ValueError(f"capex_amortization_years must be > 0, got {c.capex_amortization_years}")


def run_cost(
    bom: dict[str, Any],
    thermals: Optional[dict[str, Any]] = None,
    *,
    bundle0: Optional[dict[str, Any]] = None,
    cost: CostInputs = CostInputs(),
) -> dict[str, Any]:
    """
    Derive the full cost model from the BoM and prior step results.

    Parameters
    ----------
    bom : dict
        Output of run_bom().
    thermals : dict, optional
        Output of run_thermals().  Used for energy cost.
        If None, OpEx is estimated from BoM facility power.
    bundle0 : dict, optional
        Output of run_fundamentals().  Used for Tok and T_run_s.
        If None, cost_per_token is not computed.
    cost : CostInputs
        Unit costs and financial assumptions.

    Returns
    -------
    dict with sections: capex, opex, run_cost, cost_per_token, assumptions
    """
    _validate(cost)

    t = bom["totals"]
    nw = bom["network"]
    sb = bom["storage"]
    fac = bom["facility"]

    # ------------------------------------------------------------------
    # CapEx — itemised by component type
    # ------------------------------------------------------------------
    capex_gpus          = t["gpus"] * cost.gpu_unit_cost
    capex_chassis       = t["compute_nodes"] * cost.node_chassis_cost
    capex_nics          = (t["nics"] or 0) * cost.nic_unit_cost
    capex_switches      = (t["switches"] or 0) * cost.switch_unit_cost
    capex_cables        = (t["cables"] or 0) * cost.cable_unit_cost
    capex_storage_nodes = t["storage_nodes"] * cost.storage_node_cost
    capex_drives        = (t["drives"] or 0) * cost.drive_unit_cost
    capex_racks         = (t["racks"] or 0) * cost.rack_unit_cost

    capex_total = (
        capex_gpus + capex_chassis + capex_nics + capex_switches +
        capex_cables + capex_storage_nodes + capex_drives + capex_racks
    )

    capex = {
        "gpus":          capex_gpus,
        "node_chassis":  capex_chassis,
        "nics":          capex_nics,
        "switches":      capex_switches,
        "cables":        capex_cables,
        "storage_nodes": capex_storage_nodes,
        "drives":        capex_drives,
        "racks":         capex_racks,
        "total":         capex_total,
    }

    # ------------------------------------------------------------------
    # OpEx — energy for the training run
    # ------------------------------------------------------------------
    energy_kwh: Optional[float] = None
    T_run_s: Optional[float] = None

    if thermals is not None:
        h = thermals.get("handoff", {})
        en = h.get("energy")
        if en:
            energy_kwh = en.get("E_facility_kWh")
            T_run_s = en.get("T_run_s")

    # Fall back: estimate from facility power and bundle0 T
    if energy_kwh is None and bundle0 is not None and thermals is not None:
        h = thermals.get("handoff", {})
        pwr = h.get("power", {})
        P_facility_W = pwr.get("P_facility_W")
        T_run_s = bundle0["movement"]["t_step_max_s"] * bundle0["stepfacts"]["N_steps"]
        if P_facility_W and T_run_s:
            energy_kwh = P_facility_W * T_run_s / 3600.0 / 1000.0

    energy_cost = energy_kwh * cost.electricity_usd_kwh if energy_kwh is not None else None

    opex = {
        "energy_kwh":  energy_kwh,
        "energy_cost": energy_cost,
        "note": None if energy_kwh is not None else "thermals result required for energy cost",
    }

    # ------------------------------------------------------------------
    # Run cost — what this training run costs
    # ------------------------------------------------------------------

    # Amortised: CapEx × fraction of amortisation period consumed by this run
    amortisation_s = cost.capex_amortization_years * _SECONDS_PER_YEAR

    # Get T_run_s from bundle0 if not already set from thermals
    if T_run_s is None and bundle0 is not None:
        T_run_s = bundle0["movement"]["t_step_max_s"] * bundle0["stepfacts"]["N_steps"]

    capex_amortised_for_run: Optional[float] = None
    total_amortised: Optional[float] = None
    total_full_capex: Optional[float] = None

    if T_run_s is not None:
        capex_amortised_for_run = capex_total * (T_run_s / amortisation_s)
        opex_run = energy_cost or 0.0
        total_amortised = capex_amortised_for_run + opex_run
        total_full_capex = capex_total + opex_run

    run_cost_out = {
        "capex_amortised_for_run": capex_amortised_for_run,
        "opex_run":                energy_cost,
        "total_amortised":         total_amortised,
        "total_full_capex":        total_full_capex,
        "note": None if T_run_s is not None else "bundle0 required for run cost",
    }

    # ------------------------------------------------------------------
    # Cost per token
    # ------------------------------------------------------------------
    Tok: Optional[float] = None
    if bundle0 is not None:
        Tok = bundle0["stepfacts"]["N_steps"] * bundle0["movement"]["Tok_per_step"]

    cost_per_token_amortised: Optional[float] = None
    cost_per_token_full_capex: Optional[float] = None

    if Tok and Tok > 0 and total_amortised is not None:
        cost_per_token_amortised = total_amortised / Tok
    if Tok and Tok > 0 and total_full_capex is not None:
        cost_per_token_full_capex = total_full_capex / Tok

    cost_per_token = {
        "Tok":                    Tok,
        "amortised":              cost_per_token_amortised,
        "full_capex":             cost_per_token_full_capex,
        "note": None if Tok is not None else "bundle0 required for cost per token",
    }

    return {
        "capex":          capex,
        "opex":           opex,
        "run_cost":       run_cost_out,
        "cost_per_token": cost_per_token,
        "assumptions": {
            "amortization_years":  cost.capex_amortization_years,
            "electricity_usd_kwh": cost.electricity_usd_kwh,
            "T_run_s":             T_run_s,
        },
    }
