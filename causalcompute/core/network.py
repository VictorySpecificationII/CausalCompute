"""
Step 3 — Network.

Derives a concrete 2-tier leaf/spine fabric topology from the cluster
produced by Step 1 and a set of fabric assumptions.

Causal chain:
  Step 1 cluster (nodes, nics_per_node) → leaf count → spine count
                                        → bisection bandwidth
                                        → cable schedule
                                        → consistency check against Step 1 assumption

Two topology modes
------------------
rail_optimised = True  (HPC default)
    One NIC per GPU, each GPU's NIC connects to a dedicated leaf (rail).
    The cluster has R = nics_per_node independent rail fabrics.
    Each rail is a full leaf/spine fabric for N nodes (1 NIC each).
    Collectives stripe across all rails simultaneously.

rail_optimised = False  (standard leaf/spine)
    All NICs from a node share the same leaf domain.
    A single two-tier fabric connects all N * nics_per_node server ports.

Oversubscription
----------------
oversubscription = 1.0  →  equal server/uplink ports (full bisection)
oversubscription = 2.0  →  2× more server ports than uplink ports

The maximum number of nodes in a 2-tier design (k-port switches, os=1):
    max_nodes = (k/2)² = k²/4      (standard leaf/spine, 1 NIC/node)
    max_nodes = k²/4               (rail-opt, per rail = same)

For k=64: max_nodes per fabric = 1024. With rail-opt, that applies
per rail, so total compute nodes is still 1024 (each with 8 NICs).
If nodes > max, a 3-tier (fat-tree) is flagged but not fully designed here.

All outputs in SI units (bytes/s, counts).
"""
from __future__ import annotations

from math import ceil
from typing import Any

from .types import NetworkInputs


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_network(
    design_bundle: dict[str, Any],
    *,
    network: NetworkInputs = NetworkInputs(),
) -> dict[str, Any]:
    """
    Step 3: derive leaf/spine topology from Step 1 cluster + fabric assumptions.

    Parameters
    ----------
    design_bundle : dict
        Output of run_design(). Must be feasible.
    network : NetworkInputs
        Fabric assumptions (switch radix, port BW, oversubscription, topology).

    Returns
    -------
    dict with keys:
        topology          str
        two_tier_feasible bool
        switches          dict  — leaf/spine counts and port split
        bandwidth         dict  — per-node and bisection BW
        cables            dict  — server-to-leaf and leaf-to-spine counts
        consistency       dict  — cross-check against Step 1 fabric assumption
        handoff           dict  — stable digest for Step 4+
    """
    if not design_bundle.get("feasible", False):
        raise ValueError(
            "Design is infeasible; cannot compute network topology. "
            "Check Step 1 diagnostics."
        )

    cluster = design_bundle["handoff"]["cluster"]
    nodes: int = int(cluster["nodes"])
    step1_bw_node: float = float(
        design_bundle["diagnostics"]["physics_echo"]["BW_fabric_node_sust_Bps"]
    )

    _validate(network)

    # ------------------------------------------------------------------
    # Port split (server-facing vs uplink) based on oversubscription
    # ------------------------------------------------------------------
    # With oversubscription os:
    #   leaf_downlinks / leaf_uplinks = os
    #   leaf_downlinks + leaf_uplinks = switch_radix
    # → leaf_downlinks = switch_radix * os / (os + 1)
    k = network.switch_radix
    os = network.oversubscription
    leaf_downlinks = int(k * os / (os + 1))   # server-facing ports per leaf
    leaf_uplinks   = k - leaf_downlinks        # spine-facing ports per leaf

    # ------------------------------------------------------------------
    # Switch counts
    # ------------------------------------------------------------------
    if network.rail_optimised:
        # R independent rail fabrics, each serving N nodes with 1 NIC
        R = network.nics_per_node
        leaves_per_rail = ceil(nodes / leaf_downlinks)
        spines_per_rail = ceil(leaves_per_rail * leaf_uplinks / k)
        num_leaves = R * leaves_per_rail
        num_spines = R * spines_per_rail
        topology_label = "rail_optimised_leaf_spine"
        max_nodes_2tier = k * leaf_downlinks   # per rail
    else:
        total_server_ports = nodes * network.nics_per_node
        num_leaves = ceil(total_server_ports / leaf_downlinks)
        num_spines = ceil(num_leaves * leaf_uplinks / k)
        topology_label = "leaf_spine"
        max_nodes_2tier = k * leaf_downlinks // network.nics_per_node

    two_tier_feasible = nodes <= max_nodes_2tier

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------
    bw_per_node_raw = network.nics_per_node * network.port_bw_Bps
    bw_per_node_eff = bw_per_node_raw / os        # bisection-limited effective BW
    bisection_bw    = num_leaves * leaf_uplinks * network.port_bw_Bps

    # ------------------------------------------------------------------
    # Cable schedule
    # ------------------------------------------------------------------
    server_to_leaf_cables = nodes * network.nics_per_node
    leaf_to_spine_cables  = num_leaves * leaf_uplinks
    total_cables          = server_to_leaf_cables + leaf_to_spine_cables

    # ------------------------------------------------------------------
    # Consistency check with Step 1 fabric assumption
    # ------------------------------------------------------------------
    ratio = bw_per_node_eff / step1_bw_node if step1_bw_node > 0 else float("inf")
    # OK if Step 3 derived BW is within ±30% of Step 1 assumption,
    # or Step 3 is higher (we have more fabric than assumed — conservative).
    consistency_ok = ratio >= 0.70

    consistency = {
        "step1_assumed_bw_node_Bps": step1_bw_node,
        "step3_derived_bw_node_Bps": bw_per_node_eff,
        "ratio": ratio,
        "ok": consistency_ok,
        "note": (
            "Fabric BW consistent with Step 1 assumption."
            if consistency_ok
            else (
                f"WARNING: Step 3 derived per-node BW ({bw_per_node_eff:.2e} B/s) is "
                f"{(1 - ratio)*100:.0f}% below Step 1 assumption "
                f"({step1_bw_node:.2e} B/s). "
                "Step 1 design may be over-optimistic. "
                "Consider increasing nics_per_node, port_bw_Bps, or lowering "
                "oversubscription."
            )
        ),
    }

    handoff = {
        "nodes": nodes,
        "topology": topology_label,
        "two_tier_feasible": two_tier_feasible,
        "switches": {
            "num_leaves": num_leaves,
            "num_spines": num_spines,
            "total": num_leaves + num_spines,
            "leaf_downlinks": leaf_downlinks,
            "leaf_uplinks": leaf_uplinks,
        },
        "bandwidth": {
            "port_bw_Bps": network.port_bw_Bps,
            "bw_per_node_raw_Bps": bw_per_node_raw,
            "bw_per_node_effective_Bps": bw_per_node_eff,
            "bisection_bw_Bps": bisection_bw,
        },
        "cables": {
            "server_to_leaf": server_to_leaf_cables,
            "leaf_to_spine": leaf_to_spine_cables,
            "total": total_cables,
        },
    }

    return {
        "topology": topology_label,
        "two_tier_feasible": two_tier_feasible,
        "max_nodes_2tier": max_nodes_2tier,
        "assumptions": {
            "nics_per_node": network.nics_per_node,
            "switch_radix": k,
            "port_bw_Bps": network.port_bw_Bps,
            "oversubscription": os,
            "rail_optimised": network.rail_optimised,
        },
        "switches": {
            "num_leaves": num_leaves,
            "num_spines": num_spines,
            "total": num_leaves + num_spines,
            "leaf_downlinks": leaf_downlinks,
            "leaf_uplinks": leaf_uplinks,
        },
        "bandwidth": {
            "port_bw_Bps": network.port_bw_Bps,
            "bw_per_node_raw_Bps": bw_per_node_raw,
            "bw_per_node_effective_Bps": bw_per_node_eff,
            "bisection_bw_Bps": bisection_bw,
            "bisection_bw_Tbps": bisection_bw / 1e12,
        },
        "cables": {
            "server_to_leaf": server_to_leaf_cables,
            "leaf_to_spine": leaf_to_spine_cables,
            "total": total_cables,
        },
        "consistency": consistency,
        "handoff": handoff,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(n: NetworkInputs) -> None:
    if n.nics_per_node <= 0:
        raise ValueError("nics_per_node must be > 0")
    if n.switch_radix < 2:
        raise ValueError("switch_radix must be ≥ 2")
    if n.port_bw_Bps <= 0:
        raise ValueError("port_bw_Bps must be > 0")
    if n.oversubscription < 1.0:
        raise ValueError("oversubscription must be ≥ 1.0 (1.0 = full bisection)")
    # Sanity: with oversubscription we must have at least 1 uplink port
    k = n.switch_radix
    os = n.oversubscription
    leaf_downlinks = int(k * os / (os + 1))
    leaf_uplinks = k - leaf_downlinks
    if leaf_uplinks < 1:
        raise ValueError(
            f"switch_radix={k} with oversubscription={os} leaves no uplink ports. "
            "Reduce oversubscription or increase switch_radix."
        )
