"""
Step 5 — Bill of Materials (BoM).

Aggregates all proven component counts from prior steps into a single
procurement list, and runs two physics-based consistency checks against
the compute node specification.

Consumes:
  design   — Step 1 (compute cluster)
  thermals — Step 2 (rack count, optional)
  network  — Step 3 (switches, cables, NICs, optional)
  storage  — Step 4 (storage nodes, drives, optional)
  node_spec — ComputeNodeSpec (intra-node BW, GPUDirect, CPU, DRAM)

Consistency checks (post-hoc warnings, do not alter prior step math):
  1. NVLink: if intra_node_bw_per_gpu < BW_fabric_node AND tp > 1
             → TP comm is bottlenecked by intra-node link, not the fabric
  2. GPUDirect: if gpudirect_rdma=False
             → fabric BW assumption in Steps 0/3 may not be achievable;
               effective BW capped by CPU DRAM BW

Returns a dict with sections: compute, network, storage, facility, totals,
node_spec, warnings.
"""
from __future__ import annotations

from typing import Any, Optional

from .types import ComputeNodeSpec

# Approximate CPU DRAM BW used in GPUDirect warning message [bytes/s]
# (dual-socket server with DDR5-4800: ~8 channels × 38.4 GB/s ≈ 300 GB/s)
_DRAM_BW_APPROX_Bps = 3.0e11


def run_bom(
    design: dict[str, Any],
    thermals: Optional[dict[str, Any]] = None,
    network: Optional[dict[str, Any]] = None,
    storage: Optional[dict[str, Any]] = None,
    *,
    bundle0: Optional[dict[str, Any]] = None,
    node_spec: ComputeNodeSpec = ComputeNodeSpec(),
) -> dict[str, Any]:
    """
    Aggregate all proven component counts into a Bill of Materials.

    Parameters
    ----------
    design : dict
        Output of run_design().  Must be feasible.
    thermals : dict, optional
        Output of run_thermals().
    network : dict, optional
        Output of run_network().
    storage : dict, optional
        Output of run_storage().
    bundle0 : dict, optional
        Output of run_fundamentals().  Used for NVLink consistency check
        (provides the per-node fabric BW assumption from Step 0).
    node_spec : ComputeNodeSpec
        Physical compute node specification.

    Returns
    -------
    dict with sections: compute, network, storage, facility, totals,
    node_spec, warnings
    """
    if not design.get("feasible"):
        raise ValueError("run_bom requires a feasible design (Step 1 must succeed first)")

    sol = design["solution"]
    cl = sol["cluster"]
    par = sol["parallelism"]

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    gpus = cl["G"]
    nodes = cl["nodes"]
    gpus_per_node = cl["gpus_per_node"]

    compute = {
        "gpus": gpus,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "cpu_cores": nodes * node_spec.cpu_cores_per_node,
        "dram_bytes": nodes * node_spec.dram_bytes_per_node,
    }

    # ------------------------------------------------------------------
    # Network  (Step 3)
    # ------------------------------------------------------------------
    network_bom = None
    if network is not None:
        sw = network["switches"]
        ca = network["cables"]
        assum = network["assumptions"]
        nics = nodes * assum["nics_per_node"]
        network_bom = {
            "nics": nics,
            "nics_per_node": assum["nics_per_node"],
            "leaf_switches": sw["num_leaves"],
            "spine_switches": sw["num_spines"],
            "total_switches": sw["total"],
            "cables_server_to_leaf": ca["server_to_leaf"],
            "cables_leaf_to_spine": ca["leaf_to_spine"],
            "total_cables": ca["total"],
        }

    # ------------------------------------------------------------------
    # Storage  (Step 4)
    # ------------------------------------------------------------------
    storage_bom = None
    if storage is not None:
        nd = storage["nodes"]
        dp = storage["dataset_pool"]
        cp = storage["checkpoint_pool"]
        storage_bom = {
            "storage_nodes": nd["num_storage_nodes"],
            "drives_per_storage_node": nd["drives_per_node"],
            "total_drives": nd["total_drives"],
            "dataset_drives": dp["drives"],
            "checkpoint_drives": cp["drives"],
        }

    # ------------------------------------------------------------------
    # Facility  (Step 2 rack output, optional)
    # ------------------------------------------------------------------
    facility_bom = None
    if thermals is not None:
        h = thermals.get("handoff", {})
        rk = h.get("rack")
        pwr = h.get("power", {})
        facility_bom = {
            "racks": rk["racks"] if rk else None,
            "nodes_per_rack": rk["nodes_per_rack"] if rk else None,
            "P_IT_W": pwr.get("P_IT_W"),
            "P_facility_W": pwr.get("P_facility_W"),
        }

    # ------------------------------------------------------------------
    # Totals  — headline "what do I order?" row
    # ------------------------------------------------------------------
    total_compute_nodes = nodes
    total_storage_nodes = storage_bom["storage_nodes"] if storage_bom else 0
    total_all_nodes = total_compute_nodes + total_storage_nodes

    totals = {
        "compute_nodes": total_compute_nodes,
        "storage_nodes": total_storage_nodes,
        "all_nodes": total_all_nodes,
        "gpus": gpus,
        "cpu_cores": compute["cpu_cores"],
        "dram_bytes": compute["dram_bytes"],
        "switches": network_bom["total_switches"] if network_bom else None,
        "nics": network_bom["nics"] if network_bom else None,
        "cables": network_bom["total_cables"] if network_bom else None,
        "drives": storage_bom["total_drives"] if storage_bom else None,
        "racks": facility_bom["racks"] if facility_bom else None,
    }

    # ------------------------------------------------------------------
    # Node spec summary (for report / UI)
    # ------------------------------------------------------------------
    node_spec_out = {
        "intra_node_bw_Bps": node_spec.intra_node_bw_Bps,
        "intra_node_bw_per_gpu_Bps": node_spec.intra_node_bw_Bps / gpus_per_node,
        "gpudirect_rdma": node_spec.gpudirect_rdma,
        "cpu_cores_per_node": node_spec.cpu_cores_per_node,
        "dram_bytes_per_node": node_spec.dram_bytes_per_node,
        "root_complex": node_spec.root_complex,
    }

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    warnings: list[str] = []

    # 1. NVLink / intra-node BW vs inter-node fabric BW
    #    Step 1 assumes TP communication is fast (NVLink >> fabric).
    #    If intra-node BW per GPU < fabric BW per node, that assumption breaks.
    tp = par["tp"]
    intra_per_gpu = node_spec.intra_node_bw_Bps / gpus_per_node

    # Pull fabric BW: Step 0 meta is authoritative; fall back to Step 3 effective BW
    bw_fabric_node: Optional[float] = None
    if bundle0 is not None:
        bw_fabric_node = bundle0.get("meta", {}).get("BW_fabric_node_sust_Bps")
    if bw_fabric_node is None and network is not None:
        bw_fabric_node = network["bandwidth"]["bw_per_node_effective_Bps"]

    nvlink_ok = True
    if tp > 1 and bw_fabric_node is not None:
        if intra_per_gpu < bw_fabric_node:
            nvlink_ok = False
            warnings.append(
                f"NVLink bottleneck: intra-node BW per GPU "
                f"({intra_per_gpu/1e9:.0f} GB/s) < fabric BW per node "
                f"({bw_fabric_node/1e9:.0f} GB/s) but tp={tp}. "
                f"TP communication will be slower than Step 1 assumed. "
                f"Consider a higher-bandwidth intra-node interconnect (NVLink4 = 900 GB/s node)."
            )

    # 2. GPUDirect RDMA
    gpudirect_ok = node_spec.gpudirect_rdma
    if not gpudirect_ok:
        warnings.append(
            f"GPUDirect RDMA is disabled. All collective traffic must bounce "
            f"through CPU DRAM (~{_DRAM_BW_APPROX_Bps/1e9:.0f} GB/s), which may "
            f"cap effective fabric BW well below the NIC line rate assumed in "
            f"Step 0/3. Enable GPUDirect RDMA or revise BW_node_sust_Bps."
        )

    return {
        "compute": compute,
        "network": network_bom,
        "storage": storage_bom,
        "facility": facility_bom,
        "totals": totals,
        "node_spec": node_spec_out,
        "warnings": warnings,
        "checks": {
            "nvlink_ok": nvlink_ok,
            "gpudirect_ok": gpudirect_ok,
        },
    }
