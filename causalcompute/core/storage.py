"""
Step 4 — Storage.

Derives the storage cluster required to sustain:
  1. Dataset streaming BW (proven in Step 0)
  2. Checkpoint write BW (proven in Step 0)
  3. Total persistent data footprint (dataset + rolling checkpoints)

Consumes:
  bundle0  — Step 0 output (req dict: BW_dataset_plan_Bps, BW_ckpt_req_Bps,
             B_dataset_total_bytes, S_ckpt_bytes)
  design   — Step 1 output (feasible flag only; storage is independent of topology)

Returns a dict consumed by cli.py, report.py, and streamlit_app.py.
"""
from __future__ import annotations

from math import ceil
from typing import Any

from .types import StorageInputs


def _validate(s: StorageInputs) -> None:
    if s.drive_bw_seq_Bps <= 0:
        raise ValueError("drive_bw_seq_Bps must be > 0")
    if s.drive_capacity_bytes <= 0:
        raise ValueError("drive_capacity_bytes must be > 0")
    if s.drives_per_storage_node < 1:
        raise ValueError("drives_per_storage_node must be >= 1")
    if s.storage_net_bw_Bps <= 0:
        raise ValueError("storage_net_bw_Bps must be > 0")
    if s.dataset_replication < 1:
        raise ValueError("dataset_replication must be >= 1")
    if s.ckpt_replication < 1:
        raise ValueError("ckpt_replication must be >= 1")
    if s.ckpt_keep_count < 1:
        raise ValueError("ckpt_keep_count must be >= 1")


def run_storage(
    bundle0: dict[str, Any],
    design_bundle: dict[str, Any],
    *,
    storage: StorageInputs = StorageInputs(),
) -> dict[str, Any]:
    """
    Derive the storage cluster from Step 0 requirements.

    Parameters
    ----------
    bundle0 : dict
        Output of run_fundamentals().
    design_bundle : dict
        Output of run_design().  Must be feasible.
    storage : StorageInputs
        Drive / node / replication assumptions.

    Returns
    -------
    dict with keys: dataset_pool, checkpoint_pool, nodes, network, consistency, handoff
    """
    if not design_bundle.get("feasible"):
        raise ValueError("run_storage requires a feasible design (Step 1 must succeed first)")

    _validate(storage)

    req = bundle0["req"]
    BW_dataset = req["BW_dataset_plan_Bps"]        # bytes/s — sustained read BW
    BW_ckpt = req["BW_ckpt_req_Bps"]              # bytes/s — min checkpoint write BW
    B_dataset_total = req["B_dataset_total_bytes"] # bytes — full dataset footprint
    S_ckpt = req["S_ckpt_bytes"]                   # bytes — one checkpoint image

    drive_bw = storage.drive_bw_seq_Bps
    drive_cap = storage.drive_capacity_bytes
    drives_per_node = storage.drives_per_storage_node
    node_net_bw = storage.storage_net_bw_Bps

    # ------------------------------------------------------------------
    # Dataset pool
    # Drives must supply BW AND capacity (with replication).
    # ------------------------------------------------------------------
    dataset_drives_for_bw = ceil(BW_dataset / drive_bw)
    dataset_bytes_stored = B_dataset_total * storage.dataset_replication
    dataset_drives_for_cap = ceil(dataset_bytes_stored / drive_cap)
    dataset_drives = max(dataset_drives_for_bw, dataset_drives_for_cap)

    # ------------------------------------------------------------------
    # Checkpoint pool
    # Drives must supply BW AND hold keep_count × replication copies.
    # ------------------------------------------------------------------
    ckpt_drives_for_bw = ceil(BW_ckpt / drive_bw)
    ckpt_bytes_stored = S_ckpt * storage.ckpt_keep_count * storage.ckpt_replication
    ckpt_drives_for_cap = ceil(ckpt_bytes_stored / drive_cap)
    ckpt_drives = max(ckpt_drives_for_bw, ckpt_drives_for_cap)

    # ------------------------------------------------------------------
    # Storage nodes
    # A single pool of nodes serves both dataset and checkpoint traffic.
    # ------------------------------------------------------------------
    total_drives = dataset_drives + ckpt_drives
    num_storage_nodes = ceil(total_drives / drives_per_node)

    # ------------------------------------------------------------------
    # Network BW check
    # The storage nodes' aggregate network BW must cover peak demand.
    # ------------------------------------------------------------------
    storage_aggregate_net_bw = num_storage_nodes * node_net_bw
    required_net_bw = BW_dataset + BW_ckpt
    net_ok = storage_aggregate_net_bw >= required_net_bw
    net_ratio = storage_aggregate_net_bw / required_net_bw if required_net_bw > 0 else float("inf")

    # ------------------------------------------------------------------
    # Consistency check vs Step 0 checkpoint BW assumption
    # Step 0 used StorageCapability.BW_ckpt_sust_Bps as the assumed
    # checkpoint BW — Step 4 now derives the actual pool BW.
    # ------------------------------------------------------------------
    ckpt_pool_bw = ckpt_drives * drive_bw   # aggregate sequential write BW
    step0_ckpt_bw = bundle0["ckpt"]["BW_ckpt_sust_Bps"]
    ckpt_ratio = ckpt_pool_bw / step0_ckpt_bw if step0_ckpt_bw > 0 else float("inf")
    ckpt_consistent = ckpt_ratio >= 0.70
    ckpt_note = (
        None if ckpt_consistent
        else (
            f"Derived checkpoint pool BW ({ckpt_pool_bw/1e9:.1f} GB/s) is "
            f"{ckpt_ratio:.2f}× Step 0 assumption ({step0_ckpt_bw/1e9:.1f} GB/s). "
            "Increase drives_per_storage_node or drive_bw_seq_Bps."
        )
    )

    return {
        "dataset_pool": {
            "drives": dataset_drives,
            "drives_for_bw": dataset_drives_for_bw,
            "drives_for_capacity": dataset_drives_for_cap,
            "bytes_stored": dataset_bytes_stored,
            "aggregate_bw_Bps": dataset_drives * drive_bw,
            "replication": storage.dataset_replication,
        },
        "checkpoint_pool": {
            "drives": ckpt_drives,
            "drives_for_bw": ckpt_drives_for_bw,
            "drives_for_capacity": ckpt_drives_for_cap,
            "bytes_stored": ckpt_bytes_stored,
            "aggregate_bw_Bps": ckpt_drives * drive_bw,
            "keep_count": storage.ckpt_keep_count,
            "replication": storage.ckpt_replication,
        },
        "nodes": {
            "total_drives": total_drives,
            "num_storage_nodes": num_storage_nodes,
            "drives_per_node": drives_per_node,
        },
        "network": {
            "aggregate_net_bw_Bps": storage_aggregate_net_bw,
            "required_net_bw_Bps": required_net_bw,
            "net_ok": net_ok,
            "net_ratio": net_ratio,
        },
        "consistency": {
            "ckpt_pool_bw_Bps": ckpt_pool_bw,
            "step0_assumed_ckpt_bw_Bps": step0_ckpt_bw,
            "ratio": ckpt_ratio,
            "ok": ckpt_consistent,
            "note": ckpt_note,
        },
        "assumptions": {
            "drive_bw_seq_Bps": drive_bw,
            "drive_capacity_bytes": drive_cap,
            "drives_per_storage_node": drives_per_node,
            "storage_net_bw_Bps": node_net_bw,
        },
        "handoff": {
            "dataset_pool": {
                "drives": dataset_drives,
                "aggregate_bw_Bps": dataset_drives * drive_bw,
            },
            "checkpoint_pool": {
                "drives": ckpt_drives,
                "aggregate_bw_Bps": ckpt_drives * drive_bw,
            },
            "num_storage_nodes": num_storage_nodes,
        },
    }
