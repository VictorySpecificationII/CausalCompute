"""Tests for Step 4 — Storage pool sizing."""
from __future__ import annotations

from math import ceil

import pytest

from causalcompute.core.types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
    DesignInputs,
    Device,
    FabricCapability,
    IO,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    StorageInputs,
    Workload,
)
from causalcompute.core.fundamentals import run_fundamentals
from causalcompute.core.design import run_design
from causalcompute.core.storage import run_storage, _validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle(nodes_target: int = 16):
    """Return (bundle0, design) for a feasible cluster sized to ~nodes_target nodes."""
    b0 = run_fundamentals(
        workload=Workload(P=1e9, Tok=1e11, T=86400.0 * 30),
        state=StateBytes(),
        io=IO(),
        device=Device(F_dev_sust_flop_s=1e15, B_dev_mem_bytes=8e10),
        step=StepWorkingSet(B_step_bytes=1e10),
        schedule=StepSchedule(Tok_per_step=1e6),
        update=AlgorithmStepFacts(b_update_per_param=2.0),
        fabric=FabricCapability(BW_node_sust_Bps=1e11),
        storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
        checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
    )
    design = run_design(b0, G=nodes_target * 8,
                        inputs=DesignInputs(gpus_per_node=8))
    return b0, design


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_drive_bw(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(drive_bw_seq_Bps=0))

    def test_invalid_drive_capacity(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(drive_capacity_bytes=-1))

    def test_invalid_drives_per_node(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(drives_per_storage_node=0))

    def test_invalid_storage_net_bw(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(storage_net_bw_Bps=0))

    def test_invalid_dataset_replication(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(dataset_replication=0))

    def test_invalid_ckpt_replication(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(ckpt_replication=0))

    def test_invalid_ckpt_keep_count(self):
        b0, design = _bundle()
        with pytest.raises(ValueError):
            run_storage(b0, design, storage=StorageInputs(ckpt_keep_count=0))

    def test_infeasible_design_raises(self):
        b0, _ = _bundle()
        bad_design = {"feasible": False}
        with pytest.raises(ValueError):
            run_storage(b0, bad_design)


# ---------------------------------------------------------------------------
# Dataset pool
# ---------------------------------------------------------------------------

class TestDatasetPool:
    def test_drives_cover_bw(self):
        """Dataset drives must supply at least the required BW."""
        b0, design = _bundle()
        storage = StorageInputs(drive_bw_seq_Bps=7e9)
        result = run_storage(b0, design, storage=storage)
        dp = result["dataset_pool"]
        assert dp["drives"] * storage.drive_bw_seq_Bps >= b0["req"]["BW_dataset_plan_Bps"]

    def test_drives_cover_capacity(self):
        """Dataset drives must store the full dataset (with replication)."""
        b0, design = _bundle()
        storage = StorageInputs(dataset_replication=2)
        result = run_storage(b0, design, storage=storage)
        dp = result["dataset_pool"]
        expected_bytes = b0["req"]["B_dataset_total_bytes"] * 2
        assert dp["drives"] * storage.drive_capacity_bytes >= expected_bytes

    def test_replication_field_present(self):
        b0, design = _bundle()
        result = run_storage(b0, design, storage=StorageInputs(dataset_replication=3))
        assert result["dataset_pool"]["replication"] == 3

    def test_bytes_stored_includes_replication(self):
        b0, design = _bundle()
        rep = 2
        result = run_storage(b0, design, storage=StorageInputs(dataset_replication=rep))
        expected = b0["req"]["B_dataset_total_bytes"] * rep
        assert result["dataset_pool"]["bytes_stored"] == pytest.approx(expected)

    def test_more_replication_more_drives(self):
        b0, design = _bundle()
        r1 = run_storage(b0, design, storage=StorageInputs(dataset_replication=1))
        r2 = run_storage(b0, design, storage=StorageInputs(dataset_replication=3))
        assert r2["dataset_pool"]["drives"] >= r1["dataset_pool"]["drives"]


# ---------------------------------------------------------------------------
# Checkpoint pool
# ---------------------------------------------------------------------------

class TestCheckpointPool:
    def test_drives_cover_bw(self):
        b0, design = _bundle()
        storage = StorageInputs(drive_bw_seq_Bps=7e9)
        result = run_storage(b0, design, storage=storage)
        cp = result["checkpoint_pool"]
        assert cp["drives"] * storage.drive_bw_seq_Bps >= b0["req"]["BW_ckpt_req_Bps"]

    def test_drives_cover_capacity(self):
        b0, design = _bundle()
        storage = StorageInputs(ckpt_replication=2, ckpt_keep_count=3)
        result = run_storage(b0, design, storage=storage)
        cp = result["checkpoint_pool"]
        expected = b0["req"]["S_ckpt_bytes"] * 2 * 3
        assert cp["drives"] * storage.drive_capacity_bytes >= expected

    def test_bytes_stored_formula(self):
        b0, design = _bundle()
        rep, keep = 2, 4
        result = run_storage(b0, design, storage=StorageInputs(ckpt_replication=rep, ckpt_keep_count=keep))
        expected = b0["req"]["S_ckpt_bytes"] * rep * keep
        assert result["checkpoint_pool"]["bytes_stored"] == pytest.approx(expected)

    def test_keep_count_field_present(self):
        b0, design = _bundle()
        result = run_storage(b0, design, storage=StorageInputs(ckpt_keep_count=5))
        assert result["checkpoint_pool"]["keep_count"] == 5

    def test_more_generations_more_drives(self):
        b0, design = _bundle()
        r1 = run_storage(b0, design, storage=StorageInputs(ckpt_keep_count=1))
        r5 = run_storage(b0, design, storage=StorageInputs(ckpt_keep_count=5))
        assert r5["checkpoint_pool"]["drives"] >= r1["checkpoint_pool"]["drives"]


# ---------------------------------------------------------------------------
# Storage nodes
# ---------------------------------------------------------------------------

class TestStorageNodes:
    def test_total_drives_sum(self):
        b0, design = _bundle()
        result = run_storage(b0, design)
        nd = result["nodes"]
        dp = result["dataset_pool"]
        cp = result["checkpoint_pool"]
        assert nd["total_drives"] == dp["drives"] + cp["drives"]

    def test_num_nodes_covers_drives(self):
        b0, design = _bundle()
        result = run_storage(b0, design)
        nd = result["nodes"]
        assert nd["num_storage_nodes"] * nd["drives_per_node"] >= nd["total_drives"]

    def test_drives_per_node_field(self):
        b0, design = _bundle()
        dpn = 12
        result = run_storage(b0, design, storage=StorageInputs(drives_per_storage_node=dpn))
        assert result["nodes"]["drives_per_node"] == dpn

    def test_fewer_drives_per_node_means_more_nodes(self):
        b0, design = _bundle()
        r_large = run_storage(b0, design, storage=StorageInputs(drives_per_storage_node=24))
        r_small = run_storage(b0, design, storage=StorageInputs(drives_per_storage_node=4))
        assert r_small["nodes"]["num_storage_nodes"] >= r_large["nodes"]["num_storage_nodes"]


# ---------------------------------------------------------------------------
# Network BW check
# ---------------------------------------------------------------------------

class TestNetworkBW:
    def test_sufficient_nodes_gives_net_ok(self):
        """With very high per-node BW, network check passes."""
        b0, design = _bundle()
        result = run_storage(b0, design, storage=StorageInputs(storage_net_bw_Bps=1e13))
        assert result["network"]["net_ok"] is True

    def test_tiny_net_bw_fails(self):
        """1 byte/s per storage node → net_ok must be False."""
        b0, design = _bundle()
        result = run_storage(b0, design, storage=StorageInputs(storage_net_bw_Bps=1.0))
        assert result["network"]["net_ok"] is False

    def test_net_ratio_field_present(self):
        b0, design = _bundle()
        result = run_storage(b0, design)
        assert "net_ratio" in result["network"]

    def test_aggregate_bw_equals_nodes_times_node_bw(self):
        b0, design = _bundle()
        bw = 2.5e10
        result = run_storage(b0, design, storage=StorageInputs(storage_net_bw_Bps=bw))
        nw = result["network"]
        nd = result["nodes"]
        assert nw["aggregate_net_bw_Bps"] == pytest.approx(nd["num_storage_nodes"] * bw)


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_fast_drives_pass_consistency(self):
        """If pool BW >> Step 0 assumed BW, check passes."""
        b0, design = _bundle()
        result = run_storage(b0, design, storage=StorageInputs(
            drive_bw_seq_Bps=1e11,  # 100 GB/s per drive — enormous
        ))
        assert result["consistency"]["ok"] is True

    def test_slow_drives_fail_consistency(self):
        """If pool BW << Step 0 assumed BW, check fails."""
        b0, design = _bundle()
        # Step 0 assumed 5 GB/s; give it 1 byte/s drives and many drives needed anyway
        result = run_storage(b0, design, storage=StorageInputs(
            drive_bw_seq_Bps=1.0,   # absurdly slow
        ))
        assert result["consistency"]["ok"] is False

    def test_ratio_field_present(self):
        b0, design = _bundle()
        result = run_storage(b0, design)
        assert "ratio" in result["consistency"]

    def test_handoff_present(self):
        b0, design = _bundle()
        result = run_storage(b0, design)
        h = result["handoff"]
        assert "dataset_pool" in h
        assert "checkpoint_pool" in h
        assert "num_storage_nodes" in h
