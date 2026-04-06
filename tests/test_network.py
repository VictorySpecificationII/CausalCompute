"""Tests for Step 3 — Network (leaf/spine topology)."""
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
    NetworkInputs,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    Workload,
)
from causalcompute.core.fundamentals import run_fundamentals
from causalcompute.core.design import run_design
from causalcompute.core.network import run_network, _validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feasible_design(nodes_target: int = 16):
    """Return a feasible Step-1 bundle sized to roughly nodes_target nodes."""
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
    return run_design(b0, G=nodes_target * 8,
                      inputs=DesignInputs(gpus_per_node=8))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_nics_per_node(self):
        design = _feasible_design()
        with pytest.raises(ValueError):
            run_network(design, network=NetworkInputs(nics_per_node=0))

    def test_invalid_switch_radix(self):
        design = _feasible_design()
        with pytest.raises(ValueError):
            run_network(design, network=NetworkInputs(switch_radix=1))

    def test_invalid_port_bw(self):
        design = _feasible_design()
        with pytest.raises(ValueError):
            run_network(design, network=NetworkInputs(port_bw_Bps=0.0))

    def test_oversubscription_below_1(self):
        design = _feasible_design()
        with pytest.raises(ValueError):
            run_network(design, network=NetworkInputs(oversubscription=0.5))

    def test_infeasible_design_raises(self):
        bad = {"feasible": False, "handoff": {"cluster": None}}
        with pytest.raises(ValueError):
            run_network(bad)


# ---------------------------------------------------------------------------
# Port split
# ---------------------------------------------------------------------------

class TestPortSplit:
    def test_full_bisection_equal_split(self):
        """os=1.0 → leaf_downlinks = leaf_uplinks = radix/2."""
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(
            switch_radix=64, oversubscription=1.0
        ))
        sw = result["switches"]
        assert sw["leaf_downlinks"] == 32
        assert sw["leaf_uplinks"] == 32

    def test_2to1_oversubscription_more_downlinks(self):
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(
            switch_radix=64, oversubscription=2.0
        ))
        sw = result["switches"]
        assert sw["leaf_downlinks"] > sw["leaf_uplinks"]
        assert sw["leaf_downlinks"] + sw["leaf_uplinks"] == 64

    def test_uplinks_plus_downlinks_equals_radix(self):
        for os in [1.0, 1.5, 2.0, 4.0]:
            design = _feasible_design(4)
            result = run_network(design, network=NetworkInputs(
                switch_radix=64, oversubscription=os
            ))
            sw = result["switches"]
            assert sw["leaf_downlinks"] + sw["leaf_uplinks"] == 64


# ---------------------------------------------------------------------------
# Rail-optimised topology
# ---------------------------------------------------------------------------

class TestRailOptimised:
    def test_leaf_count_multiple_of_nics_per_node(self):
        """With rail-opt, leaves must be a multiple of nics_per_node."""
        design = _feasible_design(8)
        nics = 8
        result = run_network(design, network=NetworkInputs(
            nics_per_node=nics, switch_radix=64, oversubscription=1.0,
            rail_optimised=True,
        ))
        assert result["switches"]["num_leaves"] % nics == 0

    def test_spine_count_multiple_of_nics_per_node(self):
        design = _feasible_design(8)
        nics = 8
        result = run_network(design, network=NetworkInputs(
            nics_per_node=nics, switch_radix=64, oversubscription=1.0,
            rail_optimised=True,
        ))
        assert result["switches"]["num_spines"] % nics == 0

    def test_topology_label(self):
        design = _feasible_design(4)
        result = run_network(design, network=NetworkInputs(rail_optimised=True))
        assert result["topology"] == "rail_optimised_leaf_spine"

    def test_non_rail_label(self):
        design = _feasible_design(4)
        result = run_network(design, network=NetworkInputs(rail_optimised=False))
        assert result["topology"] == "leaf_spine"


# ---------------------------------------------------------------------------
# Standard leaf/spine (non-rail)
# ---------------------------------------------------------------------------

class TestStandardLeafSpine:
    def test_leaf_count_covers_all_server_ports(self):
        """num_leaves * leaf_downlinks >= nodes * nics_per_node."""
        nics = 4
        design = _feasible_design(16)
        result = run_network(design, network=NetworkInputs(
            nics_per_node=nics, switch_radix=64, oversubscription=1.0,
            rail_optimised=False,
        ))
        nodes = design["solution"]["cluster"]["nodes"]
        sw = result["switches"]
        assert sw["num_leaves"] * sw["leaf_downlinks"] >= nodes * nics

    def test_spine_count_covers_all_leaf_uplinks(self):
        """num_spines * switch_radix >= num_leaves * leaf_uplinks."""
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(
            switch_radix=64, oversubscription=1.0, rail_optimised=False,
        ))
        sw = result["switches"]
        assert sw["num_spines"] * 64 >= sw["num_leaves"] * sw["leaf_uplinks"]


# ---------------------------------------------------------------------------
# Bandwidth
# ---------------------------------------------------------------------------

class TestBandwidth:
    def test_bw_per_node_raw(self):
        design = _feasible_design(8)
        nics, bw = 8, 5e10
        result = run_network(design, network=NetworkInputs(
            nics_per_node=nics, port_bw_Bps=bw
        ))
        assert result["bandwidth"]["bw_per_node_raw_Bps"] == pytest.approx(nics * bw)

    def test_bw_per_node_effective_with_oversubscription(self):
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(
            nics_per_node=8, port_bw_Bps=5e10, oversubscription=2.0
        ))
        bw = result["bandwidth"]
        assert bw["bw_per_node_effective_Bps"] == pytest.approx(
            bw["bw_per_node_raw_Bps"] / 2.0
        )

    def test_bisection_bw_positive(self):
        design = _feasible_design(8)
        result = run_network(design)
        assert result["bandwidth"]["bisection_bw_Bps"] > 0

    def test_more_nodes_more_bisection_bw(self):
        """Larger cluster → more leaves → higher bisection bandwidth."""
        small = _feasible_design(4)
        large = _feasible_design(32)
        bw_small = run_network(small)["bandwidth"]["bisection_bw_Bps"]
        bw_large = run_network(large)["bandwidth"]["bisection_bw_Bps"]
        assert bw_large >= bw_small

    def test_higher_oversubscription_reduces_effective_bw(self):
        design = _feasible_design(8)
        bw_1to1 = run_network(design, network=NetworkInputs(oversubscription=1.0))["bandwidth"]["bw_per_node_effective_Bps"]
        bw_2to1 = run_network(design, network=NetworkInputs(oversubscription=2.0))["bandwidth"]["bw_per_node_effective_Bps"]
        assert bw_2to1 < bw_1to1


# ---------------------------------------------------------------------------
# Cable count
# ---------------------------------------------------------------------------

class TestCables:
    def test_server_to_leaf_equals_nodes_times_nics(self):
        nics = 8
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(nics_per_node=nics))
        nodes = design["solution"]["cluster"]["nodes"]
        assert result["cables"]["server_to_leaf"] == nodes * nics

    def test_leaf_to_spine_equals_leaves_times_uplinks(self):
        design = _feasible_design(8)
        result = run_network(design)
        sw = result["switches"]
        assert result["cables"]["leaf_to_spine"] == sw["num_leaves"] * sw["leaf_uplinks"]

    def test_total_cables_sum(self):
        design = _feasible_design(8)
        result = run_network(design)
        ca = result["cables"]
        assert ca["total"] == ca["server_to_leaf"] + ca["leaf_to_spine"]


# ---------------------------------------------------------------------------
# 2-tier feasibility
# ---------------------------------------------------------------------------

class TestTierFeasibility:
    def test_small_cluster_is_2tier_feasible(self):
        # 4 nodes with k=64, os=1 → max 32 nodes per rail: easily fits
        design = _feasible_design(4)
        result = run_network(design, network=NetworkInputs(switch_radix=64))
        assert result["two_tier_feasible"] is True

    def test_large_cluster_may_exceed_2tier(self):
        # k=4 → leaf_downlinks=2, max_nodes_2tier = 4*2 = 8 per rail
        # 16 nodes > 8 → 2-tier not feasible
        design = _feasible_design(16)
        result = run_network(design, network=NetworkInputs(
            switch_radix=4, nics_per_node=8, oversubscription=1.0,
            rail_optimised=True,
        ))
        assert result["two_tier_feasible"] is False


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_consistent_when_bw_matches(self):
        """If Step 1 fabric assumption matches Step 3 derived BW, check passes."""
        # BW_fabric_node_sust_Bps = 1e11 (100 GB/s)
        # Step 3: 8 NICs × 50 GB/s / os=1.0 = 400 GB/s → ratio > 1, passes
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(
            nics_per_node=8, port_bw_Bps=5e10, oversubscription=1.0
        ))
        assert result["consistency"]["ok"] is True

    def test_inconsistent_when_derived_bw_far_below_assumed(self):
        """Very slow NICs with high oversubscription → inconsistency warning."""
        design = _feasible_design(8)
        result = run_network(design, network=NetworkInputs(
            nics_per_node=1,
            port_bw_Bps=1e8,       # 100 MB/s — far below Step 1's 100 GB/s assumption
            oversubscription=4.0,
        ))
        assert result["consistency"]["ok"] is False

    def test_consistency_ratio_field_present(self):
        design = _feasible_design(8)
        result = run_network(design)
        assert "ratio" in result["consistency"]

    def test_handoff_present(self):
        design = _feasible_design(8)
        result = run_network(design)
        h = result["handoff"]
        assert "switches" in h
        assert "bandwidth" in h
        assert "cables" in h
