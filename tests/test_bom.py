"""Tests for Step 5 — Bill of Materials."""
from __future__ import annotations

import pytest

from causalcompute.core.types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
    DesignInputs,
    Device,
    FabricCapability,
    IO,
    NetworkInputs,
    PowerInputs,
    RackInputs,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    StorageInputs,
    ThermalInputs,
    Workload,
)
from causalcompute.core.fundamentals import run_fundamentals
from causalcompute.core.design import run_design
from causalcompute.core.thermals import run_thermals
from causalcompute.core.network import run_network
from causalcompute.core.storage import run_storage
from causalcompute.core.bom import run_bom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline(nodes_target: int = 8, with_racks: bool = False):
    """Return (design, thermals, network, storage) for a feasible cluster."""
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
    G = nodes_target * 8
    design = run_design(b0, G=G, inputs=DesignInputs(gpus_per_node=8))

    rack = RackInputs(nodes_per_rack=4) if with_racks else RackInputs()
    thermals = run_thermals(design, power=PowerInputs(), thermals=ThermalInputs(),
                            rack=rack, T_run_s=86400.0 * 30)
    network = run_network(design, network=NetworkInputs(nics_per_node=8))
    storage = run_storage(b0, design, storage=StorageInputs())
    return b0, design, thermals, network, storage


# ---------------------------------------------------------------------------
# Basic smoke
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_runs_with_all_steps(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result is not None

    def test_runs_with_only_design(self):
        _, design, _, _, _ = _pipeline()
        result = run_bom(design)
        assert result is not None

    def test_infeasible_design_raises(self):
        bad = {"feasible": False}
        with pytest.raises(ValueError):
            run_bom(bad)


# ---------------------------------------------------------------------------
# Compute section
# ---------------------------------------------------------------------------

class TestCompute:
    def test_gpus_matches_design(self):
        _, design, thermals, network, storage = _pipeline(8)
        result = run_bom(design, thermals, network, storage)
        assert result["compute"]["gpus"] == design["solution"]["cluster"]["G"]

    def test_nodes_matches_design(self):
        _, design, thermals, network, storage = _pipeline(8)
        result = run_bom(design, thermals, network, storage)
        assert result["compute"]["nodes"] == design["solution"]["cluster"]["nodes"]

    def test_gpus_per_node_matches_design(self):
        _, design, thermals, network, storage = _pipeline(8)
        result = run_bom(design, thermals, network, storage)
        assert result["compute"]["gpus_per_node"] == design["solution"]["cluster"]["gpus_per_node"]

    def test_gpus_equals_nodes_times_gpus_per_node(self):
        _, design, thermals, network, storage = _pipeline(16)
        result = run_bom(design, thermals, network, storage)
        c = result["compute"]
        assert c["gpus"] == c["nodes"] * c["gpus_per_node"]


# ---------------------------------------------------------------------------
# Network section
# ---------------------------------------------------------------------------

class TestNetwork:
    def test_network_section_present_when_provided(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["network"] is not None

    def test_network_section_none_when_not_provided(self):
        _, design, thermals, _, storage = _pipeline()
        result = run_bom(design, thermals, None, storage)
        assert result["network"] is None

    def test_total_switches_matches_network(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["network"]["total_switches"] == network["switches"]["total"]

    def test_leaf_plus_spine_equals_total_switches(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        nw = result["network"]
        assert nw["leaf_switches"] + nw["spine_switches"] == nw["total_switches"]

    def test_cables_server_to_leaf_matches_network(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["network"]["cables_server_to_leaf"] == network["cables"]["server_to_leaf"]

    def test_total_cables_sum(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        nw = result["network"]
        assert nw["total_cables"] == nw["cables_server_to_leaf"] + nw["cables_leaf_to_spine"]

    def test_nics_equals_nodes_times_nics_per_node(self):
        nics = 8
        _, design, thermals, _, storage = _pipeline()
        network = run_network(design, network=NetworkInputs(nics_per_node=nics))
        result = run_bom(design, thermals, network, storage)
        nodes = design["solution"]["cluster"]["nodes"]
        assert result["network"]["nics"] == nodes * nics


# ---------------------------------------------------------------------------
# Storage section
# ---------------------------------------------------------------------------

class TestStorage:
    def test_storage_section_present_when_provided(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["storage"] is not None

    def test_storage_section_none_when_not_provided(self):
        _, design, thermals, network, _ = _pipeline()
        result = run_bom(design, thermals, network, None)
        assert result["storage"] is None

    def test_total_drives_matches_storage(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["storage"]["total_drives"] == storage["nodes"]["total_drives"]

    def test_dataset_plus_checkpoint_drives_equals_total(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        sb = result["storage"]
        assert sb["dataset_drives"] + sb["checkpoint_drives"] == sb["total_drives"]

    def test_storage_nodes_matches_storage(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["storage"]["storage_nodes"] == storage["nodes"]["num_storage_nodes"]


# ---------------------------------------------------------------------------
# Facility section
# ---------------------------------------------------------------------------

class TestFacility:
    def test_facility_none_when_thermals_not_provided(self):
        _, design, _, network, storage = _pipeline()
        result = run_bom(design, None, network, storage)
        assert result["facility"] is None

    def test_facility_present_when_thermals_provided(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        assert result["facility"] is not None

    def test_rack_count_present_when_nodes_per_rack_set(self):
        _, design, thermals, network, storage = _pipeline(with_racks=True)
        result = run_bom(design, thermals, network, storage)
        assert result["facility"]["racks"] is not None

    def test_rack_count_none_without_rack_config(self):
        _, design, thermals, network, storage = _pipeline(with_racks=False)
        result = run_bom(design, thermals, network, storage)
        assert result["facility"]["racks"] is None

    def test_facility_power_fields_present(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        f = result["facility"]
        assert "P_IT_W" in f
        assert "P_facility_W" in f


# ---------------------------------------------------------------------------
# Totals section
# ---------------------------------------------------------------------------

class TestTotals:
    def test_all_nodes_equals_compute_plus_storage(self):
        _, design, thermals, network, storage = _pipeline()
        result = run_bom(design, thermals, network, storage)
        t = result["totals"]
        assert t["all_nodes"] == t["compute_nodes"] + t["storage_nodes"]

    def test_compute_nodes_matches_design(self):
        _, design, thermals, network, storage = _pipeline(16)
        result = run_bom(design, thermals, network, storage)
        assert result["totals"]["compute_nodes"] == design["solution"]["cluster"]["nodes"]

    def test_gpus_in_totals_matches_design(self):
        _, design, thermals, network, storage = _pipeline(8)
        result = run_bom(design, thermals, network, storage)
        assert result["totals"]["gpus"] == design["solution"]["cluster"]["G"]

    def test_switches_none_without_network(self):
        _, design, thermals, _, storage = _pipeline()
        result = run_bom(design, thermals, None, storage)
        assert result["totals"]["switches"] is None

    def test_drives_none_without_storage(self):
        _, design, thermals, network, _ = _pipeline()
        result = run_bom(design, thermals, network, None)
        assert result["totals"]["drives"] is None

    def test_storage_nodes_zero_without_storage(self):
        _, design, thermals, network, _ = _pipeline()
        result = run_bom(design, thermals, network, None)
        assert result["totals"]["storage_nodes"] == 0

    def test_larger_cluster_more_gpus(self):
        _, small_design, _, _, _ = _pipeline(4)
        _, large_design, _, _, _ = _pipeline(32)
        small_bom = run_bom(small_design)
        large_bom = run_bom(large_design)
        assert large_bom["totals"]["gpus"] > small_bom["totals"]["gpus"]
