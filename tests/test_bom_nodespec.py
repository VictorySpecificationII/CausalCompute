"""Tests for ComputeNodeSpec integration in Step 5 BoM."""
from __future__ import annotations

import pytest

from causalcompute.core.types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
    ComputeNodeSpec,
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
from causalcompute.core.network import run_network
from causalcompute.core.bom import run_bom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _design(nodes_target: int = 8, tp: int = 8):
    """Return a feasible design with tp fixed to the given value."""
    b0 = run_fundamentals(
        workload=Workload(P=1e9, Tok=1e11, T=86400.0 * 30),
        state=StateBytes(),
        io=IO(),
        device=Device(F_dev_sust_flop_s=1e15, B_dev_mem_bytes=8e10),
        step=StepWorkingSet(B_step_bytes=1e10),
        schedule=StepSchedule(Tok_per_step=1e6),
        update=AlgorithmStepFacts(b_update_per_param=2.0),
        fabric=FabricCapability(BW_node_sust_Bps=1e11),   # 100 GB/s
        storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
        checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
    )
    # Force a specific GPU count so we can control tp
    design = run_design(b0, G=nodes_target * 8, inputs=DesignInputs(gpus_per_node=8, tp_max=tp))
    return b0, design


# ---------------------------------------------------------------------------
# Node spec in BoM output
# ---------------------------------------------------------------------------

class TestNodeSpecOutput:
    def test_node_spec_section_present(self):
        _, design = _design()
        result = run_bom(design)
        assert "node_spec" in result

    def test_intra_node_bw_field(self):
        bw = 9.0e11
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(intra_node_bw_Bps=bw))
        assert result["node_spec"]["intra_node_bw_Bps"] == pytest.approx(bw)

    def test_intra_node_bw_per_gpu(self):
        bw = 9.0e11
        gpn = 8
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(intra_node_bw_Bps=bw))
        assert result["node_spec"]["intra_node_bw_per_gpu_Bps"] == pytest.approx(bw / gpn)

    def test_gpudirect_field(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(gpudirect_rdma=False))
        assert result["node_spec"]["gpudirect_rdma"] is False

    def test_cpu_cores_field(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(cpu_cores_per_node=64))
        assert result["node_spec"]["cpu_cores_per_node"] == 64

    def test_dram_field(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(dram_bytes_per_node=5.12e11))
        assert result["node_spec"]["dram_bytes_per_node"] == pytest.approx(5.12e11)

    def test_root_complex_field(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(root_complex="single_socket"))
        assert result["node_spec"]["root_complex"] == "single_socket"


# ---------------------------------------------------------------------------
# CPU and DRAM in totals and compute
# ---------------------------------------------------------------------------

class TestCpuDramTotals:
    def test_total_cpu_cores_equals_nodes_times_cores_per_node(self):
        cores = 64
        _, design = _design(8)
        result = run_bom(design, node_spec=ComputeNodeSpec(cpu_cores_per_node=cores))
        nodes = design["solution"]["cluster"]["nodes"]
        assert result["totals"]["cpu_cores"] == nodes * cores

    def test_total_dram_equals_nodes_times_dram_per_node(self):
        dram = 1e12
        _, design = _design(8)
        result = run_bom(design, node_spec=ComputeNodeSpec(dram_bytes_per_node=dram))
        nodes = design["solution"]["cluster"]["nodes"]
        assert result["totals"]["dram_bytes"] == pytest.approx(nodes * dram)

    def test_compute_section_has_cpu_cores(self):
        _, design = _design()
        result = run_bom(design)
        assert "cpu_cores" in result["compute"]

    def test_compute_section_has_dram_bytes(self):
        _, design = _design()
        result = run_bom(design)
        assert "dram_bytes" in result["compute"]


# ---------------------------------------------------------------------------
# NVLink consistency check
# ---------------------------------------------------------------------------

class TestNVLinkCheck:
    def test_nvlink_ok_when_intra_fast(self):
        """NVLink4 (900 GB/s node) >> fabric (100 GB/s node): no warning."""
        b0, design = _design()
        result = run_bom(design, bundle0=b0, node_spec=ComputeNodeSpec(intra_node_bw_Bps=9e11))
        # With NVLink4, check should pass regardless of tp
        assert result["checks"]["nvlink_ok"] is True

    def test_nvlink_warn_when_pcie_and_tp_gt_1(self):
        """PCIe (~128 GB/s node) < fabric (100 GB/s node) per GPU with tp > 1 → warn."""
        b0, design = _design()
        tp = design["solution"]["parallelism"]["tp"]
        if tp <= 1:
            pytest.skip("design chose tp=1; cannot test NVLink warning")

        # PCIe-level intra-node BW: 128 GB/s total / 8 GPUs = 16 GB/s per GPU
        # Fabric BW: 100 GB/s per node >> 16 GB/s per GPU → warning
        result = run_bom(design, bundle0=b0, node_spec=ComputeNodeSpec(intra_node_bw_Bps=1.28e11))
        assert result["checks"]["nvlink_ok"] is False
        assert len(result["warnings"]) > 0

    def test_no_nvlink_warn_when_tp_equals_1(self):
        """tp=1 means no intra-node TP comm: NVLink check should always pass."""
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
        # Force tp_max=1 so design is forced to tp=1
        design = run_design(b0, G=8, inputs=DesignInputs(gpus_per_node=8, tp_max=1))
        assert design["solution"]["parallelism"]["tp"] == 1
        result = run_bom(design, node_spec=ComputeNodeSpec(intra_node_bw_Bps=1.0))
        assert result["checks"]["nvlink_ok"] is True

    def test_warnings_list_present(self):
        _, design = _design()
        result = run_bom(design)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# GPUDirect RDMA check
# ---------------------------------------------------------------------------

class TestGPUDirectCheck:
    def test_gpudirect_ok_when_enabled(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(gpudirect_rdma=True))
        assert result["checks"]["gpudirect_ok"] is True
        # No GPUDirect warning should appear
        gpudirect_warns = [w for w in result["warnings"] if "GPUDirect" in w]
        assert len(gpudirect_warns) == 0

    def test_gpudirect_warn_when_disabled(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(gpudirect_rdma=False))
        assert result["checks"]["gpudirect_ok"] is False
        gpudirect_warns = [w for w in result["warnings"] if "GPUDirect" in w]
        assert len(gpudirect_warns) == 1

    def test_gpudirect_warning_mentions_dram(self):
        _, design = _design()
        result = run_bom(design, node_spec=ComputeNodeSpec(gpudirect_rdma=False))
        warn_text = " ".join(result["warnings"])
        assert "DRAM" in warn_text

    def test_both_warnings_can_fire_simultaneously(self):
        """PCIe + no GPUDirect + tp>1 → both warnings present."""
        b0, design = _design()
        tp = design["solution"]["parallelism"]["tp"]
        if tp <= 1:
            pytest.skip("design chose tp=1")
        result = run_bom(design, bundle0=b0, node_spec=ComputeNodeSpec(
            intra_node_bw_Bps=1.28e11,
            gpudirect_rdma=False,
        ))
        assert len(result["warnings"]) == 2


# ---------------------------------------------------------------------------
# Defaults (H100 SXM HGX node should pass all checks)
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_node_spec_passes_all_checks(self):
        """Default ComputeNodeSpec (NVLink4 + GPUDirect) should pass both checks."""
        b0, design = _design()
        result = run_bom(design, bundle0=b0)
        assert result["checks"]["gpudirect_ok"] is True
        # NVLink4: 900 GB/s node / 8 GPUs = 112.5 GB/s per GPU > 100 GB/s fabric → ok
        assert result["checks"]["nvlink_ok"] is True
        assert len(result["warnings"]) == 0
