"""Tests for bottleneck identification (core/analysis.py)."""
from __future__ import annotations

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
    Workload,
)
from causalcompute.core.fundamentals import run_fundamentals
from causalcompute.core.design import run_design
from causalcompute.core.analysis import run_bottleneck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    P=1e9, Tok=1e11, T=86400.0 * 30,
    F_dev=1e15, B_mem=8e10,
    B_step=1e10, BW_fabric=1e11,
    gpus_per_node=8, G=64,
    tp_max=8,
):
    b0 = run_fundamentals(
        workload=Workload(P=P, Tok=Tok, T=T),
        state=StateBytes(),
        io=IO(),
        device=Device(F_dev_sust_flop_s=F_dev, B_dev_mem_bytes=B_mem),
        step=StepWorkingSet(B_step_bytes=B_step),
        schedule=StepSchedule(Tok_per_step=1e6),
        update=AlgorithmStepFacts(b_update_per_param=2.0),
        fabric=FabricCapability(BW_node_sust_Bps=BW_fabric),
        storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
        checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
    )
    design = run_design(b0, G=G, inputs=DesignInputs(
        gpus_per_node=gpus_per_node, tp_max=tp_max
    ))
    return b0, design


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_runs_successfully(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert result is not None

    def test_infeasible_raises(self):
        b0, _ = _run()
        bad_design = {"feasible": False}
        with pytest.raises(ValueError):
            run_bottleneck(b0, bad_design)

    def test_required_sections_present(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        for key in ("binding", "device_bounds", "step_time", "memory",
                    "communication", "recommendations"):
            assert key in result


# ---------------------------------------------------------------------------
# Binding constraint
# ---------------------------------------------------------------------------

class TestBinding:
    def test_binding_key_is_one_of_three(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert result["binding"]["key"] in ("compute", "state", "instant")

    def test_binding_label_is_string(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert isinstance(result["binding"]["label"], str)
        assert len(result["binding"]["label"]) > 0

    def test_exactly_one_bound_is_binding(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        db = result["device_bounds"]
        binding_count = sum(
            1 for k in ("compute", "state", "instant")
            if db[k]["is_binding"]
        )
        assert binding_count == 1

    def test_binding_bound_has_lowest_headroom(self):
        """The binding bound must have the smallest headroom_x."""
        b0, design = _run()
        result = run_bottleneck(b0, design)
        db = result["device_bounds"]
        binding_key = result["binding"]["key"]
        binding_hx = db[binding_key]["headroom_x"]
        for k in ("compute", "state", "instant"):
            assert db[k]["headroom_x"] >= binding_hx

    def test_memory_bound_when_step_huge(self):
        """Enormous step working set → instantaneous memory is the binding constraint."""
        # B_step = 100 GB forces N_instant >> N_compute
        b0, design = _run(B_step=1e11, G=128)
        if not design["feasible"]:
            pytest.skip("design infeasible for this parameter set")
        result = run_bottleneck(b0, design)
        assert result["binding"]["key"] == "instant"

    def test_compute_bound_when_step_tiny_and_device_slow(self):
        """Slow device + tiny step → compute lower bound dominates."""
        # Tiny B_step, very slow device → N_compute >> N_instant
        b0, design = _run(
            F_dev=1e12,    # very slow: 1 TFLOP/s
            B_mem=8e10,
            B_step=1e6,    # tiny working set
            G=512,
        )
        if not design["feasible"]:
            pytest.skip("design infeasible for this parameter set")
        result = run_bottleneck(b0, design)
        assert result["binding"]["key"] == "compute"


# ---------------------------------------------------------------------------
# Device bounds
# ---------------------------------------------------------------------------

class TestDeviceBounds:
    def test_all_three_bounds_present(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        db = result["device_bounds"]
        for k in ("compute", "state", "instant"):
            assert k in db
            assert "N" in db[k]
            assert "headroom_x" in db[k]
            assert "is_binding" in db[k]

    def test_G_actual_matches_design(self):
        b0, design = _run(G=64)
        result = run_bottleneck(b0, design)
        assert result["device_bounds"]["G_actual"] == design["solution"]["cluster"]["G"]

    def test_N_min_matches_bundle0(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert result["device_bounds"]["N_min"] == pytest.approx(
            b0["device_bounds"]["N_min_lower_bound"]
        )

    def test_headroom_x_always_at_least_1(self):
        """G >= N_min always (design guarantees it), so headroom >= 1× for binding."""
        b0, design = _run()
        result = run_bottleneck(b0, design)
        db = result["device_bounds"]
        binding_key = result["binding"]["key"]
        assert db[binding_key]["headroom_x"] >= 1.0


# ---------------------------------------------------------------------------
# Step time
# ---------------------------------------------------------------------------

class TestStepTime:
    def test_step_time_fields_present(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        st = result["step_time"]
        for f in ("t_step_s", "t_step_max_s", "t_compute_s", "t_comm_s",
                  "headroom_s", "headroom_pct", "compute_pct", "comm_pct",
                  "step_time_tight"):
            assert f in st

    def test_headroom_pct_non_negative(self):
        """A feasible design must have headroom >= 0."""
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert result["step_time"]["headroom_pct"] >= 0.0

    def test_compute_and_comm_pct_sum_leq_100(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        st = result["step_time"]
        assert st["compute_pct"] + st["comm_pct"] <= 100.0 + 1e-6

    def test_step_time_tight_flag_type(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert isinstance(result["step_time"]["step_time_tight"], bool)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_fields_present(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        mem = result["memory"]
        for f in ("bytes_per_device", "B_dev_mem_bytes", "utilization_pct",
                  "headroom_bytes", "memory_tight"):
            assert f in mem

    def test_utilization_between_0_and_100(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        util = result["memory"]["utilization_pct"]
        assert 0.0 <= util <= 100.0

    def test_memory_tight_flag_type(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert isinstance(result["memory"]["memory_tight"], bool)

    def test_headroom_bytes_non_negative_for_feasible(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert result["memory"]["headroom_bytes"] >= 0.0


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------

class TestCommunication:
    def test_comm_fields_present(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        comm = result["communication"]
        for f in ("t_comm_s", "comm_pct", "comm_heavy"):
            assert f in comm

    def test_comm_heavy_flag_type(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert isinstance(result["communication"]["comm_heavy"], bool)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_recommendations_is_list(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert isinstance(result["recommendations"], list)

    def test_at_least_one_recommendation(self):
        """There should always be at least the binding constraint recommendation."""
        b0, design = _run()
        result = run_bottleneck(b0, design)
        assert len(result["recommendations"]) >= 1

    def test_recommendations_are_strings(self):
        b0, design = _run()
        result = run_bottleneck(b0, design)
        for r in result["recommendations"]:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_memory_recommendation_when_instant_binding(self):
        b0, design = _run(B_step=1e11, G=128)
        if not design["feasible"]:
            pytest.skip("infeasible")
        result = run_bottleneck(b0, design)
        if result["binding"]["key"] == "instant":
            text = " ".join(result["recommendations"])
            assert "Memory" in text or "memory" in text
