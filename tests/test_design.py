"""Tests for Step 1 — Design."""
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
from causalcompute.core.design import run_design, _divisors, _ring_allreduce_bytes_per_rank, _inter_node_fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle0(
    P=1e9, Tok=1e11, T=86400.0,
    F_dev=1e15, B_mem=8e10,
    B_step=1e10, Tok_per_step=1e6,
    BW_fabric=1e11,
):
    return run_fundamentals(
        workload=Workload(P=P, Tok=Tok, T=T),
        state=StateBytes(),
        io=IO(),
        device=Device(F_dev_sust_flop_s=F_dev, B_dev_mem_bytes=B_mem),
        step=StepWorkingSet(B_step_bytes=B_step),
        schedule=StepSchedule(Tok_per_step=Tok_per_step),
        update=AlgorithmStepFacts(b_update_per_param=2.0),
        fabric=FabricCapability(BW_node_sust_Bps=BW_fabric),
        storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
        checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
    )


def _easy_bundle():
    """Bundle that is easy to make feasible: generous deadline, small model."""
    return _bundle0(P=1e9, Tok=1e11, T=86400.0 * 30)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_divisors_of_8(self):
        assert _divisors(8) == [1, 2, 4, 8]

    def test_divisors_of_prime(self):
        assert _divisors(7) == [1, 7]

    def test_divisors_of_1(self):
        assert _divisors(1) == [1]

    def test_ring_allreduce_single_rank(self):
        assert _ring_allreduce_bytes_per_rank(1000.0, 1) == 0.0

    def test_ring_allreduce_two_ranks(self):
        # 2*(2-1)/2 * payload = 1 * payload
        assert _ring_allreduce_bytes_per_rank(1000.0, 2) == pytest.approx(1000.0)

    def test_ring_allreduce_large_dp(self):
        # As dp → ∞, factor → 2
        result = _ring_allreduce_bytes_per_rank(1000.0, 1000)
        assert result == pytest.approx(2 * (999 / 1000) * 1000.0)

    def test_inter_node_fraction_single_node(self):
        # dp == gpus_per_node → all intra-node
        assert _inter_node_fraction(8, 8) == pytest.approx(0.0)

    def test_inter_node_fraction_clamped(self):
        assert 0.0 <= _inter_node_fraction(100, 8) <= 1.0

    def test_inter_node_fraction_dp1(self):
        assert _inter_node_fraction(1, 8) == 0.0


# ---------------------------------------------------------------------------
# Auto-size (mode A)
# ---------------------------------------------------------------------------

class TestAutoSize:
    def test_returns_feasible(self):
        b0 = _easy_bundle()
        result = run_design(b0)
        assert result["feasible"] is True

    def test_solution_has_required_keys(self):
        b0 = _easy_bundle()
        sol = run_design(b0)["solution"]
        assert "cluster" in sol
        assert "parallelism" in sol
        assert "timing" in sol
        assert "memory" in sol
        assert "communication" in sol

    def test_dp_tp_pp_multiply_to_G(self):
        b0 = _easy_bundle()
        sol = run_design(b0)["solution"]
        cl = sol["cluster"]
        par = sol["parallelism"]
        assert par["dp"] * par["tp"] * par["pp"] == cl["G"]

    def test_t_step_within_budget(self):
        b0 = _easy_bundle()
        sol = run_design(b0)["solution"]
        assert sol["timing"]["t_step_s"] <= sol["timing"]["t_step_max_s"] + 1e-9

    def test_memory_within_capacity(self):
        b0 = _easy_bundle()
        sol = run_design(b0)["solution"]
        mem = sol["memory"]
        assert mem["bytes_per_device"] <= mem["B_dev_mem_bytes"] + 1e-3

    def test_nodes_ceiling_of_G_over_gpn(self):
        b0 = _easy_bundle()
        from math import ceil
        sol = run_design(b0)["solution"]
        cl = sol["cluster"]
        assert cl["nodes"] == ceil(cl["G"] / cl["gpus_per_node"])

    def test_infeasible_when_no_memory(self):
        """Tiny memory forces infeasibility."""
        b0 = _bundle0(B_mem=1e6)  # 1 MB — impossibly small
        result = run_design(b0)
        assert result["feasible"] is False
        assert result["solution"] is None

    def test_infeasible_has_hints(self):
        b0 = _bundle0(B_mem=1e6)
        result = run_design(b0)
        assert len(result["no_solution_reason"]["hints"]) > 0


# ---------------------------------------------------------------------------
# Fixed-G (mode B)
# ---------------------------------------------------------------------------

class TestFixedG:
    def test_fixed_G_feasible(self):
        b0 = _easy_bundle()
        result = run_design(b0, G=8)
        assert result["feasible"] is True

    def test_fixed_G_uses_exact_G(self):
        b0 = _easy_bundle()
        sol = run_design(b0, G=16)["solution"]
        assert sol["cluster"]["G"] == 16

    def test_fixed_G_too_small_infeasible(self):
        # 1 GPU can't hold the state of a model that needs many
        b0 = _bundle0(P=1e12, B_mem=8e10)  # 1T params, huge state
        result = run_design(b0, G=1)
        assert result["feasible"] is False

    def test_fixed_G_invalid_raises(self):
        b0 = _easy_bundle()
        with pytest.raises(ValueError):
            run_design(b0, G=0)


# ---------------------------------------------------------------------------
# Handoff contract
# ---------------------------------------------------------------------------

class TestHandoff:
    def test_handoff_present_on_feasible(self):
        b0 = _easy_bundle()
        result = run_design(b0)
        h = result["handoff"]
        assert h["cluster"] is not None
        assert h["parallelism"] is not None
        assert h["communication"] is not None

    def test_handoff_cluster_on_infeasible_is_none(self):
        b0 = _bundle0(B_mem=1e6)
        result = run_design(b0)
        assert result["handoff"]["cluster"] is None

    def test_handoff_efficiency_always_present(self):
        b0 = _easy_bundle()
        result = run_design(b0)
        assert "eta_compute" in result["handoff"]["efficiency"]
        assert "eta_fabric" in result["handoff"]["efficiency"]


# ---------------------------------------------------------------------------
# Design input validation
# ---------------------------------------------------------------------------

class TestDesignInputValidation:
    def test_invalid_eta_compute(self):
        b0 = _easy_bundle()
        bad = DesignInputs(eta_compute=0.0)
        with pytest.raises(ValueError):
            run_design(b0, inputs=bad)

    def test_invalid_eta_fabric(self):
        b0 = _easy_bundle()
        bad = DesignInputs(eta_fabric=1.5)
        with pytest.raises(ValueError):
            run_design(b0, inputs=bad)

    def test_invalid_comm_model(self):
        b0 = _easy_bundle()
        bad = DesignInputs(comm_model="made_up_model")
        with pytest.raises(ValueError):
            run_design(b0, inputs=bad)
