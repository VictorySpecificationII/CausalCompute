"""Tests for ZeRO stage modeling in Step 1 (design.py)."""
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
from causalcompute.core.design import run_design, _state_bytes_per_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle0(
    P=13e9, Tok=3e12, T=86400.0 * 30,
    b_w=2.0, b_g=2.0, b_opt=8.0,
    F_dev=1e15, B_mem=8e10,
    B_step=1.5e11,
):
    return run_fundamentals(
        workload=Workload(P=P, Tok=Tok, T=T),
        state=StateBytes(b_w=b_w, b_g=b_g, b_opt=b_opt),
        io=IO(),
        device=Device(F_dev_sust_flop_s=F_dev, B_dev_mem_bytes=B_mem),
        step=StepWorkingSet(B_step_bytes=B_step),
        schedule=StepSchedule(Tok_per_step=4e7),
        update=AlgorithmStepFacts(b_update_per_param=2.0),
        fabric=FabricCapability(BW_node_sust_Bps=1e11),
        storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
        checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
    )


def _design(b0, zero_stage=0, G=None, tp_max=8, gpus_per_node=8):
    return run_design(
        b0,
        G=G,
        inputs=DesignInputs(
            gpus_per_node=gpus_per_node,
            tp_max=tp_max,
            zero_stage=zero_stage,
        ),
    )


# ---------------------------------------------------------------------------
# _state_bytes_per_device unit tests
# ---------------------------------------------------------------------------

class TestStateHelper:
    """Direct tests of the state memory formula for each ZeRO stage."""

    def _check(self, stage, dp, tp, pp, B_w, B_g, B_opt, expected):
        result = _state_bytes_per_device(B_w, B_g, B_opt, dp, tp, pp, stage)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_zero0_no_dp_sharding(self):
        """ZeRO-0: state = (B_w + B_g + B_opt) / (tp × pp)."""
        self._check(0, dp=4, tp=2, pp=2, B_w=4.0, B_g=4.0, B_opt=16.0, expected=6.0)

    def test_zero1_only_opt_sharded(self):
        """ZeRO-1: only B_opt divided by dp additionally."""
        # (B_w + B_g) / (tp×pp) + B_opt / (tp×pp×dp)
        # (4+4)/4 + 16/(4×4) = 2.0 + 1.0 = 3.0
        self._check(1, dp=4, tp=2, pp=2, B_w=4.0, B_g=4.0, B_opt=16.0, expected=3.0)

    def test_zero2_grads_and_opt_sharded(self):
        """ZeRO-2: B_w/(tp×pp) + (B_g + B_opt)/(tp×pp×dp)."""
        # 4/4 + (4+16)/(4×4) = 1.0 + 1.25 = 2.25
        self._check(2, dp=4, tp=2, pp=2, B_w=4.0, B_g=4.0, B_opt=16.0, expected=2.25)

    def test_zero3_everything_sharded(self):
        """ZeRO-3: (B_w + B_g + B_opt) / (tp × pp × dp) = B_state / G."""
        # (4+4+16) / (4×4) = 24/16 = 1.5
        self._check(3, dp=4, tp=2, pp=2, B_w=4.0, B_g=4.0, B_opt=16.0, expected=1.5)

    def test_zero3_equals_state_over_G(self):
        """ZeRO-3 state per device == B_state / G."""
        B_w, B_g, B_opt = 26e9, 26e9, 104e9
        dp, tp, pp = 8, 4, 2
        G = dp * tp * pp
        result = _state_bytes_per_device(B_w, B_g, B_opt, dp, tp, pp, 3)
        assert result == pytest.approx((B_w + B_g + B_opt) / G)

    def test_monotone_across_stages(self):
        """Higher ZeRO stage → less state per device (for dp > 1)."""
        B_w, B_g, B_opt = 26e9, 26e9, 104e9
        dp, tp, pp = 8, 2, 1
        vals = [
            _state_bytes_per_device(B_w, B_g, B_opt, dp, tp, pp, s)
            for s in (0, 1, 2, 3)
        ]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_zero0_dp1_equals_zero3_dp1(self):
        """With dp=1 all ZeRO stages are identical."""
        B_w, B_g, B_opt = 26e9, 26e9, 104e9
        dp, tp, pp = 1, 2, 2
        vals = {s: _state_bytes_per_device(B_w, B_g, B_opt, dp, tp, pp, s) for s in (0, 1, 2, 3)}
        assert vals[0] == pytest.approx(vals[1])
        assert vals[0] == pytest.approx(vals[2])
        assert vals[0] == pytest.approx(vals[3])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_zero_stage_raises(self):
        b0 = _bundle0()
        with pytest.raises(ValueError, match="zero_stage"):
            run_design(b0, inputs=DesignInputs(zero_stage=4))

    def test_negative_zero_stage_raises(self):
        b0 = _bundle0()
        with pytest.raises(ValueError, match="zero_stage"):
            run_design(b0, inputs=DesignInputs(zero_stage=-1))

    def test_valid_stages_do_not_raise(self):
        b0 = _bundle0()
        for stage in (0, 1, 2, 3):
            d = _design(b0, zero_stage=stage, G=256)
            assert d is not None


# ---------------------------------------------------------------------------
# Memory ordering — more ZeRO → fewer GPUs needed (or same)
# ---------------------------------------------------------------------------

class TestMemoryOrdering:
    def test_zero3_needs_fewer_or_equal_gpus_than_zero0(self):
        """ZeRO-3 enables a smaller or equal cluster than ZeRO-0."""
        b0 = _bundle0()
        d0 = _design(b0, zero_stage=0)
        d3 = _design(b0, zero_stage=3)
        if not d0["feasible"] or not d3["feasible"]:
            pytest.skip("one design infeasible for this param set")
        G0 = d0["solution"]["cluster"]["G"]
        G3 = d3["solution"]["cluster"]["G"]
        assert G3 <= G0

    def test_zero0_state_per_device_greater_than_zero3(self):
        """For the same (dp, tp, pp, G), ZeRO-0 uses more state memory."""
        b0 = _bundle0()
        d3 = _design(b0, zero_stage=3)
        if not d3["feasible"]:
            pytest.skip("design infeasible")
        G = d3["solution"]["cluster"]["G"]
        # Force both stages to the same G and check memory
        d0_fixed = _design(b0, zero_stage=0, G=G)
        d3_fixed = _design(b0, zero_stage=3, G=G)
        if not d0_fixed["feasible"] or not d3_fixed["feasible"]:
            pytest.skip("one design infeasible at fixed G")
        mem0 = d0_fixed["solution"]["memory"]["state_bytes_per_device"]
        mem3 = d3_fixed["solution"]["memory"]["state_bytes_per_device"]
        assert mem0 >= mem3

    def test_stage_in_solution_matches_input(self):
        """The zero_stage reported in the solution matches what was requested."""
        b0 = _bundle0()
        for stage in (0, 1, 2, 3):
            d = _design(b0, zero_stage=stage, G=256)
            if d["feasible"]:
                assert d["solution"]["memory"]["zero_stage"] == stage


# ---------------------------------------------------------------------------
# Communication — ZeRO-3 has more comm than ZeRO-0 for dp > 1
# ---------------------------------------------------------------------------

class TestCommOrdering:
    """
    Use a small model (1B params) so that a modest G is both compute- and
    memory-feasible, making dp > 1 achievable without a huge cluster.
    """

    def _b0_small(self):
        # 1B params, short deadline → compute-feasible at G=16-32
        return _bundle0(P=1e9, Tok=1e11, T=86400.0 * 7, B_step=2e10)

    def _get_comm(self, b0, zero_stage, G):
        d = _design(b0, zero_stage=zero_stage, G=G, tp_max=8)
        if not d["feasible"]:
            return None, None
        return (
            d["solution"]["communication"]["B_comm_per_gpu_bytes_per_step"],
            d["solution"]["parallelism"]["dp"],
        )

    def test_zero3_comm_geq_zero0_for_same_dp_gt_1(self):
        """ZeRO-3 moves more bytes per step than ZeRO-0 when dp > 1."""
        b0 = self._b0_small()
        G = 32
        c0, dp0 = self._get_comm(b0, zero_stage=0, G=G)
        c3, dp3 = self._get_comm(b0, zero_stage=3, G=G)
        if c0 is None or c3 is None:
            pytest.skip("one design infeasible")
        if dp0 <= 1 or dp3 <= 1:
            pytest.skip("dp == 1 for this config")
        assert c3 >= c0

    def test_zero0_comm_label(self):
        """ZeRO-0/1/2 use the ring allreduce comm model label."""
        b0 = self._b0_small()
        for stage in (0, 1, 2):
            d = _design(b0, zero_stage=stage, G=32)
            if d["feasible"]:
                label = d["solution"]["communication"]["model"]
                assert label == "ring_allreduce_dp_only"

    def test_zero3_comm_label(self):
        """ZeRO-3 uses the reduce-scatter/allgather label."""
        b0 = self._b0_small()
        d = _design(b0, zero_stage=3, G=32)
        if d["feasible"]:
            label = d["solution"]["communication"]["model"]
            assert label == "zero3_reduce_scatter_allgather"

    def test_zero1_comm_same_as_zero0(self):
        """ZeRO-1 and ZeRO-0 have identical communication volume."""
        b0 = self._b0_small()
        G = 32
        c0, _ = self._get_comm(b0, zero_stage=0, G=G)
        c1, _ = self._get_comm(b0, zero_stage=1, G=G)
        if c0 is None or c1 is None:
            pytest.skip("one design infeasible")
        assert c0 == pytest.approx(c1, rel=1e-9)

    def test_zero2_comm_same_as_zero0(self):
        """ZeRO-2 and ZeRO-0 have identical communication volume."""
        b0 = self._b0_small()
        G = 32
        c0, _ = self._get_comm(b0, zero_stage=0, G=G)
        c2, _ = self._get_comm(b0, zero_stage=2, G=G)
        if c0 is None or c2 is None:
            pytest.skip("one design infeasible")
        assert c0 == pytest.approx(c2, rel=1e-9)

    def test_dp1_means_no_comm_any_stage(self):
        """With dp=1 (single DP rank) there is no DP communication."""
        b0 = _bundle0()
        # Force G=tp*pp with no room for dp>1 by using a small G with tp_max=G
        for stage in (0, 1, 2, 3):
            d = run_design(
                b0,
                G=4,
                inputs=DesignInputs(tp_max=4, pp_max=4, zero_stage=stage),
            )
            if d["feasible"] and d["solution"]["parallelism"]["dp"] == 1:
                comm = d["solution"]["communication"]["B_comm_per_gpu_bytes_per_step"]
                assert comm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Default is ZeRO-0
# ---------------------------------------------------------------------------

class TestDefault:
    def test_default_zero_stage_is_zero(self):
        assert DesignInputs().zero_stage == 0

    def test_design_default_uses_zero0(self):
        b0 = _bundle0()
        d = run_design(b0, G=256)
        if d["feasible"]:
            assert d["solution"]["memory"]["zero_stage"] == 0
