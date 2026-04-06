"""Tests for Step 0 — Fundamentals."""
from __future__ import annotations

import pytest
from math import ceil

from causalcompute.core.types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_brief():
    """Small, easy-to-reason-about workload for unit tests."""
    return dict(
        workload=Workload(P=1e9, Tok=1e11, T=86400.0),  # 1B params, 100B tok, 1 day
        state=StateBytes(b_w=2.0, b_g=2.0, b_opt=8.0),
        io=IO(b_tok=2.0, A_io=1.0, b_ckpt=2.0, t_ckpt_max=300.0),
        device=Device(F_dev_sust_flop_s=1e15, B_dev_mem_bytes=8e10),
        step=StepWorkingSet(B_step_bytes=1e10),
        schedule=StepSchedule(Tok_per_step=1e6),
        update=AlgorithmStepFacts(b_update_per_param=2.0, k_update=1.0),
        fabric=FabricCapability(BW_node_sust_Bps=1e11),
        storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
        checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
    )


def run(brief_kwargs):
    return run_fundamentals(**brief_kwargs)


# ---------------------------------------------------------------------------
# Compute requirements
# ---------------------------------------------------------------------------

class TestComputeRequirements:
    def test_F_total(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        expected = w.c * w.P * w.Tok
        assert b["req"]["F_total_flop"] == pytest.approx(expected)

    def test_F_req(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        assert b["req"]["F_req_flop_s"] == pytest.approx(w.c * w.P * w.Tok / w.T)

    def test_R_tok(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        assert b["req"]["R_tok_req_tok_s"] == pytest.approx(w.Tok / w.T)

    def test_flops_coefficient_effect(self, tiny_brief):
        """Doubling c doubles F_total and F_req."""
        base = run(tiny_brief)
        doubled = dict(tiny_brief)
        doubled["workload"] = Workload(
            P=tiny_brief["workload"].P,
            Tok=tiny_brief["workload"].Tok,
            T=tiny_brief["workload"].T,
            c=tiny_brief["workload"].c * 2,
        )
        b2 = run(doubled)
        assert b2["req"]["F_total_flop"] == pytest.approx(2 * base["req"]["F_total_flop"])


# ---------------------------------------------------------------------------
# Model-state memory
# ---------------------------------------------------------------------------

class TestStateMemory:
    def test_state_components(self, tiny_brief):
        b = run(tiny_brief)
        req = b["req"]
        st = tiny_brief["state"]
        w = tiny_brief["workload"]
        assert req["B_weights_min_bytes"] == pytest.approx(st.b_w * w.P)
        assert req["B_grads_min_bytes"] == pytest.approx(st.b_g * w.P)
        assert req["B_opt_min_bytes"] == pytest.approx(st.b_opt * w.P)

    def test_state_total(self, tiny_brief):
        b = run(tiny_brief)
        req = b["req"]
        st = tiny_brief["state"]
        w = tiny_brief["workload"]
        expected = (st.b_w + st.b_g + st.b_opt) * w.P
        assert req["B_state_min_bytes"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# I/O contracts
# ---------------------------------------------------------------------------

class TestIO:
    def test_dataset_bw_includes_headroom(self, tiny_brief):
        b = run(tiny_brief)
        req = b["req"]
        w = tiny_brief["workload"]
        io = tiny_brief["io"]
        ideal = (w.Tok / w.T) * io.b_tok
        assert req["BW_dataset_plan_Bps"] == pytest.approx(ideal * io.A_io)

    def test_checkpoint_size(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        io = tiny_brief["io"]
        assert b["req"]["S_ckpt_bytes"] == pytest.approx(io.b_ckpt * w.P)

    def test_checkpoint_bw_req(self, tiny_brief):
        b = run(tiny_brief)
        S = b["req"]["S_ckpt_bytes"]
        io = tiny_brief["io"]
        assert b["req"]["BW_ckpt_req_Bps"] == pytest.approx(S / io.t_ckpt_max)


# ---------------------------------------------------------------------------
# Device lower bounds
# ---------------------------------------------------------------------------

class TestDeviceBounds:
    def test_compute_bound(self, tiny_brief):
        b = run(tiny_brief)
        req = b["req"]
        dev = tiny_brief["device"]
        expected = ceil(req["F_req_flop_s"] / dev.F_dev_sust_flop_s)
        assert b["device_bounds"]["N_compute_lower_bound"] == pytest.approx(expected)

    def test_state_bound(self, tiny_brief):
        b = run(tiny_brief)
        req = b["req"]
        dev = tiny_brief["device"]
        expected = ceil(req["B_state_min_bytes"] / dev.B_dev_mem_bytes)
        assert b["device_bounds"]["N_state_memory_lower_bound"] == pytest.approx(expected)

    def test_instant_bound_includes_step_working_set(self, tiny_brief):
        b = run(tiny_brief)
        req = b["req"]
        dev = tiny_brief["device"]
        step = tiny_brief["step"]
        B_instant = req["B_state_min_bytes"] + step.B_step_bytes
        expected = ceil(B_instant / dev.B_dev_mem_bytes)
        assert b["device_bounds"]["N_instant_device_lower_bound"] == pytest.approx(expected)

    def test_N_min_is_max_of_bounds(self, tiny_brief):
        b = run(tiny_brief)
        bounds = b["device_bounds"]
        N_min = bounds["N_min_lower_bound"]
        assert N_min == max(
            bounds["N_compute_lower_bound"],
            bounds["N_state_memory_lower_bound"],
            bounds["N_instant_device_lower_bound"],
        )


# ---------------------------------------------------------------------------
# Step timing
# ---------------------------------------------------------------------------

class TestStepTiming:
    def test_t_step_max(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        sched = tiny_brief["schedule"]
        R_tok = w.Tok / w.T
        expected = sched.Tok_per_step / R_tok
        assert b["movement"]["t_step_max_s"] == pytest.approx(expected)

    def test_F_step(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        sched = tiny_brief["schedule"]
        expected = w.c * w.P * sched.Tok_per_step
        assert b["stepfacts"]["F_step_flop"] == pytest.approx(expected)

    def test_N_steps(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        sched = tiny_brief["schedule"]
        assert b["stepfacts"]["N_steps"] == pytest.approx(w.Tok / sched.Tok_per_step)

    def test_update_payload(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        upd = tiny_brief["update"]
        expected = upd.k_update * w.P * upd.b_update_per_param
        assert b["movement"]["B_update_total_bytes_per_step"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Checkpoint timing
# ---------------------------------------------------------------------------

class TestCheckpointTiming:
    def test_t_ckpt_min(self, tiny_brief):
        b = run(tiny_brief)
        ckpt = b["ckpt"]
        sto = tiny_brief["storage"]
        expected = ckpt["S_ckpt_bytes"] / sto.BW_ckpt_sust_Bps
        assert ckpt["t_ckpt_min_s"] == pytest.approx(expected)

    def test_N_ckpt(self, tiny_brief):
        b = run(tiny_brief)
        w = tiny_brief["workload"]
        cp = tiny_brief["checkpoint"]
        expected = ceil(w.T / cp.seconds_per_ckpt)
        assert b["ckpt"]["N_ckpt"] == pytest.approx(expected)

    def test_ckpt_fraction_in_range(self, tiny_brief):
        b = run(tiny_brief)
        frac = b["ckpt"]["ckpt_fraction_of_run"]
        assert 0.0 <= frac <= 1.0

    def test_total_ckpt_time(self, tiny_brief):
        b = run(tiny_brief)
        ckpt = b["ckpt"]
        expected = ckpt["N_ckpt"] * ckpt["t_ckpt_min_s"]
        assert ckpt["T_ckpt_total_min_s"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Meta / fabric handoff
# ---------------------------------------------------------------------------

class TestMeta:
    def test_meta_contains_fabric_bw(self, tiny_brief):
        b = run(tiny_brief)
        fab = tiny_brief["fabric"]
        assert b["meta"]["BW_fabric_node_sust_Bps"] == pytest.approx(fab.BW_node_sust_Bps)

    def test_meta_contains_device_specs(self, tiny_brief):
        b = run(tiny_brief)
        dev = tiny_brief["device"]
        assert b["meta"]["F_dev_sust_flop_s"] == pytest.approx(dev.F_dev_sust_flop_s)
        assert b["meta"]["B_dev_mem_bytes"] == pytest.approx(dev.B_dev_mem_bytes)
