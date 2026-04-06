"""
Integration tests — run the full pipeline end-to-end.

Also includes a regression test against the Nebius H100 validation numbers
from the original CausalCompute evidence file.
"""
from __future__ import annotations

import pytest

from causalcompute.core.types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
    DesignInputs,
    Device,
    FabricCapability,
    IO,
    PowerInputs,
    RackInputs,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    ThermalInputs,
    Workload,
)
from causalcompute.core.fundamentals import run_fundamentals
from causalcompute.core.design import run_design
from causalcompute.core.thermals import run_thermals
from causalcompute.io.loader import load_brief_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THIRTEEN_B_BRIEF = {
    "workload": {"P": 13e9, "Tok": 3e12, "T": 2592000.0, "c": 6.0},
    "state_bytes": {"b_w": 2.0, "b_g": 2.0, "b_opt": 8.0},
    "io": {"b_tok": 2.0, "A_io": 1.3, "b_ckpt": 2.0, "t_ckpt_max": 300.0},
    "device": {"F_dev_sust_flop_s": 1e15, "B_dev_mem_bytes": 8e10},
    "step": {
        "B_step_bytes": 1.5e11,
        "Tok_per_step": 4e7,
        "update": {"b_update_per_param": 2.0, "k_update": 1.0},
    },
    "capabilities": {
        "fabric": {"BW_node_sust_Bps": 1e11},
        "storage": {"BW_ckpt_sust_Bps": 5e9},
        "checkpoint_policy": {"seconds_per_ckpt": 3600.0},
    },
    "design": {
        "G": None,
        "gpus_per_node": 8,
        "eta_compute": 0.35,
        "eta_fabric": 0.80,
        "tp_max": 16,
        "pp_max": 16,
        "g_max_multiplier": 8,
        "comm_model": "ring_allreduce_dp_only",
    },
    "power_thermals": {
        "power": {"P_gpu_W": 700.0, "P_cpu_W_per_node": 250.0, "P_other_W_per_node": 300.0, "PUE": 1.30},
        "cooling": {"mode": "liquid", "deltaT_air_C": 15.0, "deltaT_liquid_C": 12.0},
        "rack": {"nodes_per_rack": None, "rack_power_limit_W": None, "racks": None},
    },
}


def run_pipeline(brief_dict: dict):
    brief = load_brief_dict(brief_dict)
    b0 = run_fundamentals(
        workload=brief.workload, state=brief.state, io=brief.io,
        device=brief.device, step=brief.step, schedule=brief.schedule,
        update=brief.update, fabric=brief.fabric, storage=brief.storage,
        checkpoint=brief.checkpoint,
    )
    design = run_design(b0, G=brief.design_G, inputs=brief.design)
    thermals_result = None
    if design["feasible"]:
        thermals_result = run_thermals(
            design, power=brief.power, thermals=brief.thermals,
            rack=brief.rack, T_run_s=brief.workload.T,
        )
    return b0, design, thermals_result


# ---------------------------------------------------------------------------
# Basic pipeline tests
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    def test_13b_pipeline_runs(self):
        b0, design, thermals_result = run_pipeline(THIRTEEN_B_BRIEF)
        assert b0 is not None
        assert design is not None
        assert design["feasible"] is True
        assert thermals_result is not None

    def test_13b_design_has_valid_cluster(self):
        from math import ceil
        _, design, _ = run_pipeline(THIRTEEN_B_BRIEF)
        cl = design["solution"]["cluster"]
        assert cl["G"] > 0
        assert cl["nodes"] > 0
        # nodes is the ceiling of G / gpus_per_node (G need not be a multiple)
        assert cl["nodes"] == ceil(cl["G"] / cl["gpus_per_node"])

    def test_13b_step_time_feasible(self):
        _, design, _ = run_pipeline(THIRTEEN_B_BRIEF)
        tim = design["solution"]["timing"]
        assert tim["t_step_s"] <= tim["t_step_max_s"]

    def test_13b_memory_feasible(self):
        _, design, _ = run_pipeline(THIRTEEN_B_BRIEF)
        mem = design["solution"]["memory"]
        assert mem["bytes_per_device"] <= mem["B_dev_mem_bytes"]

    def test_13b_thermals_has_positive_power(self):
        _, _, thermals_result = run_pipeline(THIRTEEN_B_BRIEF)
        assert thermals_result["power"]["P_IT_W"] > 0
        assert thermals_result["power"]["P_facility_W"] > thermals_result["power"]["P_IT_W"]

    def test_13b_cooling_flow_positive(self):
        _, _, thermals_result = run_pipeline(THIRTEEN_B_BRIEF)
        assert thermals_result["cooling"]["vol_flow_m3_s"] > 0

    def test_13b_energy_over_run(self):
        _, _, thermals_result = run_pipeline(THIRTEEN_B_BRIEF)
        assert thermals_result["energy"] is not None
        assert thermals_result["energy"]["E_IT_kWh"] > 0


# ---------------------------------------------------------------------------
# Regression — Nebius H100 InfiniBand validation
#
# The original CausalCompute was validated against measured Nebius H100 runs.
# Measured step times from evidence/nebius_h100_ib_validation.md:
#   8 GPU:  0.037809 s/step  (predicted: ~0.037810 — error < 0.003%)
#   16 GPU: 0.039018 s/step  (predicted: ~0.038975 — error: 0.11%)
#
# We reproduce the setup here and verify the predicted step time stays
# within 1% of the measured values.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Nebius H100 validation briefs (exact parameters from evidence/)
#
# Synthetic transformer: hidden=8192, layers=8, batch_per_gpu=8
#   P = 9 × 8192² ≈ 603,979,776 params
#   T is set to the step-time deadline (0.0405 s), Tok = one step's tokens
#
# Calibrated: eta_compute=0.3505, eta_fabric=1.0, comm_exposed=0.05
# Measured fabric: BW_node_sust_Bps = 4.15e11 B/s (NCCL all-reduce plateau)
# ---------------------------------------------------------------------------

NEBIUS_DP8 = {
    "workload": {"P": 6.03979776e8, "Tok": 29256.0, "T": 0.0405, "c": 6.0},
    "state_bytes": {"b_w": 2.0, "b_g": 2.0, "b_opt": 8.0},
    "io": {"b_tok": 2.0, "A_io": 1.0, "b_ckpt": 2.0, "t_ckpt_max": 300.0},
    "device": {"F_dev_sust_flop_s": 1.0e15, "B_dev_mem_bytes": 8.0e10},
    "step": {
        "B_step_bytes": 1.0e9,
        "Tok_per_step": 29256.0,
        "update": {"b_update_per_param": 2.0, "k_update": 1.0},
    },
    "capabilities": {
        "fabric": {"BW_node_sust_Bps": 4.15e11},
        "storage": {"BW_ckpt_sust_Bps": 5e9},
        "checkpoint_policy": {"seconds_per_ckpt": 1.0e12},
    },
    "design": {
        "G": 8, "gpus_per_node": 8,
        "eta_compute": 0.3505, "eta_fabric": 1.0,
        "tp_max": 1, "pp_max": 1, "g_max_multiplier": 1,
        "comm_model": "ring_allreduce_dp_only",
        "comm_exposed_fraction": 0.05,
    },
    "power_thermals": {
        "power": {"P_gpu_W": 700.0, "P_cpu_W_per_node": 250.0,
                  "P_other_W_per_node": 300.0, "PUE": 1.30},
        "cooling": {"mode": "liquid", "deltaT_air_C": 15.0, "deltaT_liquid_C": 12.0},
        "rack": {},
    },
}

NEBIUS_DP16 = {
    **{k: v for k, v in NEBIUS_DP8.items() if k not in ("workload", "step", "design")},
    "workload": {"P": 6.03979776e8, "Tok": 58512.0, "T": 0.0405, "c": 6.0},
    "step": {
        "B_step_bytes": 1.0e9,
        "Tok_per_step": 58512.0,
        "update": {"b_update_per_param": 2.0, "k_update": 1.0},
    },
    "design": {
        "G": 16, "gpus_per_node": 8,
        "eta_compute": 0.3505, "eta_fabric": 1.0,
        "tp_max": 1, "pp_max": 1, "g_max_multiplier": 1,
        "comm_model": "ring_allreduce_dp_only",
        "comm_exposed_fraction": 0.05,
    },
}


class TestNebiusRegression:
    """
    Regression against measured Nebius H100 InfiniBand step times.

    Measured on real hardware (evidence/nebius_h100_ib_validation.md):
      8 GPU:  0.037809 s  (predicted: 0.037810, error < 0.003%)
      16 GPU: 0.039018 s  (predicted: 0.038975, error < 0.12%)
    """

    def _step_time(self, brief_dict: dict) -> float:
        brief = load_brief_dict(brief_dict)
        b0 = run_fundamentals(
            workload=brief.workload, state=brief.state, io=brief.io,
            device=brief.device, step=brief.step, schedule=brief.schedule,
            update=brief.update, fabric=brief.fabric, storage=brief.storage,
            checkpoint=brief.checkpoint,
        )
        G = brief.design_G
        design = run_design(b0, G=G, inputs=brief.design)
        assert design["feasible"], f"Design infeasible for G={G}"
        return design["solution"]["timing"]["t_step_s"]

    def test_8gpu_step_time_within_1pct(self):
        measured = 0.037809
        predicted = self._step_time(NEBIUS_DP8)
        assert abs(predicted - measured) / measured < 0.01, (
            f"8-GPU: predicted {predicted:.6f} s vs measured {measured:.6f} s "
            f"(error {abs(predicted - measured) / measured * 100:.2f}%)"
        )

    def test_16gpu_step_time_within_1pct(self):
        measured = 0.039018
        predicted = self._step_time(NEBIUS_DP16)
        assert abs(predicted - measured) / measured < 0.01, (
            f"16-GPU: predicted {predicted:.6f} s vs measured {measured:.6f} s "
            f"(error {abs(predicted - measured) / measured * 100:.2f}%)"
        )

    def test_scaling_penalty_reasonable(self):
        """16-GPU step time > 8-GPU due to inter-node comm."""
        t8 = self._step_time(NEBIUS_DP8)
        t16 = self._step_time(NEBIUS_DP16)
        assert t16 > t8, "16-GPU should be slower than 8-GPU due to inter-node comm"


# ---------------------------------------------------------------------------
# Loader round-trip
# ---------------------------------------------------------------------------

class TestLoaderRoundTrip:
    def test_load_brief_dict_produces_brief(self):
        from causalcompute.core.types import Brief
        brief = load_brief_dict(THIRTEEN_B_BRIEF)
        assert isinstance(brief, Brief)

    def test_workload_values_preserved(self):
        brief = load_brief_dict(THIRTEEN_B_BRIEF)
        assert brief.workload.P == pytest.approx(13e9)
        assert brief.workload.Tok == pytest.approx(3e12)
        assert brief.workload.T == pytest.approx(2592000.0)

    def test_design_G_none_for_autosize(self):
        brief = load_brief_dict(THIRTEEN_B_BRIEF)
        assert brief.design_G is None

    def test_design_G_parsed_as_int(self):
        import copy
        d = copy.deepcopy(THIRTEEN_B_BRIEF)
        d["design"]["G"] = 64
        brief = load_brief_dict(d)
        assert brief.design_G == 64

    def test_missing_required_field_raises(self):
        import copy
        d = copy.deepcopy(THIRTEEN_B_BRIEF)
        del d["workload"]
        with pytest.raises(KeyError):
            load_brief_dict(d)

    def test_wrong_type_raises(self):
        import copy
        d = copy.deepcopy(THIRTEEN_B_BRIEF)
        d["workload"]["P"] = "not-a-number"
        with pytest.raises(ValueError):
            load_brief_dict(d)
