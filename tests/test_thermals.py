"""Tests for Step 2 — Thermals."""
from __future__ import annotations

import pytest

from causalcompute.core.types import (
    AirCoolingInputs,
    AlgorithmStepFacts,
    CheckpointPolicy,
    Device,
    DesignInputs,
    FabricCapability,
    IO,
    LiquidCoolingInputs,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feasible_design():
    """A feasible Step-1 bundle using a generous workload."""
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
    return run_design(b0)


# ---------------------------------------------------------------------------
# Power ledger
# ---------------------------------------------------------------------------

class TestPowerLedger:
    def test_P_IT_correct(self, feasible_design):
        pwr = PowerInputs(P_gpu_W=700.0, P_cpu_W_per_node=250.0, P_other_W_per_node=300.0, PUE=1.3)
        result = run_thermals(feasible_design, power=pwr)
        G = result["cluster"]["G"]
        nodes = result["cluster"]["nodes"]
        expected_IT = G * 700 + nodes * (250 + 300)
        assert result["power"]["P_IT_W"] == pytest.approx(expected_IT)

    def test_P_facility_is_PUE_times_IT(self, feasible_design):
        pwr = PowerInputs(PUE=1.5)
        result = run_thermals(feasible_design, power=pwr)
        assert result["power"]["P_facility_W"] == pytest.approx(result["power"]["P_IT_W"] * 1.5)

    def test_heat_equals_IT_power(self, feasible_design):
        result = run_thermals(feasible_design)
        assert result["heat"]["Qdot_W"] == pytest.approx(result["power"]["P_IT_W"])

    def test_kW_conversion(self, feasible_design):
        result = run_thermals(feasible_design)
        assert result["power"]["P_IT_kW"] == pytest.approx(result["power"]["P_IT_W"] / 1000.0)


# ---------------------------------------------------------------------------
# Air cooling
# ---------------------------------------------------------------------------

class TestAirCooling:
    def test_vol_flow_formula(self, feasible_design):
        """V̇ = Q / (ρ · cₚ · ΔT)"""
        air = AirCoolingInputs(deltaT_C=15.0, rho_kg_m3=1.2, cp_J_kgK=1005.0)
        th = ThermalInputs(mode="air", air=air)
        result = run_thermals(feasible_design, thermals=th)
        Q = result["heat"]["Qdot_W"]
        m_dot = Q / (air.cp_J_kgK * air.deltaT_C)
        V_dot = m_dot / air.rho_kg_m3
        assert result["cooling"]["vol_flow_m3_s"] == pytest.approx(V_dot)

    def test_smaller_delta_T_means_more_flow(self, feasible_design):
        th_hot = ThermalInputs(mode="air", air=AirCoolingInputs(deltaT_C=20.0))
        th_cold = ThermalInputs(mode="air", air=AirCoolingInputs(deltaT_C=10.0))
        flow_hot = run_thermals(feasible_design, thermals=th_hot)["cooling"]["vol_flow_m3_s"]
        flow_cold = run_thermals(feasible_design, thermals=th_cold)["cooling"]["vol_flow_m3_s"]
        assert flow_cold > flow_hot

    def test_mode_is_air(self, feasible_design):
        th = ThermalInputs(mode="air")
        result = run_thermals(feasible_design, thermals=th)
        assert result["cooling"]["mode"] == "air"


# ---------------------------------------------------------------------------
# Liquid cooling
# ---------------------------------------------------------------------------

class TestLiquidCooling:
    def test_vol_flow_formula(self, feasible_design):
        liq = LiquidCoolingInputs(deltaT_C=7.0, rho_kg_m3=1000.0, cp_J_kgK=4186.0)
        th = ThermalInputs(mode="liquid", liquid=liq)
        result = run_thermals(feasible_design, thermals=th)
        Q = result["heat"]["Qdot_W"]
        m_dot = Q / (liq.cp_J_kgK * liq.deltaT_C)
        V_dot = m_dot / liq.rho_kg_m3
        assert result["cooling"]["vol_flow_m3_s"] == pytest.approx(V_dot)

    def test_liquid_uses_less_flow_than_air(self, feasible_design):
        """Water has much higher specific heat → lower volumetric flow for same Q."""
        air_th = ThermalInputs(mode="air")
        liq_th = ThermalInputs(mode="liquid")
        air_flow = run_thermals(feasible_design, thermals=air_th)["cooling"]["vol_flow_m3_s"]
        liq_flow = run_thermals(feasible_design, thermals=liq_th)["cooling"]["vol_flow_m3_s"]
        assert liq_flow < air_flow

    def test_mode_is_liquid(self, feasible_design):
        th = ThermalInputs(mode="liquid")
        result = run_thermals(feasible_design, thermals=th)
        assert result["cooling"]["mode"] == "liquid"


# ---------------------------------------------------------------------------
# Rack sanity
# ---------------------------------------------------------------------------

class TestRackSanity:
    def test_rack_count_from_nodes_per_rack(self, feasible_design):
        from math import ceil
        nodes = feasible_design["solution"]["cluster"]["nodes"]
        rack = RackInputs(nodes_per_rack=4)
        result = run_thermals(feasible_design, rack=rack)
        expected_racks = ceil(nodes / 4)
        assert result["rack"]["racks"] == expected_racks

    def test_explicit_rack_count(self, feasible_design):
        rack = RackInputs(racks=10)
        result = run_thermals(feasible_design, rack=rack)
        assert result["rack"]["racks"] == 10

    def test_rack_power_ok_when_within_limit(self, feasible_design):
        rack = RackInputs(racks=1, rack_power_limit_W=1e9)  # 1 GW limit — always passes
        result = run_thermals(feasible_design, rack=rack)
        assert result["rack"]["rack_power_ok"] is True

    def test_rack_power_fail_when_over_limit(self, feasible_design):
        rack = RackInputs(racks=1000, rack_power_limit_W=1.0)  # 1 W limit — always fails
        result = run_thermals(feasible_design, rack=rack)
        assert result["rack"]["rack_power_ok"] is False

    def test_no_rack_config_gives_none(self, feasible_design):
        result = run_thermals(feasible_design, rack=RackInputs())
        assert result["rack"] is None


# ---------------------------------------------------------------------------
# Energy over run
# ---------------------------------------------------------------------------

class TestEnergyOverRun:
    def test_energy_computed_when_T_run_given(self, feasible_design):
        result = run_thermals(feasible_design, T_run_s=86400.0)
        assert result["energy"] is not None
        assert result["energy"]["E_IT_kWh"] > 0

    def test_energy_none_when_T_run_not_given(self, feasible_design):
        result = run_thermals(feasible_design)
        assert result["energy"] is None

    def test_energy_formula(self, feasible_design):
        T = 3600.0  # 1 hour
        result = run_thermals(feasible_design, T_run_s=T)
        P_IT = result["power"]["P_IT_W"]
        expected_kWh = (P_IT * T) / 3_600_000.0
        assert result["energy"]["E_IT_kWh"] == pytest.approx(expected_kWh)

    def test_facility_energy_is_larger_than_IT(self, feasible_design):
        result = run_thermals(feasible_design, T_run_s=3600.0)
        assert result["energy"]["E_facility_kWh"] > result["energy"]["E_IT_kWh"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_infeasible_design_raises(self):
        """run_thermals must raise on an infeasible Step-1 bundle."""
        bad_design = {"feasible": False, "handoff": {"cluster": None}}
        with pytest.raises(ValueError):
            run_thermals(bad_design)

    def test_invalid_PUE_raises(self, feasible_design):
        with pytest.raises(ValueError):
            run_thermals(feasible_design, power=PowerInputs(PUE=0.5))

    def test_invalid_P_gpu_raises(self, feasible_design):
        with pytest.raises(ValueError):
            run_thermals(feasible_design, power=PowerInputs(P_gpu_W=-100.0))

    def test_invalid_cooling_mode_raises(self, feasible_design):
        with pytest.raises(ValueError):
            run_thermals(feasible_design, thermals=ThermalInputs(mode="steam"))  # type: ignore

    def test_invalid_T_run_raises(self, feasible_design):
        with pytest.raises(ValueError):
            run_thermals(feasible_design, T_run_s=-1.0)
