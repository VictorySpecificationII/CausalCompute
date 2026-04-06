"""Tests for Step 6 — Cost Model."""
from __future__ import annotations

import pytest

from causalcompute.core.types import (
    AlgorithmStepFacts,
    CheckpointPolicy,
    CostInputs,
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
from causalcompute.core.cost import run_cost

_SECONDS_PER_YEAR = 365.25 * 86400.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_pipeline(nodes_target: int = 8):
    """Return (bundle0, design, thermals, network, storage, bom)."""
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
    design = run_design(b0, G=nodes_target * 8, inputs=DesignInputs(gpus_per_node=8))
    thermals = run_thermals(design, power=PowerInputs(), thermals=ThermalInputs(),
                            rack=RackInputs(), T_run_s=86400.0 * 30)
    network = run_network(design, network=NetworkInputs(nics_per_node=8))
    storage = run_storage(b0, design, storage=StorageInputs())
    bom = run_bom(design, thermals, network, storage, bundle0=b0)
    return b0, design, thermals, network, storage, bom


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_gpu_cost_raises(self):
        _, _, _, _, _, bom = _full_pipeline()
        with pytest.raises(ValueError):
            run_cost(bom, cost=CostInputs(gpu_unit_cost=-1.0))

    def test_zero_amortization_raises(self):
        _, _, _, _, _, bom = _full_pipeline()
        with pytest.raises(ValueError):
            run_cost(bom, cost=CostInputs(capex_amortization_years=0))

    def test_negative_electricity_raises(self):
        _, _, _, _, _, bom = _full_pipeline()
        with pytest.raises(ValueError):
            run_cost(bom, cost=CostInputs(electricity_usd_kwh=-0.01))


# ---------------------------------------------------------------------------
# CapEx
# ---------------------------------------------------------------------------

class TestCapEx:
    def test_gpu_capex_equals_count_times_unit_cost(self):
        b0, _, _, _, _, bom = _full_pipeline(8)
        gpu_price = 25_000.0
        result = run_cost(bom, cost=CostInputs(gpu_unit_cost=gpu_price))
        assert result["capex"]["gpus"] == pytest.approx(bom["totals"]["gpus"] * gpu_price)

    def test_chassis_capex_equals_nodes_times_unit_cost(self):
        b0, _, _, _, _, bom = _full_pipeline(8)
        chassis_price = 10_000.0
        result = run_cost(bom, cost=CostInputs(node_chassis_cost=chassis_price))
        assert result["capex"]["node_chassis"] == pytest.approx(
            bom["totals"]["compute_nodes"] * chassis_price
        )

    def test_switch_capex_zero_when_no_network(self):
        b0, design, thermals, _, _, _ = _full_pipeline()
        bom_no_net = run_bom(design, thermals, None, None, bundle0=b0)
        result = run_cost(bom_no_net, thermals, bundle0=b0)
        assert result["capex"]["switches"] == 0

    def test_total_capex_is_sum_of_parts(self):
        _, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals)
        cx = result["capex"]
        expected = (cx["gpus"] + cx["node_chassis"] + cx["nics"] + cx["switches"] +
                    cx["cables"] + cx["storage_nodes"] + cx["drives"] + cx["racks"])
        assert cx["total"] == pytest.approx(expected)

    def test_total_capex_positive(self):
        _, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals)
        assert result["capex"]["total"] > 0

    def test_larger_cluster_higher_capex(self):
        _, _, t4, _, _, bom4 = _full_pipeline(4)
        _, _, t16, _, _, bom16 = _full_pipeline(16)
        cost4 = run_cost(bom4, t4)["capex"]["total"]
        cost16 = run_cost(bom16, t16)["capex"]["total"]
        assert cost16 > cost4

    def test_zero_unit_costs_give_zero_capex(self):
        _, _, _, _, _, bom = _full_pipeline()
        result = run_cost(bom, cost=CostInputs(
            gpu_unit_cost=0, node_chassis_cost=0, nic_unit_cost=0,
            switch_unit_cost=0, cable_unit_cost=0, storage_node_cost=0,
            drive_unit_cost=0, rack_unit_cost=0,
        ))
        assert result["capex"]["total"] == 0.0


# ---------------------------------------------------------------------------
# OpEx
# ---------------------------------------------------------------------------

class TestOpEx:
    def test_energy_cost_present_with_thermals(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        assert result["opex"]["energy_cost"] is not None

    def test_energy_cost_none_without_thermals(self):
        b0, design, _, _, _, _ = _full_pipeline()
        bom = run_bom(design, None, None, None, bundle0=b0)
        result = run_cost(bom, None, bundle0=b0)
        assert result["opex"]["energy_cost"] is None

    def test_energy_cost_equals_kwh_times_rate(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        rate = 0.10
        result = run_cost(bom, thermals, bundle0=b0, cost=CostInputs(electricity_usd_kwh=rate))
        kwh = result["opex"]["energy_kwh"]
        assert result["opex"]["energy_cost"] == pytest.approx(kwh * rate)

    def test_higher_pue_higher_energy_cost(self):
        b0, design, _, _, _, bom = _full_pipeline()
        thermals_low = run_thermals(design, power=PowerInputs(PUE=1.1), thermals=ThermalInputs(),
                                    rack=RackInputs(), T_run_s=86400.0 * 30)
        thermals_high = run_thermals(design, power=PowerInputs(PUE=1.8), thermals=ThermalInputs(),
                                     rack=RackInputs(), T_run_s=86400.0 * 30)
        cost_low = run_cost(bom, thermals_low, bundle0=b0)["opex"]["energy_cost"]
        cost_high = run_cost(bom, thermals_high, bundle0=b0)["opex"]["energy_cost"]
        assert cost_high > cost_low


# ---------------------------------------------------------------------------
# Run cost
# ---------------------------------------------------------------------------

class TestRunCost:
    def test_total_amortised_present_with_bundle0(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        assert result["run_cost"]["total_amortised"] is not None

    def test_total_full_capex_present_with_bundle0(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        assert result["run_cost"]["total_full_capex"] is not None

    def test_amortised_less_than_full_capex(self):
        """A 30-day run amortised over 3 years should be a fraction of full CapEx."""
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0,
                          cost=CostInputs(capex_amortization_years=3.0))
        rc = result["run_cost"]
        assert rc["total_amortised"] < rc["total_full_capex"]

    def test_longer_amortization_reduces_run_cost(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        r3 = run_cost(bom, thermals, bundle0=b0, cost=CostInputs(capex_amortization_years=3.0))
        r5 = run_cost(bom, thermals, bundle0=b0, cost=CostInputs(capex_amortization_years=5.0))
        assert r5["run_cost"]["total_amortised"] < r3["run_cost"]["total_amortised"]

    def test_amortised_capex_fraction_formula(self):
        """capex_amortised_for_run ≈ total_capex * T_run / amortization_period."""
        b0, _, thermals, _, _, bom = _full_pipeline()
        years = 4.0
        result = run_cost(bom, thermals, bundle0=b0,
                          cost=CostInputs(capex_amortization_years=years))
        rc = result["run_cost"]
        T_run_s = result["assumptions"]["T_run_s"]
        expected = result["capex"]["total"] * T_run_s / (years * _SECONDS_PER_YEAR)
        assert rc["capex_amortised_for_run"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Cost per token
# ---------------------------------------------------------------------------

class TestCostPerToken:
    def test_cost_per_token_present_with_bundle0(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        assert result["cost_per_token"]["amortised"] is not None
        assert result["cost_per_token"]["full_capex"] is not None

    def test_cost_per_token_none_without_bundle0(self):
        _, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals)
        assert result["cost_per_token"]["amortised"] is None

    def test_cost_per_token_positive(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        assert result["cost_per_token"]["amortised"] > 0
        assert result["cost_per_token"]["full_capex"] > 0

    def test_cost_per_token_full_capex_greater_than_amortised(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        ct = result["cost_per_token"]
        assert ct["full_capex"] > ct["amortised"]

    def test_cost_per_token_equals_run_cost_over_tok(self):
        b0, _, thermals, _, _, bom = _full_pipeline()
        result = run_cost(bom, thermals, bundle0=b0)
        ct = result["cost_per_token"]
        rc = result["run_cost"]
        Tok = ct["Tok"]
        assert ct["amortised"] == pytest.approx(rc["total_amortised"] / Tok, rel=1e-6)
        assert ct["full_capex"] == pytest.approx(rc["total_full_capex"] / Tok, rel=1e-6)

    def test_more_tokens_lower_full_capex_cost_per_token(self):
        """
        Same cluster, 2× tokens (2× runtime) → lower full-CapEx cost/token.

        CapEx is fixed; doubling Tok (and runtime) doubles OpEx but halves the
        CapEx contribution per token.  Amortised cost/token is invariant (both
        numerator and denominator scale with T_run), so we use the full_capex
        view which captures the fixed-hardware-cost-spread-over-more-tokens effect.
        """
        G = 64
        # Short run
        b0_s = run_fundamentals(
            workload=Workload(P=1e9, Tok=1e11, T=86400.0 * 30),
            state=StateBytes(), io=IO(),
            device=Device(F_dev_sust_flop_s=1e15, B_dev_mem_bytes=8e10),
            step=StepWorkingSet(B_step_bytes=1e10),
            schedule=StepSchedule(Tok_per_step=1e6),
            update=AlgorithmStepFacts(b_update_per_param=2.0),
            fabric=FabricCapability(BW_node_sust_Bps=1e11),
            storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
            checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
        )
        # Long run — same cluster, 2× tokens, 2× wall-clock
        b0_l = run_fundamentals(
            workload=Workload(P=1e9, Tok=2e11, T=86400.0 * 60),
            state=StateBytes(), io=IO(),
            device=Device(F_dev_sust_flop_s=1e15, B_dev_mem_bytes=8e10),
            step=StepWorkingSet(B_step_bytes=1e10),
            schedule=StepSchedule(Tok_per_step=1e6),
            update=AlgorithmStepFacts(b_update_per_param=2.0),
            fabric=FabricCapability(BW_node_sust_Bps=1e11),
            storage=StorageCapability(BW_ckpt_sust_Bps=5e9),
            checkpoint=CheckpointPolicy(seconds_per_ckpt=3600.0),
        )
        design_s = run_design(b0_s, G=G, inputs=DesignInputs(gpus_per_node=8))
        design_l = run_design(b0_l, G=G, inputs=DesignInputs(gpus_per_node=8))
        thermals_s = run_thermals(design_s, power=PowerInputs(), thermals=ThermalInputs(),
                                   rack=RackInputs(), T_run_s=86400.0 * 30)
        thermals_l = run_thermals(design_l, power=PowerInputs(), thermals=ThermalInputs(),
                                   rack=RackInputs(), T_run_s=86400.0 * 60)
        bom_s = run_bom(design_s, thermals_s, bundle0=b0_s)
        bom_l = run_bom(design_l, thermals_l, bundle0=b0_l)
        # Use a high capex_amortization_years so CapEx >> OpEx and the effect is clear
        cost_cfg = CostInputs(capex_amortization_years=10.0)
        ct_s = run_cost(bom_s, thermals_s, bundle0=b0_s, cost=cost_cfg)["cost_per_token"]["full_capex"]
        ct_l = run_cost(bom_l, thermals_l, bundle0=b0_l, cost=cost_cfg)["cost_per_token"]["full_capex"]
        # Full-CapEx cost/token: (fixed_capex + 2×opex) / (2×Tok) < (fixed_capex + opex) / Tok
        # when capex >> opex (which we enforce via long amortization)
        assert ct_l < ct_s
