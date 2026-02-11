# step1_design/design.py
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Step 1 (Design) inputs — architecture knobs only
# =============================================================================

@dataclass(frozen=True)
class DesignInputs:
    # Cluster shape
    gpus_per_node: int = 8

    # Real-world sustained efficiencies (penalties vs physics best-case)
    eta_compute: float = 0.35   # (0,1]
    eta_fabric: float = 0.80    # (0,1]

    # Search bounds for factorization
    tp_max: int = 16
    pp_max: int = 16

    # Auto-sizing search bounds (only used when G is None)
    g_max_multiplier: int = 8   # search up to g_max_multiplier * N_guess

    # Communication model selection (v1)
    comm_model: str = "ring_allreduce_dp_only"  # placeholder for future variants


# =============================================================================
# Helpers
# =============================================================================

def _divisors(n: int) -> List[int]:
    out: List[int] = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
    return sorted(out)


def _validate_inputs(inp: DesignInputs) -> None:
    if inp.gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be > 0")
    if not (0.0 < inp.eta_compute <= 1.0):
        raise ValueError("eta_compute must be in (0,1]")
    if not (0.0 < inp.eta_fabric <= 1.0):
        raise ValueError("eta_fabric must be in (0,1]")
    if inp.tp_max <= 0 or inp.pp_max <= 0:
        raise ValueError("tp_max and pp_max must be > 0")
    if inp.g_max_multiplier <= 0:
        raise ValueError("g_max_multiplier must be > 0")
    if inp.comm_model not in ("ring_allreduce_dp_only",):
        raise ValueError("Unsupported comm_model (v1 only supports ring_allreduce_dp_only).")


def _score_candidate(c: Dict[str, Any], mode: str) -> Tuple[float, int, int, int]:
    """
    Lower score is better.

    mode == "A" (auto-size): minimize G first, then prefer bigger dp, then smaller pp, then smaller tp.
    mode == "B" (fixed G): maximize headroom, then prefer bigger dp, then smaller pp, then smaller tp.
    """
    dp = int(c["parallelism"]["dp"])
    tp = int(c["parallelism"]["tp"])
    pp = int(c["parallelism"]["pp"])
    G = int(c["cluster"]["G"])

    headroom = float(c["timing"]["t_step_max_s"] - c["timing"]["t_step_s"])

    if mode == "A":
        return (float(G), -dp, pp, tp)
    else:
        return (-headroom, -dp, pp, tp)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _ring_allreduce_bytes_per_rank(payload_bytes: float, dp: int) -> float:
    """
    Ring all-reduce cost per rank (bytes moved by one rank):
      2 * (dp - 1) / dp * payload
    """
    if dp <= 1:
        return 0.0
    return 2.0 * (dp - 1) / dp * payload_bytes


def _estimate_inter_node_fraction(dp: int, gpus_per_node: int) -> float:
    """
    Very simple placement-agnostic estimator of how much ring traffic is inter-node.

    frac_inter ≈ 1 - (gpus_per_node - 1)/(dp - 1), clamped to [0,1]
    """
    if dp <= 1:
        return 0.0
    return _clamp01(1.0 - (gpus_per_node - 1) / (dp - 1))


def _get_update_bytes_per_step(bundle0: Dict[str, Any]) -> float:
    """
    Step-0 changed names over time. Accept either:
      - movement.B_update_total_bytes_per_step (new)
      - movement.B_exchange_per_step_bytes     (old)
    """
    mv = bundle0.get("movement", {})
    if "B_update_total_bytes_per_step" in mv:
        return float(mv["B_update_total_bytes_per_step"])
    if "B_exchange_per_step_bytes" in mv:
        return float(mv["B_exchange_per_step_bytes"])
    # if neither exists, that's a contract break
    raise KeyError(
        "Step 0 movement is missing update-bytes-per-step. "
        "Expected 'B_update_total_bytes_per_step' (new) or 'B_exchange_per_step_bytes' (old)."
    )


# =============================================================================
# Core: run_design
# =============================================================================

def run_design(
    bundle0: Dict[str, Any],
    *,
    G: Optional[int] = None,
    inputs: Optional[DesignInputs] = None,
) -> Dict[str, Any]:
    """
    Step 1 (Design): consume Step 0 bundle and introduce DP/TP/PP, node shape, efficiencies.

    Step 0 provides global invariants (e.g. update bytes/step).
    Step 1 chooses a comm model + dp/tp/pp and produces per-GPU / per-node comm.
    """
    inp = inputs or DesignInputs()
    _validate_inputs(inp)

    # -------------------------------------------------------------------------
    # Extract Step-0 physics facts (DO NOT recompute)
    # -------------------------------------------------------------------------
    req = bundle0["req"]
    inst = bundle0["instant"]
    mv = bundle0["movement"]
    stepfacts = bundle0["stepfacts"]
    timecmp = bundle0["timecmp"]
    meta = bundle0["meta"]

    # Memory model ingredients
    B_state_total = float(req["B_state_min_bytes"])     # total state bytes (weights+grads+opt)
    B_step_total = float(inst["B_step_bytes"])          # instantaneous working set (global)

    # Timing model ingredients (compute side)
    t_step_max_s = float(timecmp["t_step_max_s"]) if "t_step_max_s" in timecmp else float(mv["t_step_max_s"])
    F_step_flop = float(stepfacts["F_step_flop"])

    # Capabilities
    F_dev_sust_flop_s = float(meta["F_dev_sust_flop_s"])
    B_dev_mem_bytes = float(meta["B_dev_mem_bytes"])

    # Per-node sustained payload bandwidth abstraction (v1)
    BW_fabric_sust_Bps = float(meta["BW_fabric_sust_Bps"])

    # Step-0 strongest "must exist" device count (lower bound, not a design)
    N_guess = int(ceil(float(timecmp["N_guess_devices"])))

    # Step-0 update bytes/step invariant (global) — NAME CHANGED IN STEP 0
    B_update_total_bytes_per_step = _get_update_bytes_per_step(bundle0)

    # -------------------------------------------------------------------------
    # Design-layer compute time under efficiencies
    # -------------------------------------------------------------------------
    def t_compute_s(Gi: int) -> float:
        return F_step_flop / (Gi * F_dev_sust_flop_s * inp.eta_compute)

    # -------------------------------------------------------------------------
    # Design-layer memory under DP/TP/PP
    # per-device = state/DP + instant/(TP*PP)
    # -------------------------------------------------------------------------
    def mem_per_device_bytes(dp: int, tp: int, pp: int) -> float:
        return (B_state_total / dp) + (B_step_total / (tp * pp))

    # -------------------------------------------------------------------------
    # Communication model (v1): ring allreduce on DP only, with TP sharding payload
    # -------------------------------------------------------------------------
    def comm_bytes_per_gpu_per_step(dp: int, tp: int, pp: int) -> Dict[str, float]:
        # v1 ignores PP+TP comm and focuses on DP gradient allreduce.
        # TP reduces the payload per rank by sharding parameters.
        tp_eff = max(1, tp)
        payload_per_rank = B_update_total_bytes_per_step / tp_eff

        B_dp = _ring_allreduce_bytes_per_rank(payload_per_rank, dp)

        # v1: tp/pp comm modeled as 0 (future work)
        B_tp = 0.0
        B_pp = 0.0

        B_total = B_dp + B_tp + B_pp

        frac_inter = _estimate_inter_node_fraction(dp=dp, gpus_per_node=inp.gpus_per_node)
        B_inter = B_total * frac_inter

        return {
            "model": "ring_allreduce_dp_only",
            "B_update_total_bytes_per_step": float(B_update_total_bytes_per_step),
            "payload_per_rank_bytes": float(payload_per_rank),
            "B_dp_allreduce_bytes_per_step": float(B_dp),
            "B_tp_bytes_per_step": float(B_tp),
            "B_pp_bytes_per_step": float(B_pp),
            "B_comm_per_gpu_bytes_per_step": float(B_total),
            "frac_inter_node_est": float(frac_inter),
            "B_comm_inter_per_gpu_bytes_per_step": float(B_inter),
            "B_comm_inter_per_node_bytes_per_step": float(B_inter * inp.gpus_per_node),
        }

    # -------------------------------------------------------------------------
    # Candidate generator for a fixed Gi
    # -------------------------------------------------------------------------
    def search_factorizations(Gi: int, mode: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None

        # For feasibility diagnostics
        diag: Dict[str, Any] = {
            "Gi": int(Gi),
            "t_step_max_s": float(t_step_max_s),
            "time_note": "Timing depends on dp/tp/pp via comm model (ring all-reduce baseline).",
            "mem_best_case_bytes": float(mem_per_device_bytes(dp=Gi, tp=1, pp=1)),
            "mem_best_case_ok": bool(mem_per_device_bytes(dp=Gi, tp=1, pp=1) <= B_dev_mem_bytes),
            "B_update_total_bytes_per_step": float(B_update_total_bytes_per_step),
        }

        # Enumerate dp divisors; enforce dp*tp*pp == Gi
        for dp in _divisors(Gi):
            m = Gi // dp

            for tp in range(1, min(inp.tp_max, m) + 1):
                if m % tp != 0:
                    continue
                pp = m // tp
                if pp < 1 or pp > inp.pp_max:
                    continue

                mem = mem_per_device_bytes(dp, tp, pp)
                if mem > B_dev_mem_bytes:
                    continue

                comm = comm_bytes_per_gpu_per_step(dp=dp, tp=tp, pp=pp)
                B_inter_node_bytes_per_step = float(comm["B_comm_inter_per_node_bytes_per_step"])

                # Effective sustained fabric payload (per node) after eta_fabric
                BW_effective_node_Bps = BW_fabric_sust_Bps * inp.eta_fabric
                t_comm = B_inter_node_bytes_per_step / BW_effective_node_Bps if BW_effective_node_Bps > 0 else float("inf")

                tc = t_compute_s(Gi)
                ts = tc + t_comm

                if ts > t_step_max_s:
                    continue

                cand = {
                    "cluster": {
                        "G": int(Gi),
                        "gpus_per_node": int(inp.gpus_per_node),
                        "nodes": int(ceil(Gi / inp.gpus_per_node)),
                    },
                    "parallelism": {"dp": int(dp), "tp": int(tp), "pp": int(pp)},
                    "efficiency": {"eta_compute": float(inp.eta_compute), "eta_fabric": float(inp.eta_fabric)},
                    "memory": {
                        "model": "state/DP + instant/(TP×PP)",
                        "B_dev_mem_bytes": float(B_dev_mem_bytes),
                        "bytes_per_device": float(mem),
                        "state_bytes_total": float(B_state_total),
                        "instant_bytes_total": float(B_step_total),
                        "mem_ok": True,
                    },
                    "communication": comm,
                    "timing": {
                        "t_step_max_s": float(t_step_max_s),
                        "t_compute_s": float(tc),
                        "t_comm_s": float(t_comm),
                        "t_step_s": float(ts),
                        "headroom_s": float(t_step_max_s - ts),
                        "time_ok": True,
                    },
                    "constraints": {"dp_tp_pp_equals_G": True, "ok": True},
                }

                if best is None or _score_candidate(cand, mode) < _score_candidate(best, mode):
                    best = cand

        if best is not None:
            diag.update(
                {
                    "time_ok": True,
                    "t_compute_s": float(best["timing"]["t_compute_s"]),
                    "t_step_s": float(best["timing"]["t_step_s"]),
                }
            )
        else:
            diag.update(
                {
                    "time_ok": False,
                    "t_compute_s": float(t_compute_s(Gi)),
                }
            )

        return best, diag

    # -------------------------------------------------------------------------
    # Mode selection and search
    # -------------------------------------------------------------------------
    mode = "B" if G is not None else "A"

    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "inputs": inp.__dict__.copy(),
        "physics_echo": {
            "B_state_min_bytes": float(B_state_total),
            "B_step_bytes": float(B_step_total),
            "F_step_flop": float(F_step_flop),
            "t_step_max_s": float(t_step_max_s),
            "capabilities": {
                "F_dev_sust_flop_s": float(F_dev_sust_flop_s),
                "B_dev_mem_bytes": float(B_dev_mem_bytes),
                "BW_fabric_sust_Bps": float(BW_fabric_sust_Bps),
            },
            "N_guess_devices": int(N_guess),
            "B_update_total_bytes_per_step": float(B_update_total_bytes_per_step),
        },
        "search": {},
    }

    best_solution: Optional[Dict[str, Any]] = None
    last_diag: Optional[Dict[str, Any]] = None

    if mode == "B":
        Gi = int(G)
        if Gi <= 0:
            raise ValueError("G must be a positive integer when provided.")
        best_solution, last_diag = search_factorizations(Gi, mode="B")
        diagnostics["search"] = {"G_fixed": Gi, "diag": last_diag}
    else:
        G_start = max(1, N_guess)
        G_max = max(G_start, int(G_start * inp.g_max_multiplier))

        diagnostics["search"] = {"G_start": int(G_start), "G_max": int(G_max), "attempts": []}

        for Gi in range(G_start, G_max + 1):
            sol, d = search_factorizations(Gi, mode="A")
            diagnostics["search"]["attempts"].append(d)
            last_diag = d

            if sol is not None:
                best_solution = sol
                break

    if best_solution is None:
        if last_diag is None:
            last_diag = {"note": "no diagnostics available"}

        reason = {
            "note": (
                "No (dp,tp,pp) satisfied memory+time under the provided efficiencies "
                "within the searched G range." if mode == "A"
                else "No (dp,tp,pp) satisfied memory+time for the fixed G."
            ),
            "last_attempt": last_diag,
            "hints": [],
        }

        if isinstance(last_diag, dict):
            if not last_diag.get("mem_best_case_ok", True):
                reason["hints"].append(
                    "Memory closure failed even in best-case sharding: increase device memory, "
                    "increase G (for more DP sharding), or reduce state/instantaneous working set."
                )
            reason["hints"].append(
                "Time closure depends on dp/tp/pp under the comm model: consider increasing BW_fabric_sust_Bps, "
                "increasing eta_fabric, increasing G, or reducing the update payload per step (Tok_per_step)."
            )

        return {
            "feasible": False,
            "solution": None,
            "handoff": {
                "cluster": None,
                "parallelism": None,
                "communication": None,
                "efficiency": {"eta_compute": float(inp.eta_compute), "eta_fabric": float(inp.eta_fabric)},
            },
            "diagnostics": diagnostics,
            "no_solution_reason": reason,
        }

    comm_handoff = {
        "model": str(best_solution["communication"]["model"]),
        "B_comm_per_gpu_bytes_per_step": float(best_solution["communication"]["B_comm_per_gpu_bytes_per_step"]),
        "B_comm_inter_per_gpu_bytes_per_step": float(best_solution["communication"]["B_comm_inter_per_gpu_bytes_per_step"]),
        "B_comm_inter_per_node_bytes_per_step": float(best_solution["communication"]["B_comm_inter_per_node_bytes_per_step"]),
        "frac_inter_node_est": float(best_solution["communication"]["frac_inter_node_est"]),
    }

    return {
        "feasible": True,
        "solution": best_solution,
        "handoff": {
            "cluster": best_solution["cluster"],
            "parallelism": best_solution["parallelism"],
            "communication": comm_handoff,
            "efficiency": best_solution["efficiency"],
        },
        "diagnostics": diagnostics,
    }

