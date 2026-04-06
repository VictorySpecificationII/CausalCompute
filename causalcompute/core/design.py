"""
Step 1 — Design.

Introduces parallelism (DP/TP/PP), node shape, and efficiency penalties.
Consumes the Step 0 bundle; does NOT recompute anything from Step 0.

Search strategy:
  Mode A (G=None): find the smallest G ≥ N_min that yields a feasible (dp, tp, pp).
  Mode B (G fixed): find the best (dp, tp, pp) for the given G.

Communication model v1: ring all-reduce on DP gradients only.
  TP reduces the per-rank payload by sharding parameters.
  PP and TP inter-stage comm are modelled as zero (future work).
"""
from __future__ import annotations

from math import ceil
from typing import Any, Optional

from .types import DesignInputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divisors(n: int) -> list[int]:
    out: list[int] = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
    return sorted(out)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ring_allreduce_bytes_per_rank(payload_bytes: float, dp: int) -> float:
    """Bytes moved by one rank in a ring all-reduce: 2·(dp-1)/dp · payload."""
    if dp <= 1:
        return 0.0
    return 2.0 * (dp - 1) / dp * payload_bytes


def _inter_node_fraction(dp: int, gpus_per_node: int) -> float:
    """Fraction of ring traffic that crosses node boundaries (placement-agnostic estimate)."""
    if dp <= 1:
        return 0.0
    return _clamp01(1.0 - (gpus_per_node - 1) / (dp - 1))


def _score(candidate: dict, mode: str) -> tuple:
    """Lower is better. Mode A minimises G; mode B maximises headroom."""
    dp = candidate["parallelism"]["dp"]
    tp = candidate["parallelism"]["tp"]
    pp = candidate["parallelism"]["pp"]
    G = candidate["cluster"]["G"]
    headroom = candidate["timing"]["t_step_max_s"] - candidate["timing"]["t_step_s"]
    if mode == "A":
        return (float(G), -dp, pp, tp)
    return (-headroom, -dp, pp, tp)


def _validate(inp: DesignInputs) -> None:
    if inp.gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be > 0")
    if not (0.0 < inp.eta_compute <= 1.0):
        raise ValueError("eta_compute must be in (0, 1]")
    if not (0.0 < inp.eta_fabric <= 1.0):
        raise ValueError("eta_fabric must be in (0, 1]")
    if inp.tp_max <= 0 or inp.pp_max <= 0:
        raise ValueError("tp_max and pp_max must be > 0")
    if inp.g_max_multiplier <= 0:
        raise ValueError("g_max_multiplier must be > 0")
    if inp.comm_model != "ring_allreduce_dp_only":
        raise ValueError("Unsupported comm_model; v1 supports 'ring_allreduce_dp_only' only.")
    if not (0.0 <= inp.comm_exposed_fraction <= 1.0):
        raise ValueError("comm_exposed_fraction must be in [0, 1]")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_design(
    bundle0: dict[str, Any],
    *,
    G: Optional[int] = None,
    inputs: Optional[DesignInputs] = None,
) -> dict[str, Any]:
    """
    Step 1: find a feasible cluster + parallelism strategy.

    Parameters
    ----------
    bundle0 : dict
        Output of run_fundamentals().
    G : int or None
        Fixed GPU count (mode B), or None to auto-size (mode A).
    inputs : DesignInputs or None
        Architecture knobs. Defaults to DesignInputs() if None.

    Returns
    -------
    dict with keys:
        feasible        bool
        solution        dict | None
        handoff         dict   (stable digest for Step 2)
        diagnostics     dict
        no_solution_reason  dict | None
    """
    inp = inputs or DesignInputs()
    _validate(inp)

    # -- Unpack Step-0 bundle (never recompute, only consume) ----------------
    req = bundle0["req"]
    inst = bundle0["instant"]
    mv = bundle0["movement"]
    stepfacts = bundle0["stepfacts"]
    timecmp = bundle0["timecmp"]
    meta = bundle0["meta"]

    B_state = float(req["B_state_min_bytes"])
    B_step = float(inst["B_step_bytes"])
    t_step_max = float(timecmp["t_step_max_s"])
    F_step = float(stepfacts["F_step_flop"])
    F_dev = float(meta["F_dev_sust_flop_s"])
    B_dev_mem = float(meta["B_dev_mem_bytes"])
    BW_fabric = float(meta["BW_fabric_node_sust_Bps"])
    N_guess = int(ceil(float(timecmp["N_guess_devices"])))
    B_update = float(mv["B_update_total_bytes_per_step"])

    # -- Inner functions (closures over extracted scalars) -------------------

    def t_compute(Gi: int) -> float:
        return F_step / (Gi * F_dev * inp.eta_compute)

    def mem_per_device(dp: int, tp: int, pp: int) -> float:
        # State sharded across DP; working set sharded across TP×PP
        return (B_state / dp) + (B_step / (tp * pp))

    def comm_breakdown(dp: int, tp: int, pp: int) -> dict:
        payload_per_rank = B_update / max(1, tp)
        B_dp = _ring_allreduce_bytes_per_rank(payload_per_rank, dp)
        frac_inter = _inter_node_fraction(dp, inp.gpus_per_node)
        B_inter_gpu = B_dp * frac_inter
        B_inter_node = B_inter_gpu * inp.gpus_per_node
        return {
            "model": "ring_allreduce_dp_only",
            "B_update_total_bytes_per_step": B_update,
            "payload_per_rank_bytes": payload_per_rank,
            "B_dp_allreduce_bytes_per_step": B_dp,
            "B_comm_per_gpu_bytes_per_step": B_dp,
            "frac_inter_node_est": frac_inter,
            "B_comm_inter_per_gpu_bytes_per_step": B_inter_gpu,
            "B_comm_inter_per_node_bytes_per_step": B_inter_node,
        }

    def search_G(Gi: int, mode: str) -> tuple[Optional[dict], dict]:
        """Try all (dp, tp, pp) for a given Gi. Returns (best_candidate, diag)."""
        best: Optional[dict] = None
        diag: dict = {
            "Gi": Gi,
            "t_step_max_s": t_step_max,
            "mem_best_case_bytes": mem_per_device(Gi, 1, 1),
            "mem_best_case_ok": mem_per_device(Gi, 1, 1) <= B_dev_mem,
        }

        for dp in _divisors(Gi):
            m = Gi // dp
            for tp in range(1, min(inp.tp_max, m) + 1):
                if m % tp != 0:
                    continue
                pp = m // tp
                if pp < 1 or pp > inp.pp_max:
                    continue

                mem = mem_per_device(dp, tp, pp)
                if mem > B_dev_mem:
                    continue

                comm = comm_breakdown(dp, tp, pp)
                BW_eff = BW_fabric * inp.eta_fabric
                t_comm = (
                    comm["B_comm_inter_per_node_bytes_per_step"] / BW_eff
                    if BW_eff > 0 else float("inf")
                )
                tc = t_compute(Gi)
                ts = tc + inp.comm_exposed_fraction * t_comm

                if ts > t_step_max:
                    continue

                cand = {
                    "cluster": {
                        "G": Gi,
                        "nodes": ceil(Gi / inp.gpus_per_node),
                        "gpus_per_node": inp.gpus_per_node,
                    },
                    "parallelism": {"dp": dp, "tp": tp, "pp": pp},
                    "efficiency": {
                        "eta_compute": inp.eta_compute,
                        "eta_fabric": inp.eta_fabric,
                    },
                    "memory": {
                        "model": "state/DP + instant/(TP×PP)",
                        "B_dev_mem_bytes": B_dev_mem,
                        "bytes_per_device": mem,
                        "state_bytes_total": B_state,
                        "instant_bytes_total": B_step,
                    },
                    "communication": comm,
                    "timing": {
                        "t_step_max_s": t_step_max,
                        "t_compute_s": tc,
                        "t_comm_s": t_comm,
                        "t_step_s": ts,
                        "headroom_s": t_step_max - ts,
                    },
                }

                if best is None or _score(cand, mode) < _score(best, mode):
                    best = cand

        if best is not None:
            diag.update({"time_ok": True, "t_step_s": best["timing"]["t_step_s"]})
        else:
            diag.update({"time_ok": False, "t_compute_s": t_compute(Gi)})

        return best, diag

    # -- Mode selection & search ---------------------------------------------
    mode = "B" if G is not None else "A"
    diagnostics: dict = {
        "mode": mode,
        "inputs": inp.__dict__.copy(),
        "physics_echo": {
            "B_state_min_bytes": B_state,
            "B_step_bytes": B_step,
            "F_step_flop": F_step,
            "t_step_max_s": t_step_max,
            "F_dev_sust_flop_s": F_dev,
            "B_dev_mem_bytes": B_dev_mem,
            "BW_fabric_node_sust_Bps": BW_fabric,
            "N_guess_devices": N_guess,
            "B_update_total_bytes_per_step": B_update,
        },
    }

    best: Optional[dict] = None
    last_diag: Optional[dict] = None

    if mode == "B":
        Gi = int(G)  # type: ignore[arg-type]
        if Gi <= 0:
            raise ValueError("G must be a positive integer when provided.")
        best, last_diag = search_G(Gi, "B")
        diagnostics["search"] = {"G_fixed": Gi, "diag": last_diag}
    else:
        G_start = max(1, N_guess)
        G_max = max(G_start, G_start * inp.g_max_multiplier)
        attempts: list[dict] = []
        for Gi in range(G_start, G_max + 1):
            sol, d = search_G(Gi, "A")
            attempts.append(d)
            last_diag = d
            if sol is not None:
                best = sol
                break
        diagnostics["search"] = {
            "G_start": G_start,
            "G_max": G_max,
            "attempts": attempts,
        }

    # -- Build return value --------------------------------------------------
    if best is None:
        hints = []
        if last_diag and not last_diag.get("mem_best_case_ok", True):
            hints.append(
                "Memory failed even at best-case sharding. "
                "Increase device memory, raise G, or reduce state / working set."
            )
        hints.append(
            "Timing may fail due to comm model. "
            "Try increasing BW_fabric, eta_fabric, G, or lowering Tok_per_step."
        )
        return {
            "feasible": False,
            "solution": None,
            "handoff": {
                "cluster": None,
                "parallelism": None,
                "communication": None,
                "efficiency": {
                    "eta_compute": inp.eta_compute,
                    "eta_fabric": inp.eta_fabric,
                },
            },
            "diagnostics": diagnostics,
            "no_solution_reason": {
                "note": (
                    "No (dp, tp, pp) satisfied memory + timing within the searched G range."
                    if mode == "A"
                    else "No (dp, tp, pp) satisfied memory + timing for the fixed G."
                ),
                "last_attempt": last_diag,
                "hints": hints,
            },
        }

    comm_h = best["communication"]
    return {
        "feasible": True,
        "solution": best,
        "handoff": {
            "cluster": best["cluster"],
            "parallelism": best["parallelism"],
            "communication": {
                "model": comm_h["model"],
                "B_comm_per_gpu_bytes_per_step": comm_h["B_comm_per_gpu_bytes_per_step"],
                "B_comm_inter_per_gpu_bytes_per_step": comm_h["B_comm_inter_per_gpu_bytes_per_step"],
                "B_comm_inter_per_node_bytes_per_step": comm_h["B_comm_inter_per_node_bytes_per_step"],
                "frac_inter_node_est": comm_h["frac_inter_node_est"],
            },
            "efficiency": best["efficiency"],
        },
        "diagnostics": diagnostics,
        "no_solution_reason": None,
    }
