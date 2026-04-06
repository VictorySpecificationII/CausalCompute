"""
Text report formatters for CLI and narrative output.

All formatting is kept here so that core modules stay pure.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# SI display helpers
# ---------------------------------------------------------------------------

def _si(x: float) -> str:
    """Scientific notation, 4 significant figures."""
    return f"{x:.4e}"


def _gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB"


def _gbs(x: float) -> str:
    return f"{x / 1e9:.2f} GB/s"


def _gflops(x: float) -> str:
    return f"{x / 1e12:.2f} TFLOP/s"


def _kw(x: float) -> str:
    return f"{x / 1000:.1f} kW"


def _m3s(x: float) -> str:
    return f"{x:.4f} m³/s"


def _days(x: float) -> str:
    return f"{x / 86400:.1f} days"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# Summary table (printed by CLI)
# ---------------------------------------------------------------------------

RULE = "=" * 64


def _usd(x: float) -> str:
    if x >= 1e9:
        return f"${x / 1e9:.2f}B"
    if x >= 1e6:
        return f"${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"${x / 1e3:.1f}k"
    return f"${x:.2f}"


def print_summary(
    bundle0: dict[str, Any],
    design: dict[str, Any],
    thermals: Optional[dict[str, Any]],
    network: Optional[dict[str, Any]] = None,
    storage: Optional[dict[str, Any]] = None,
    bom: Optional[dict[str, Any]] = None,
    cost: Optional[dict[str, Any]] = None,
    bottleneck: Optional[dict[str, Any]] = None,
) -> None:
    req = bundle0["req"]
    mv = bundle0["movement"]

    print(f"\n{RULE}")
    print("CAUSALCOMPUTE — SUMMARY")
    print(RULE)

    # -- Step 0 ------------------------------------------------------------
    print("\nStep 0 — Fundamentals")
    print(f"  Compute required      {_gflops(req['F_req_flop_s'])}")
    print(f"  Token rate required   {_si(req['R_tok_req_tok_s'])} tok/s")
    print(f"  Dataset BW (planned)  {_gbs(req['BW_dataset_plan_Bps'])}")
    print(f"  Model state (1 copy)  {_gb(req['B_state_min_bytes'])}")
    print(f"  Checkpoint size       {_gb(req['S_ckpt_bytes'])}")
    print(f"  Step time budget      {mv['t_step_max_s']:.4f} s/step")
    print(f"  Update payload/step   {_gb(mv['B_update_total_bytes_per_step'])}")

    # -- Step 1 ------------------------------------------------------------
    print("\nStep 1 — Design")
    if not design.get("feasible"):
        reason = design.get("no_solution_reason", {})
        print(f"  INFEASIBLE — {reason.get('note', 'no feasible cluster found')}")
        for hint in reason.get("hints", []):
            print(f"    hint: {hint}")
        print(f"\n{RULE}\n")
        return

    sol = design["solution"]
    cl = sol["cluster"]
    par = sol["parallelism"]
    tim = sol["timing"]
    mem = sol["memory"]

    print(f"  Cluster     {cl['G']} GPUs  ({cl['nodes']} nodes × {cl['gpus_per_node']} GPUs/node)")
    print(f"  Parallelism dp={par['dp']}  tp={par['tp']}  pp={par['pp']}")
    print(f"  Step time   {tim['t_step_s']:.4f} s  (budget {tim['t_step_max_s']:.4f} s, "
          f"headroom {tim['headroom_s']:.4f} s)")
    print(f"  Mem/device  {_gb(mem['bytes_per_device'])}  (cap {_gb(mem['B_dev_mem_bytes'])})")

    # -- Bottleneck analysis -----------------------------------------------
    print("\nBottleneck Analysis")
    if bottleneck is None:
        print("  skipped")
    else:
        b = bottleneck["binding"]
        db = bottleneck["device_bounds"]
        st = bottleneck["step_time"]
        mem = bottleneck["memory"]
        comm = bottleneck["communication"]

        print(f"\n  Binding constraint: {b['label'].upper()}  "
              f"(headroom {db[b['key']]['headroom_x']:.2f}×)")

        print(f"\n  {'Bound':<24}  {'N':>6}  {'Headroom':>10}  {'Active':>8}")
        print(f"  {'-'*52}")
        for k, info in [
            ("compute", db["compute"]),
            ("state",   db["state"]),
            ("instant", db["instant"]),
        ]:
            marker = "  ◀ BINDING" if info["is_binding"] else ""
            print(f"  {info['label']:<24}  {info['N']:>6.0f}  {info['headroom_x']:>9.2f}×{marker}")
        print(f"  {'Actual G':<24}  {db['G_actual']:>6}")

        print(f"\n  Step time   {st['t_step_s']:.4f} s / {st['t_step_max_s']:.4f} s budget  "
              f"({st['headroom_pct']:.1f}% headroom)")
        print(f"  Breakdown   compute {st['compute_pct']:.0f}%  "
              f"comm {st['comm_pct']:.0f}%  "
              f"headroom {100 - st['compute_pct'] - st['comm_pct']:.0f}%")
        print(f"  Memory      {mem['bytes_per_device']/1e9:.1f} GB / "
              f"{mem['B_dev_mem_bytes']/1e9:.1f} GB  ({mem['utilization_pct']:.0f}% utilised)")

        for rec in bottleneck["recommendations"]:
            print(f"\n  → {rec}")

    # -- Step 2 ------------------------------------------------------------
    print("\nStep 2 — Power & Thermals")
    if thermals is None:
        print("  skipped")
    else:
        h = thermals["handoff"]
        pwr = h["power"]
        cool = h["cooling"]

        print(f"  IT power      {_kw(pwr['P_IT_W'])}")
        print(f"  Facility pwr  {_kw(pwr['P_facility_W'])}  (PUE {pwr['PUE']:.2f})")
        print(f"  Heat load     {_kw(h['heat']['Qdot_W'])}")

        if cool["mode"] == "air":
            print(f"  Airflow       {_m3s(cool['vol_flow_m3_s'])}  (ΔT {cool['deltaT_C']:.1f} °C)")
        else:
            print(f"  Coolant flow  {_m3s(cool['vol_flow_m3_s'])}  (ΔT {cool['deltaT_C']:.1f} °C)")

        rk = h.get("rack")
        if rk:
            print(f"  Racks         {rk['racks']}  ({_kw(rk['P_IT_per_rack_W'])}/rack)")

        en = h.get("energy")
        if en:
            print(f"  Energy (IT)   {en['E_IT_kWh']:.0f} kWh  over {_days(en['T_run_s'])}")

    # -- Step 3 ------------------------------------------------------------
    print("\nStep 3 — Network")
    if network is None:
        print("  skipped")
    else:
        sw = network["switches"]
        bw = network["bandwidth"]
        ca = network["cables"]
        cons = network["consistency"]

        topology = "rail-optimised leaf/spine" if network["topology"] == "rail_optimised_leaf_spine" else "leaf/spine"
        print(f"  Topology      {topology}")
        print(f"  Leaf switches {sw['num_leaves']}  ({sw['leaf_downlinks']} down / {sw['leaf_uplinks']} up ports)")
        print(f"  Spine switches {sw['num_spines']}")
        print(f"  Total switches {sw['total']}")
        print(f"  Per-node BW   {_gbs(bw['bw_per_node_effective_Bps'])}  (raw {_gbs(bw['bw_per_node_raw_Bps'])})")
        print(f"  Bisection BW  {bw['bisection_bw_Tbps']:.2f} Tb/s")
        print(f"  Cables        {ca['total']}  ({ca['server_to_leaf']} server→leaf, {ca['leaf_to_spine']} leaf→spine)")
        if not network["two_tier_feasible"]:
            print(f"  WARNING: nodes ({network['assumptions']['nics_per_node']} NICs) exceeds 2-tier capacity "
                  f"({network['max_nodes_2tier']} nodes). Consider a 3-tier fat-tree.")
        if not cons["ok"]:
            print(f"  WARNING: {cons['note']}")

    # -- Step 4 ------------------------------------------------------------
    print("\nStep 4 — Storage")
    if storage is None:
        print("  skipped")
    else:
        nd = storage["nodes"]
        dp = storage["dataset_pool"]
        cp = storage["checkpoint_pool"]
        nw = storage["network"]
        cons = storage["consistency"]

        print(f"  Storage nodes   {nd['num_storage_nodes']}  ({nd['drives_per_node']} drives/node, {nd['total_drives']} drives total)")
        print(f"  Dataset pool    {dp['drives']} drives  ({_gbs(dp['aggregate_bw_Bps'])} agg BW, {_gb(dp['bytes_stored'])} stored, ×{dp['replication']} replication)")
        print(f"  Checkpoint pool {cp['drives']} drives  ({_gbs(cp['aggregate_bw_Bps'])} agg BW, {_gb(cp['bytes_stored'])} stored, ×{cp['replication']} replication, {cp['keep_count']} generations)")
        print(f"  Storage net BW  {_gbs(nw['aggregate_net_bw_Bps'])}  (need {_gbs(nw['required_net_bw_Bps'])}, ratio {nw['net_ratio']:.2f}×)")
        if not nw["net_ok"]:
            print(f"  WARNING: Storage network BW insufficient — add more storage nodes or higher-BW NICs.")
        if not cons["ok"]:
            print(f"  WARNING: {cons['note']}")

    # -- Step 5 ------------------------------------------------------------
    print("\nStep 5 — Bill of Materials")
    if bom is None:
        print("  skipped")
    else:
        t = bom["totals"]
        c = bom["compute"]
        ns = bom["node_spec"]

        # Node spec header
        nvlink_gb = ns["intra_node_bw_Bps"] / 1e9
        gdrdma = "yes" if ns["gpudirect_rdma"] else "NO"
        dram_per_node_TB = ns["dram_bytes_per_node"] / 1e12
        print(f"\n  Node spec  {ns['root_complex']}  |  "
              f"NVLink {nvlink_gb:.0f} GB/s/node  |  "
              f"GPUDirect RDMA: {gdrdma}  |  "
              f"{ns['cpu_cores_per_node']} cores/node  |  "
              f"{dram_per_node_TB:.1f} TB DRAM/node")

        print(f"\n  {'COMPONENT':<28}  {'QTY':>8}")
        print(f"  {'-'*38}")
        print(f"  {'Compute nodes':<28}  {t['compute_nodes']:>8}")
        print(f"  {'  └─ GPUs':<28}  {t['gpus']:>8}  ({c['gpus_per_node']} per node)")
        print(f"  {'  └─ CPU cores':<28}  {t['cpu_cores']:>8}  ({ns['cpu_cores_per_node']} per node)")
        print(f"  {'  └─ System DRAM':<28}  {'':>8}  {t['dram_bytes'] / 1e12:.1f} TB total  ({ns['dram_bytes_per_node'] / 1e12:.1f} TB/node)")
        if t["nics"] is not None:
            print(f"  {'  └─ NICs':<28}  {t['nics']:>8}  ({bom['network']['nics_per_node']} per node)")
        if t["switches"] is not None:
            nw = bom["network"]
            print(f"  {'Leaf switches':<28}  {nw['leaf_switches']:>8}")
            print(f"  {'Spine switches':<28}  {nw['spine_switches']:>8}")
            print(f"  {'Total switches':<28}  {t['switches']:>8}")
            print(f"  {'Cables (server→leaf)':<28}  {nw['cables_server_to_leaf']:>8}")
            print(f"  {'Cables (leaf→spine)':<28}  {nw['cables_leaf_to_spine']:>8}")
            print(f"  {'Total cables':<28}  {t['cables']:>8}")
        if t["storage_nodes"] > 0:
            sb = bom["storage"]
            print(f"  {'Storage nodes':<28}  {t['storage_nodes']:>8}  ({sb['drives_per_storage_node']} drives/node)")
            print(f"  {'  └─ Dataset drives':<28}  {sb['dataset_drives']:>8}")
            print(f"  {'  └─ Checkpoint drives':<28}  {sb['checkpoint_drives']:>8}")
            print(f"  {'  └─ Total drives':<28}  {t['drives']:>8}")
        if t["racks"] is not None:
            print(f"  {'Racks':<28}  {t['racks']:>8}")
        print(f"  {'-'*38}")
        print(f"  {'ALL NODES (compute+storage)':<28}  {t['all_nodes']:>8}")

        for w in bom["warnings"]:
            print(f"\n  WARNING: {w}")

    # -- Step 6 ------------------------------------------------------------
    print("\nStep 6 — Cost Model")
    if cost is None:
        print("  skipped")
    else:
        cx = cost["capex"]
        op = cost["opex"]
        rc = cost["run_cost"]
        ct = cost["cost_per_token"]
        assum = cost["assumptions"]

        print(f"\n  CapEx breakdown")
        print(f"  {'GPUs':<28}  {_usd(cx['gpus']):>10}  ({bom['totals']['gpus']} × unit cost)" if bom else f"  {'GPUs':<28}  {_usd(cx['gpus']):>10}")
        print(f"  {'Node chassis':<28}  {_usd(cx['node_chassis']):>10}")
        if cx['nics'] > 0:
            print(f"  {'NICs':<28}  {_usd(cx['nics']):>10}")
        if cx['switches'] > 0:
            print(f"  {'Switches':<28}  {_usd(cx['switches']):>10}")
        if cx['cables'] > 0:
            print(f"  {'Cables':<28}  {_usd(cx['cables']):>10}")
        if cx['storage_nodes'] > 0:
            print(f"  {'Storage nodes':<28}  {_usd(cx['storage_nodes']):>10}")
        if cx['drives'] > 0:
            print(f"  {'Drives':<28}  {_usd(cx['drives']):>10}")
        if cx['racks'] > 0:
            print(f"  {'Racks':<28}  {_usd(cx['racks']):>10}")
        print(f"  {'-'*40}")
        print(f"  {'Total CapEx':<28}  {_usd(cx['total']):>10}")

        if op['energy_cost'] is not None:
            print(f"\n  OpEx (energy)  {_usd(op['energy_cost'])}  "
                  f"({op['energy_kwh']:,.0f} kWh × ${assum['electricity_usd_kwh']}/kWh)")

        if rc['total_amortised'] is not None:
            print(f"\n  Run cost (amortised, {assum['amortization_years']:.0f}yr)  "
                  f"{_usd(rc['total_amortised'])}")
            print(f"  Run cost (full CapEx)            {_usd(rc['total_full_capex'])}")

        if ct['amortised'] is not None:
            print(f"\n  Cost / token (amortised)   ${ct['amortised']:.6f}")
            print(f"  Cost / token (full CapEx)  ${ct['full_capex']:.6f}")
            print(f"  ({ct['Tok']:.2e} tokens)")

        if op.get('note'):
            print(f"\n  note: {op['note']}")
        if ct.get('note'):
            print(f"  note: {ct['note']}")

    print(f"\n{RULE}\n")


# ---------------------------------------------------------------------------
# Step-0 narrative (--story mode)
# ---------------------------------------------------------------------------

def print_story(bundle0: dict[str, Any], workload_params: float, workload_tokens: float, deadline_s: float) -> None:
    req = bundle0["req"]
    bounds = bundle0["device_bounds"]
    inst = bundle0["instant"]
    mv = bundle0["movement"]
    sf = bundle0["stepfacts"]
    tc = bundle0["timecmp"]
    ckpt = bundle0["ckpt"]
    meta = bundle0["meta"]

    print("\n" + RULE)
    print("STEP 0 — CAUSAL STORY")
    print(RULE)

    P = workload_params
    Tok = workload_tokens
    T = deadline_s

    print(f"\nWorkload: {_si(P)} params, {_si(Tok)} tokens, deadline {_days(T)}")
    print(f"\nTotal compute: {_si(req['F_total_flop'])} FLOP")
    print(f"Sustained compute required: {_gflops(req['F_req_flop_s'])}")
    print(f"Token rate required: {_si(req['R_tok_req_tok_s'])} tok/s")

    print(f"\nModel state (single copy):")
    print(f"  Weights:    {_gb(req['B_weights_min_bytes'])}")
    print(f"  Gradients:  {_gb(req['B_grads_min_bytes'])}")
    print(f"  Optimizer:  {_gb(req['B_opt_min_bytes'])}")
    print(f"  Total:      {_gb(req['B_state_min_bytes'])}")

    print(f"\nI/O:")
    print(f"  Dataset BW (planned): {_gbs(req['BW_dataset_plan_Bps'])}")
    print(f"  Checkpoint size:      {_gb(req['S_ckpt_bytes'])}")
    print(f"  Checkpoint BW req:    {_gbs(req['BW_ckpt_req_Bps'])}")

    print(f"\nDevice lower bounds (device: {_gflops(meta['F_dev_sust_flop_s'])}, "
          f"{_gb(meta['B_dev_mem_bytes'])} mem):")
    print(f"  Compute bound: {bounds['N_compute_lower_bound']:.0f} devices")
    print(f"  State bound:   {bounds['N_state_memory_lower_bound']:.0f} devices")
    print(f"  Instant bound: {bounds['N_instant_device_lower_bound']:.0f} devices")
    print(f"  → N_min:       {bounds['N_min_lower_bound']:.0f} devices")

    print(f"\nStep timing:")
    print(f"  Steps in run:    {sf['N_steps']:.0f}")
    print(f"  FLOPs/step:      {_si(sf['F_step_flop'])}")
    print(f"  Step time budget: {mv['t_step_max_s']:.4f} s")
    print(f"  Min compute time: {tc['t_step_compute_min_s']:.6f} s")

    print(f"\nCheckpointing:")
    print(f"  Checkpoint size: {_gb(ckpt['S_ckpt_bytes'])}")
    print(f"  Min ckpt time:   {ckpt['t_ckpt_min_s']:.1f} s")
    print(f"  Checkpoints:     {ckpt['N_ckpt']:.0f}")
    print(f"  Total ckpt time: {ckpt['T_ckpt_total_min_s'] / 3600:.2f} h ({_pct(ckpt['ckpt_fraction_of_run'])} of run)")

    print()
