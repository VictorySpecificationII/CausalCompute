# first-principles/run_012.py
from __future__ import annotations

import argparse
from pprint import pprint

from briefs.loader import load_brief_yaml

from step0_fundamentals.fundamentals import run_fundamentals, print_report
from step1_design.design import run_design
from step2_powerandthermals.powerandthermals import run_powerandthermals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSDT engine (Steps 0–2) driven by a YAML brief.")
    p.add_argument("brief", help="Path to YAML brief (e.g., briefs/13b.yaml)")
    p.add_argument("--story", action="store_true", help="Print Step 0 narrative only, then exit.")
    p.add_argument("--debug", action="store_true", help="Print full bundles (Step 0/1/2) as dicts.")
    return p.parse_args()


def _fmt_si(x: float) -> str:
    return f"{x:.6e}"


def main() -> None:
    args = parse_args()
    b = load_brief_yaml(args.brief)

    # Optional: narrative Step-0 story (great for demos)
    if args.story:
        print_report(b.w, b.st, b.io, b.dev, b.step, b.sched, b.facts, b.fabric, b.storage, b.ckpt_policy)
        return

    # ----------------------------
    # Step 0
    # ----------------------------
    bundle0 = run_fundamentals(
        w=b.w,
        st=b.st,
        io=b.io,
        dev=b.dev,
        step=b.step,
        sched=b.sched,
        facts=b.facts,
        fabric=b.fabric,
        storage=b.storage,
        ckpt_policy=b.ckpt_policy,
    )

    # ----------------------------
    # Step 1
    # ----------------------------
    design = run_design(bundle0, G=b.design_G, inputs=b.design_inputs)

    # ----------------------------
    # Step 2
    # ----------------------------
    pt = None
    if design.get("feasible") is True:
        pt = run_powerandthermals(
            design,
            power=b.pt_power,
            thermals=b.pt_thermals,
            rack=b.pt_rack,
            T_run_s=b.w.T,  # energy over the run
        )

    # ----------------------------
    # Summary (0–2)
    # ----------------------------
    print("\n" + "=" * 80)
    print("SUMMARY (Steps 0–2)")
    print("=" * 80)

    req = bundle0["req"]
    mv = bundle0["movement"]

    print("\nStep 0 — Fundamentals")
    print(f"- F_req:          {_fmt_si(float(req['F_req_flop_s']))} FLOP/s")
    print(f"- Tok rate:       {_fmt_si(float(req['R_tok_req_tok_s']))} tok/s")
    print(f"- BW dataset:     {_fmt_si(float(req['BW_dataset_plan_Bps']))} B/s")
    print(f"- S_ckpt:         {_fmt_si(float(req['S_ckpt_bytes']))} B")
    print(f"- BW_ckpt_req:    {_fmt_si(float(req['BW_ckpt_req_Bps']))} B/s")
    print(f"- t_step_max:     {_fmt_si(float(mv['t_step_max_s']))} s/step")
    print(f"- B_update/step:  {_fmt_si(float(mv['B_update_total_bytes_per_step']))} B/step")

    print("\nStep 1 — Design")
    if design.get("feasible") is not True:
        print("- infeasible")
    else:
        sol = design["solution"]
        cl = sol["cluster"]
        par = sol["parallelism"]
        tim = sol["timing"]
        mem = sol["memory"]

        print(f"- G:           {cl['G']} GPUs  ({cl['nodes']} nodes @ {cl['gpus_per_node']}/node)")
        print(f"- dp/tp/pp:     {par['dp']}/{par['tp']}/{par['pp']}")
        print(f"- t_step:       {_fmt_si(float(tim['t_step_s']))} s  (max {_fmt_si(float(tim['t_step_max_s']))} s)")
        print(f"- mem/dev:      {_fmt_si(float(mem['bytes_per_device']))} B (cap {_fmt_si(float(mem['B_dev_mem_bytes']))} B)")

        print("\nStep 2 — Power & Thermals")
        if pt is None:
            print("- skipped")
        else:
            h = pt["handoff"]
            P_IT = float(h["power"]["P_IT_W"])
            P_fac = float(h["power"]["P_facility_W"])
            Q = float(h["heat"]["Qdot_W"])
            th = h["thermals"]

            print(f"- IT power:       {_fmt_si(P_IT)} W")
            print(f"- Facility power: {_fmt_si(P_fac)} W (PUE={float(h['power']['PUE']):.2f})")
            print(f"- Heat:           {_fmt_si(Q)} W")

            if th["mode"] == "air":
                print(f"- Airflow:        {_fmt_si(float(th['vol_flow_m3_s']))} m^3/s (ΔT={float(th['deltaT_C']):.1f}°C)")
            else:
                print(f"- Coolant flow:   {_fmt_si(float(th['vol_flow_m3_s']))} m^3/s (ΔT={float(th['deltaT_C']):.1f}°C)")

            rk = h.get("rack", None)
            if rk:
                print(f"- Racks:          {rk['racks']}  (P_IT/rack {_fmt_si(float(rk['P_IT_per_rack_W']))} W)")

    print("\n" + "=" * 80)
    print("END SUMMARY")
    print("=" * 80)

    # Optional: dump full bundles
    if args.debug:
        print("\n---\nBUNDLE0 (Step 0)\n---")
        pprint(bundle0)
        print("\n---\nDESIGN (Step 1)\n---")
        pprint(design)
        print("\n---\nPOWER & THERMALS (Step 2)\n---")
        pprint(pt)


if __name__ == "__main__":
    main()

